"""Compile free-text rubric criteria into mandatory code checks.

Deterministic and conservative. Only phrasings matched below become code checks;
anything else goes to the LLM judge unchanged. A clause mixing an objective
requirement with other content is hybrid: code checks its objective part and the
judge sees the clause with those limits declared out of scope. No LLM is involved.
"""
import json
import logging
import re

log = logging.getLogger(__name__)

VERSION = 'rubric-parser-v7'   # v2: judge_scope; v3: item_unit, only_items, lead-in groups; v4: markdown tables;
                               # v5: more limit phrasings; v6: full-clause negations; v7: wrapping-quote checks

_NUMBERS = {w: str(i) for i, w in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
    'fifteen sixteen seventeen eighteen nineteen twenty'.split())}
_ITEM = (r'(?:keywords?|key phrases?|items?|lines?|bullet points?|bullets?|phrases?|headlines?|titles?|'
         r'taglines?|slogans?|entries|entry|options?|suggestions?|variants?|variations?|ideas?|names?)')
# (No hashtags/tags: "use at least 2 hashtags" means inline tags, not list lines.)
# Units that are short by nature: text after a separator is commentary, not content.
# Every other unit (headline, tagline, sentence, item, ...) is judged whole.
_SHORT_UNITS = re.compile(r'^(?:keywords?|key phrases?|phrases?|names?)$')
_UNITS = {'words': r'words?', 'characters': r'(?:characters?|chars?)'}
_N = r'(\d+)'
# Negations a limit can follow ("must not surpass", "can't go beyond", "should never exceed").
_NEG = r"(?:not|never|cannot|can't|don't|doesn't|won't|shouldn't|mustn't)"
_NEGATED = re.compile(r"\b(?:not|never|no|cannot)\b|n't\b")
# (pattern, bound) where bound maps the captured numbers to (min, max), inclusive.
_BOUNDS = [
    (rf'between {_N} and {_N} {{u}}', lambda a, b: (int(a), int(b))),
    (rf'{_N} ?(?:-|to) ?{_N} {{u}}', lambda a, b: (int(a), int(b))),
    (rf'(?:no more than|not more than|at most|up to|a maximum of|maximum of|max(?:imum)?|<=|no longer than) {_N} {{u}}', lambda n: (None, int(n))),
    (rf'(?:{_NEG} (?:exceed(?:ing)?|go(?:ing)? over|be (?:longer|more) than)|within) {_N} {{u}}', lambda n: (None, int(n))),
    (rf'{_NEG} (?:surpass(?:ing)?|go(?:ing)? (?:beyond|above)|top(?:ping)?|(?:be |go |going )?(?:over|above|beyond)|'
     rf'be (?:greater|higher) than) {_N} {{u}}', lambda n: (None, int(n))),
    (rf'{_N} {{u}} (?:max(?:imum)?|at most|tops|or under)', lambda n: (None, int(n))),
    (rf'{_N} {{u}} or (?:fewer|less)(?: each)?', lambda n: (None, int(n))),
    (rf'(?:fewer than|less than|under|below|<) {_N} {{u}}', lambda n: (None, int(n) - 1)),
    (rf'(?:no fewer than|no less than|at least|a minimum of|minimum of|min(?:imum)?|>=) {_N} {{u}}', lambda n: (int(n), None)),
    (rf'{_N} {{u}} or more', lambda n: (int(n), None)),
    (rf'(?:more than|over|>) {_N} {{u}}', lambda n: (int(n) + 1, None)),
    # Bare "exceed / surpass N" (no negation before it, see _bound): a minimum, more than N.
    (rf'(?:exceed(?:s|ing)?|surpass(?:es|ing)?|go(?:es)? beyond) {_N} {{u}}', lambda n: (int(n) + 1, None)),
    (rf'(?:exactly |precisely )?{_N} {{u}}', lambda n: (int(n), int(n))),
]
_STOP = set('''a an the each every all any per must should shall will needs need to be is are was have has had
contain contains containing include includes including consist consists of in on at total overall with only
no not than equal or and it its their they them this that these those there answer response output reply result
return returns provide provides list lists give gives generate generates write writes produce produces keep use
uses exactly precisely long length limit limited maximum minimum max min number count up so as for
please make sure ensure present presented format formatted display exceed exceeding within never
surpass surpassing beyond above over top go going cannot can don doesn won shouldn mustn t
whole entire fail fails rule'''.split())
_PLACEHOLDER = '\x00{}\x00'


def _normalize(text):
    t = text.lower().replace('≤', '<=').replace('≥', '>=').replace('–', '-')
    t = re.sub(r'\b(?:less|fewer) than or equal to\b', '<=', t)
    t = re.sub(r'\b(?:greater|more) than or equal to\b', '>=', t)
    t = re.sub(r'\b(' + '|'.join(_NUMBERS) + r')\b', lambda m: _NUMBERS[m.group(1)], t)
    return re.sub(r'\s+', ' ', t).strip()


def _protect_quotes(text):
    terms = []
    def keep(match):
        terms.append(match.group(2))
        return _PLACEHOLDER.format(len(terms) - 1)
    return re.sub(r'(["\'“‘])(.+?)(["\'”’])', keep, text), terms


def _clauses(text):
    # "between 3 and 5" must not be split on "and".
    text = re.sub(r'between (\d+) and (\d+)', r'between \1 AND_ \2', text)
    parts = re.split(r';|\.\s+|,? and (?!\d)|,? but ', text)
    return [p.replace('AND_', 'and').strip(' .,:') for p in parts if p.strip(' .,:')]


def _scope(criterion):
    """(per_item, item_scoped) for the whole criterion; clause splitting must not lose it."""
    # "each/every/per" scopes to items, except before a counting unit: "counting every character",
    # "per word" describe how to count, not a list ("the tweet is at most 280 characters, counting every
    # character" is one length for the whole answer).
    per_item = bool(re.search(r'\b(?:each|every|per)\b(?!\s+(?:single\s+)?(?:characters?|chars?|words?|letters?|'
                              r'spaces?|emojis?|tokens?|symbols?)\b)', criterion))
    return per_item, per_item or bool(re.search(rf'\b{_ITEM}\b', criterion))


def _take(clause, pattern):
    match = re.search(pattern, clause)
    if not match:
        return None, clause
    return match, clause[:match.start()] + ' ' + clause[match.end():]


# Words that make a bare "N units" a limit of some kind rather than an exact count. When one comes just
# before it and no bound pattern above recognised the phrasing ("must not surpass 200 characters"), the
# limit is left to the LLM judge: guessing "exactly N" would fail every answer that meets the limit.
_QUALIFIER = re.compile(r'\b(?:not|no|never|exceed\w*|surpass\w*|over|under|above|below|beyond|less|more|fewer|'
                        r'max\w*|min\w*|limit\w*|up|within|than|cap\w*|longer|shorter|around|about|approximately|roughly)\b')


def _bound(clause, unit):
    for n, (template, bound) in enumerate(_BOUNDS):
        match, rest = _take(clause, r'(?<!\w)' + template.format(u=unit) + r'(?!\w)')
        if match and n == len(_BOUNDS) - 1 and not re.match(r'(?:exactly|precisely) ', match.group(0)) \
                and _QUALIFIER.search(' '.join(clause[:match.start()].split()[-3:])):
            return None, clause
        if match and template.startswith('(?:exceed') and _NEGATED.search(clause[:match.start()]):
            # An intervening phrase must not turn "do not ... exceed N" into a
            # minimum. If a maximum pattern missed the phrasing, defer to the judge.
            return None, clause
        if match:
            low, high = bound(*match.groups())
            if (low is None or low >= 0) and (high is None or high >= 0) and (low is None or high is None or low <= high):
                return {'min': low, 'max': high}, rest
    return None, clause


# "Output only the keywords", "nothing else", "no other text", "one per line": the answer
# must consist of the items alone. Marker wording decides the list style; default bare lines.
_EXTRA = r'(?:introductory text|intro text|text|commentary|explanations?|preamble|notes?|intro(?:duction)?)'
_ONLY = (rf'\bonly (?:the )?(?:numbered |bulleted )?{_ITEM}\b|\bjust (?:the )?(?:numbered |bulleted )?{_ITEM}\b|'
         r'\bnothing else\b|\bno other text\b|'
         r'\bno (?:additional |extra |introductory )?(?:text|commentary|explanations?|preamble|intro(?:duction)?)\b|'
         rf"\b(?:do not|don't|never|must not|should not) (?:include|add|write|output) (?:any )?(?:additional |extra |introductory )?"
         rf'{_EXTRA}(?:(?:,\s*|\s+)(?:or |and )?(?:any )?{_EXTRA})*|'
         r'\bwithout (?:any )?(?:additional |extra )?(?:text|commentary|explanations?)\b|\b(?:one|1) per line\b')   # _normalize turns 'one' into '1'
_MARKER_STYLE = [('numbered', r'\bnumbered(?: list)?\b'), ('bulleted', r'\b(?:bulleted|bullet(?:ed)?) list\b'),
                 ('none', r'\b(?:no|without) (?:numbering|numbers|bullets|markers|list markers)\b|\bplain (?:text|lines)\b')]


def _compile_only_items(rest):
    found, rest = _take(rest, _ONLY)
    if not found:
        return None, rest
    # None = style not stated in this clause; compile_rubric merges clauses (default: bare lines).
    markers = 'numbered' if 'numbered' in found.group(0) else 'bulleted' if 'bulleted' in found.group(0) else None
    for style, pattern in [] if markers else _MARKER_STYLE:
        hit, rest2 = _take(rest, pattern)
        if hit:
            markers, rest = style, rest2
            break
    while True:   # "one per line, nothing else" names the rule twice
        more, rest2 = _take(rest, _ONLY)
        if not more:
            break
        rest = rest2
    return {'kind': 'only_items', 'markers': markers}, rest


# Quotation marks wrapping the whole answer ("Any wrapping quotes FAIL", "without any wrapper quotes",
# "must not be wrapped in quotation marks"): checked in code, where a judge is unreliable (a quote
# mark is one character). Only wrapping: "no quotes" alone could mean none anywhere, so it stays
# with the judge; so does a per-item rule ("each tagline ...").
_QUOTES = r'(?:(?:double |single )?(?:quotation marks?|quotes?))(?!\w)'
_WRAP = r'(?:wrap(?:ping|per|pers|ped|s)?|surround(?:ing|ed|s)?|enclos(?:ing|ed|es)|outer)'
_TARGET = r'(?:the |its )?(?:whole |entire |full )?(?:answer|output|response|reply|text|motto|it)'
_UNQUOTED = (rf'\b(?:without|no) (?:any )?{_WRAP} {_QUOTES}(?: (?:around|wrapping) {_TARGET})?|'
             rf'\b(?:any |all )?{_WRAP} {_QUOTES} (?:fail|fails|are not allowed|is not allowed|not allowed|are forbidden|forbidden)\b|'
             rf"\b(?:must not|should not|may not|cannot|can't|never|not|do not|don't) (?:be )?(?:{_WRAP}|put|placed|presented) "
             rf'(?:in|with|by|inside|within) (?:any )?{_QUOTES}|'
             rf"\b(?:do not|don't|never|must not|should not) {_WRAP} {_TARGET} (?:in|with) (?:any )?{_QUOTES}|"
             rf'\b{_QUOTES} (?:must not|should not|may not|never|do not|cannot) (?:wrap|surround|enclose) {_TARGET}')


def _compile_clause(clause, terms, per_item, item_scope):
    checks, rest = [], clause
    only, rest = _compile_only_items(rest)
    if only:
        checks.append(only)
    else:   # "as a numbered list": list style only, no exclusivity rule
        style, rest2 = _take(rest, r'\b(?:as |in )?(?:a )?(numbered|bulleted|bullet) list\b')
        if style:
            checks.append({'kind': 'list_style', 'markers': 'numbered' if style.group(1) == 'numbered' else 'bulleted'})
            rest = rest2
    count, rest = _bound(rest, _ITEM)
    if count:
        checks.append({'kind': 'item_count', **count})
    for unit, pattern in _UNITS.items():
        size, rest2 = _bound(rest, pattern)
        if size:
            checks.append({'kind': 'length', 'unit': unit, 'scope': 'item' if item_scope else 'answer', **size})
            rest = rest2
    for match in list(re.finditer(r'\b(?:must not|should not|may not|cannot|never|not|no|avoid|without)?\s*'
                                  r'(?:include|contain|mention|use|using|including|containing|mentioning)?s?\s*'
                                  r'(?:the )?(?:words?|phrases?|terms?|keywords?)?\s*\x00(\d+)\x00', rest)):
        negated = bool(re.search(r'\b(?:not|never|no|avoid|without)\b', match.group(0)))
        verb = re.search(r'\b(?:include|contain|mention|use|using|including|containing|mentioning|avoid|without)', match.group(0))
        if not verb:
            continue
        checks.append({'kind': 'contains', 'term': terms[int(match.group(1))], 'negate': negated,
                       'scope': 'item' if per_item else 'answer'})
        rest = rest.replace(match.group(0), ' ', 1)
    if not item_scope:
        unquoted, rest2 = _take(rest, _UNQUOTED)
        if unquoted:   # "no surrounding double quotes" bans double quotes only
            said = unquoted.group(0)
            checks.append({'kind': 'unquoted', 'quotes': 'double' if 'double' in said else 'single' if 'single' in said else 'any'})
            rest = rest2
    json_match, rest2 = _take(rest, r'\b(?:(?:be |as |in )?(?:a |an )?(?:valid )?json(?: format)?(?: (object|array|list))?)\b')
    if json_match and 'json' in json_match.group(0):
        shape = {'object': 'object', 'array': 'array', 'list': 'array'}.get(json_match.group(1), 'any')
        checks.append({'kind': 'json', 'shape': shape})
        rest = rest2
    unique, rest2 = _take(rest, r'\bno (?:duplicates?|duplicated|repeated|repeats?)\b|\b(?:be |are )?(?:all )?(?:unique|distinct)$')
    if unique and (unique.group(0).startswith('no ') or item_scope):
        checks.append({'kind': 'unique_items'})
        rest = rest2
    leftover = [w for w in re.findall(r'[a-z0-9\x00]+', rest) if w not in _STOP and not re.fullmatch(_ITEM, w)]
    return checks, bool(leftover)


_SUB_MARKER = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+')


def group_criteria(criteria):
    """[(text, source_lines)]: rejoin criteria that were written as a nested list.

    "Ensure each keyword:" followed by "- Is 5 words or fewer." arrives as separate
    lines, and alone the sub-line loses its scope ("each keyword"). A line ending
    with ':' that is followed by marked lines is a lead-in, not a criterion: each
    marked line below it becomes "<lead-in> <sub-line>". A marked line without a
    lead-in only loses its marker. A lead-in with nothing under it stays as written."""
    grouped, index = [], 0
    while index < len(criteria):
        line = str(criteria[index]).strip()
        subs = []
        while (line.endswith(':') and index + 1 + len(subs) < len(criteria)
               and _SUB_MARKER.match(str(criteria[index + 1 + len(subs)]))):
            subs.append(index + 1 + len(subs))
        if subs:
            for sub in subs:
                text = _SUB_MARKER.sub('', str(criteria[sub]), count=1).strip()
                grouped.append((f'{line} {text[:1].lower()}{text[1:]}', [index, sub]))
            index += 1 + len(subs)
            continue
        grouped.append((_SUB_MARKER.sub('', line, count=1).strip() or line, [index]))
        index += 1
    return grouped


def compile_rubric(criteria):
    """[{criterion_index, criterion, method, checks, judge_text, judge_scope, item_unit}] in criterion order.

    judge_scope is 'item' when a criterion with a qualitative part applies to each
    item ("Each keyword must be relevant..."): code extracts the items and the judge
    decides each one alone. Otherwise 'answer' (or None when nothing is judged).
    item_unit is the item noun the criterion names ("keyword", "headline"), or None.
    Nested criteria are regrouped first (see group_criteria); source_lines keeps the
    input line indexes each compiled criterion came from."""
    compiled = []
    for index, (criterion, source_lines) in enumerate(group_criteria(criteria)):
        protected, terms = _protect_quotes(criterion)
        checks, semantic = [], []
        normalized = _normalize(protected)
        normalized = re.sub(r'\((?:\s*\d+[.)],?)+\s*\)', ' ', normalized)   # "(1., 2., 3.)" is an example, not a clause
        per_item, item_scope = _scope(normalized)   # e.g. "Each tagline must include X and be <= 8 words"
        for clause in _clauses(normalized):
            found, leftover = _compile_clause(clause, terms, per_item, item_scope)
            checks += found
            if leftover or not found:
                semantic.append(clause)
        numeric = [c for c in semantic if re.search(r'\d', re.sub(r'\x00\d+\x00', ' ', c))]
        if numeric:   # a count or limit the patterns above did not recognise: worth a new phrasing
            log.info('rubric_parser: %d numeric clause(s) left to the judge', len(numeric))
        judge_text = None
        if semantic:
            judge_text = criterion if not checks else (
                f'{criterion} (Judge only the qualitative requirement. Counts, lengths, quoted terms, '
                f'format and uniqueness are verified separately in code; ignore them.)')
        method = 'deterministic' if checks and not semantic else 'hybrid' if checks else 'llm_judge'
        judge_scope = None if judge_text is None else 'item' if per_item else 'answer'
        unit = re.search(rf'\b{_ITEM}\b', normalized)
        only = [c for c in checks if c['kind'] == 'only_items']
        if only:   # several clauses may state the rule; one check, the stated style wins
            merged = {'kind': 'only_items', 'markers': next((c['markers'] for c in only if c['markers']), 'none'),
                      'unit': unit.group(0) if unit else None}   # commentary rule depends on the unit
            first = checks.index(only[0])
            checks = [c for c in checks if c['kind'] != 'only_items']
            checks.insert(first, merged)
        compiled.append({'criterion_index': index, 'criterion': criterion, 'method': method,
                         'checks': checks, 'judge_text': judge_text, 'judge_scope': judge_scope,
                         'item_unit': unit.group(0) if unit else None, 'source_lines': source_lines})
    return compiled


# ---- answer parsing ---------------------------------------------------------

_MARKER = re.compile(r'^(?:\d+[.)]|[-*•])(?:\s+|$)')
_FENCE = re.compile(r'^```[\w-]*\n(.*)\n```$', re.S)


_TABLE_ROW = re.compile(r'^\|.*\|$')
_TABLE_RULE = re.compile(r'^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?$')


_INDEX_CELL = re.compile(r'^[#\d.)\s]*$')


def _cells(line):
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def _table_items(lines):
    """Body rows of the first markdown table (header and |---| rule dropped), or None when
    the answer has no table. Each row is 'key cell | other cells': the key column is the
    one whose header names an item unit ("Keyword", "Headline"), else the first column
    that is not a row index ("#", "1"). Index cells are dropped."""
    for index in range(1, len(lines)):
        if _TABLE_RULE.match(lines[index]) and _TABLE_ROW.match(lines[index - 1]):
            header = _cells(lines[index - 1])
            body = [_cells(line) for line in lines[index + 1:] if _TABLE_ROW.match(line)]
            body = body[:next((i for i, line in enumerate(lines[index + 1:]) if not _TABLE_ROW.match(line)), len(body))]
            if not body:
                return None
            named = [i for i, name in enumerate(header) if re.fullmatch(rf'(?i){_ITEM}', _EMPHASIS.sub('', name).strip())]
            plain = [i for i in range(len(header)) if not all(_INDEX_CELL.match(row[i] if i < len(row) else '') for row in body)]
            key = named[0] if named else plain[0] if plain else 0
            rows = []
            for row in body:
                cells = [row[key]] + [c for i, c in enumerate(row) if i != key and not _INDEX_CELL.match(c)] if key < len(row) else row
                rows.append(' | '.join(c for c in cells if c))
            return [row for row in rows if row] or None
    return None


def extract_items(output):
    """(items, detected_format). JSON array of strings, else a markdown table's body rows,
    else list lines, else plain lines."""
    text = (output or '').strip()
    fenced = _FENCE.match(text)
    body = fenced.group(1).strip() if fenced else text
    try:
        value = json.loads(body)
    except ValueError:
        value = None
    else:
        if isinstance(value, list) and value and all(isinstance(v, str) and v.strip() for v in value):
            return [v.strip() for v in value], 'json_array'
        return None, 'json'
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    table = _table_items(lines)   # a table holds the items; text around it is prose
    if table:
        return table, 'table'
    marked = [line for line in lines if _MARKER.match(line)]
    chosen = marked or lines   # with a marked list, unmarked lines are prose (lead-ins, headings)
    items = [re.sub(r'^(["“])(.*)(["”])$', r'\2', _MARKER.sub('', line, count=1).strip()).strip() for line in chosen]
    if not items or not all(items):
        return None, 'lines'
    return items, 'marked_lines' if marked else 'lines'


_EMPHASIS = re.compile(r'[*_`]+')
_SEPARATOR = r'\s*(?:[—–:(|]|\s-\s)'
# A leading span counts only when the item ends there or a separator follows it.
_QUOTED_HEAD = re.compile(rf'^[*_`\s]*["“](.+?)["”][*_`]*(?=\s*$|{_SEPARATOR})')
_BOLD_HEAD = re.compile(rf'^\s*(\*\*|__)(.+?)\1(?=\s*$|{_SEPARATOR})')
_HEAD_SEPARATOR = re.compile(r'\s+[—–|-]\s+')   # not ':' — 'Keyword 1: x' would yield the label


def _unwrap(text):
    """Strip emphasis/quotes that wrap the whole text; inner punctuation is content."""
    previous = None
    while text != previous:
        previous = text
        text = text.strip()
        for left, right in (('**', '**'), ('__', '__'), ('*', '*'), ('_', '_'), ('`', '`'), ('"', '"'), ('“', '”')):
            if len(text) > len(left) + len(right) and text.startswith(left) and text.endswith(right):
                text = text[len(left):-len(right)]
    return text


_WRAP_QUOTES = (('"', '"'), ('“', '”'), ("'", "'"), ('‘', '’'))
_QUOTE_KIND = {'"': 'double', '“': 'double', "'": 'single', '‘': 'single'}
# Part of the content, not a quotation mark: an apostrophe inside a word (don't, it’s) or a plural
# possessive before a space (customers' choice); and any escaped character (\" inside "...").
_APOSTROPHE = re.compile(r"(?<=\w)['’](?=\w)|(?<=s)['’](?=\s)")
_ESCAPED = re.compile(r'\\.')


def _wrapping(text):
    """The kind of quotation marks wrapping all of the (stripped) text, 'double' or 'single', or None."""
    for left, right in _WRAP_QUOTES:
        inner = _ESCAPED.sub('', text[1:-1])
        if left in "'‘":
            inner = _APOSTROPHE.sub('', inner)
        backslashes = len(text[:-1]) - len(text[:-1].rstrip('\\'))
        if len(text) >= 2 and text.startswith(left) and text.endswith(right) and backslashes % 2 == 0 \
                and left not in inner and (left == right or right not in inner):
            return _QUOTE_KIND[left]
    return None


def _unquote(text):
    """The answer without quotation marks that wrap all of it ("Save 40% on inference" -> Save 40% on
    inference). A length limit is about the content: wrapping quotes are presentation, so they are never
    counted. Quotes inside the answer, or around only part of it, are content and stay."""
    text = text.strip()
    return text[1:-1].strip() if _wrapping(text) else text


def is_short_unit(unit):
    return bool(unit) and bool(_SHORT_UNITS.match(unit))


def extract_judged_text(item, unit=None):
    """(text, rule): what the item judge sees for one item.

    Short units (keyword, key phrase, phrase, name): a leading quoted or bold span
    followed by the end or a separator, else the text before a spaced dash, else the
    whole item; commentary is left out. Any other unit (headline, tagline, sentence,
    or unnamed): the whole item, only unwrapped, because a dash or colon there is
    content ("Save 20% — today only"). Length and format checks always measure the
    whole item; this only scopes the judge. A markdown table row ('cell | cell') is judged
    on its first cell for any unit: the columns make the structure explicit."""
    if ' | ' in item:
        first = _unwrap(item.split(' | ', 1)[0])
        if first:
            return _EMPHASIS.sub('', first).strip(' "“”'), 'cell'
    if is_short_unit(unit):
        for rule, pattern, group in (('quoted', _QUOTED_HEAD, 1), ('bold', _BOLD_HEAD, 2)):
            match = pattern.match(item)
            if match and match.group(group).strip():
                return _EMPHASIS.sub('', match.group(group)).strip(' "“”'), rule
        parts = _HEAD_SEPARATOR.split(item, maxsplit=1)
        head = _unwrap(parts[0])
        if len(parts) == 2 and head:
            return head, 'head'
    return _unwrap(item) or item, 'whole'


def _within(value, check):
    return (check['min'] is None or value >= check['min']) and (check['max'] is None or value <= check['max'])


def _describe(check):
    low, high = check['min'], check['max']
    return (f'exactly {low}' if low == high else f'between {low} and {high}' if low is not None and high is not None
            else f'at most {high}' if high is not None else f'at least {low}')


def _term_in(term, text):
    t = re.escape(term.lower())
    return re.search((r'\b' if term[:1].isalnum() else '') + t + (r'\b' if term[-1:].isalnum() else ''), text.lower()) is not None


_NUMBERED = re.compile(r'^\d+[.)]\s+')
_BULLETED = re.compile(r'^[-*•]\s+')
_WRAPPED = re.compile(r'^(?:\*\*|__|[*_`"“]).*(?:\*\*|__|[*_`"”])$')


def _list_style(output, check):
    """Every list line uses the requested marker; numbered lists count 1, 2, 3, ...
    Text around the list is not judged here (that is only_items)."""
    text = output.strip()
    try:
        json.loads(text)
        return False, 'The answer is JSON, not a list.', {'format': 'json'}
    except ValueError:
        pass
    items = [line.strip() for line in text.splitlines() if _MARKER.match(line.strip())]
    style = check['markers']
    if not items:
        return False, f'No {style} list found.', {'format': 'lines', 'items': 0}
    pattern = _NUMBERED if style == 'numbered' else _BULLETED
    wrong = [n for n, line in enumerate(items, 1) if not pattern.match(line)]
    if style == 'numbered' and not wrong:
        numbers = [int(re.match(r'\d+', line).group(0)) for line in items]
        if numbers != list(range(1, len(items) + 1)):
            return False, f"Numbered items are {', '.join(map(str, numbers))}; expected 1 to {len(items)}.", \
                {'format': 'lines', 'items': len(items), 'numbers': numbers}
    if wrong:
        return False, f"{len(wrong)} of {len(items)} list items are not {style}.", {'format': 'lines', 'items': len(items)}
    return True, f'All {len(items)} list items are {style}.', {'format': 'lines', 'items': len(items)}


def _only_items(output, check):
    """Every non-empty line is one item and nothing else: no lead-in or closing text,
    no emphasis or wrapping quotes, the requested list style, and for short units
    (keywords, names, ...) no commentary after a dash or colon. A JSON array of
    strings is clean unless a numbered or bulleted list was asked for."""
    style = check['markers']
    text = output.strip()
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if value is not None:
        clean = (isinstance(value, list) and value and all(isinstance(v, str) and v.strip() for v in value)
                 and style == 'none')
        note = 'The answer is a JSON array of strings.' if clean else 'The answer is JSON, not the requested list.'
        return bool(clean), note, {'format': 'json', 'violations': [] if clean else [{'line': None, 'reason': 'json'}]}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short = is_short_unit(check.get('unit'))
    violations = []
    for number, line in enumerate(lines, 1):
        body, reason = line, None
        if style == 'numbered':
            reason = None if _NUMBERED.match(line) else 'not a numbered item'
            body = _NUMBERED.sub('', line, count=1)
        elif style == 'bulleted':
            reason = None if _BULLETED.match(line) else 'not a bulleted item'
            body = _BULLETED.sub('', line, count=1)
        elif _MARKER.match(line):
            reason = 'list marker'
        if reason is None and line.startswith('```'):
            reason = 'code fence'
        elif reason is None and line.startswith('|'):
            reason = 'table'
        elif reason is None and body.rstrip().endswith(':'):
            reason = 'lead-in text'
        elif reason is None and ('**' in body or '__' in body or '`' in body or _WRAPPED.match(body)):
            reason = 'emphasis or quotes'
        elif reason is None and short and (_HEAD_SEPARATOR.search(body) or re.search(r':\s', body)):
            reason = 'commentary'
        if reason:
            violations.append({'line': number, 'reason': reason, 'text': line[:80]})
    wanted = {'none': 'bare lines', 'numbered': 'numbered items', 'bulleted': 'bulleted items'}[style]
    if not lines:
        return False, 'The answer is empty.', {'format': 'lines', 'lines': 0, 'violations': []}
    if not violations:
        return True, f'All {len(lines)} lines are {wanted}; no other text.', {'format': 'lines', 'lines': len(lines), 'violations': []}
    shown = '; '.join(f"line {v['line']} ({v['reason']}): {v['text']}" for v in violations[:3])
    more = f' (+{len(violations) - 3} more)' if len(violations) > 3 else ''
    return False, f'Required {wanted} and nothing else. {len(violations)} of {len(lines)} lines violate it: {shown}{more}.', \
        {'format': 'lines', 'lines': len(lines), 'violations': violations}


def run_check(output, check):
    """One check -> {passed, explanation, observed}. Never raises on model output."""
    kind = check['kind']
    if kind == 'json':
        try:
            value = json.loads((output or '').strip())
        except ValueError:
            return False, 'The answer is not valid JSON.', {'json_valid': False}
        ok = check['shape'] == 'any' or isinstance(value, dict if check['shape'] == 'object' else list)
        return ok, f"Valid JSON{'' if ok else ', but not a JSON ' + check['shape']}.", {'json_valid': True}
    if kind == 'only_items':
        return _only_items(output or '', check)
    if kind == 'unquoted':
        found, banned = _wrapping((output or '').strip()), check.get('quotes', 'any')
        wrapped = found is not None and banned in ('any', found)
        name = 'quotation marks' if banned == 'any' else f'{banned} quotation marks'
        return (not wrapped, f'The answer is wrapped in {name}.' if wrapped
                else f'The answer is not wrapped in {name}.', {'wrapped_in_quotes': found})
    if kind == 'list_style':
        return _list_style(output or '', check)
    if kind == 'length' and check['scope'] == 'answer':
        text = _unquote(output or '')
        size = len(text.split()) if check['unit'] == 'words' else len(text)
        note = ' (quotation marks wrapping the whole answer not counted)' if len(text) != len((output or '').strip()) else ''
        return _within(size, check), f"Answer has {size} {check['unit']}{note}; required {_describe(check)}.", {check['unit']: size}
    if kind == 'contains' and check['scope'] == 'answer':
        found = _term_in(check['term'], output or '')
        ok = found != check['negate']
        return ok, f"\"{check['term']}\" {'found' if found else 'not found'}; it must {'not ' if check['negate'] else ''}appear.", {'found': found}
    items, fmt = extract_items(output)
    if items is None:
        return False, 'No list items found (expected a JSON array of strings or one item per line).', {'format': fmt, 'items': None}
    observed = {'format': fmt, 'items': len(items)}
    if kind == 'item_count':
        return _within(len(items), check), f'Observed {len(items)} items; required {_describe(check)}.', observed
    if kind == 'length':
        sizes = [len([w for w in i.split() if w != '|']) if check['unit'] == 'words' else len(i) for i in items]
        observed[f"{check['unit']}_per_item"] = sizes
        note = ' Words are whitespace-separated; hyphenated text counts as one word.' if check['unit'] == 'words' else ''
        return all(_within(s, check) for s in sizes), f"{check['unit'].capitalize()} per item: {', '.join(map(str, sizes))}; required {_describe(check)}.{note}", observed
    if kind == 'contains':
        hits = [_term_in(check['term'], i) for i in items]
        observed['found_per_item'] = hits
        ok = all(h != check['negate'] for h in hits)
        return ok, f"\"{check['term']}\" found in {sum(hits)} of {len(items)} items; it must {'appear in none' if check['negate'] else 'appear in every item'}.", observed
    if kind == 'unique_items':
        keys = [re.sub(r'\s+', ' ', i.lower()) for i in items]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        observed['duplicates'] = dupes
        return not dupes, ('All items are distinct.' if not dupes else f'Duplicate items: {", ".join(dupes)}.'), observed
    raise ValueError(f'Unknown check kind {kind!r}')
