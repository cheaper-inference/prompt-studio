"""Offline verification for prompt_optimizer.

Exercises every building block against the real MLflow prompt registry, without
making any network call to the model endpoint. The one thing it cannot do is run
the actual optimizer (that needs the live reflection/judge models); everything
that leads up to and follows that call is checked here.

    MLFLOW_TRACKING_URI=sqlite:////tmp/po_test.db python -m test.test_prompt_optimizer
"""

from __future__ import annotations

import os
import sys
import types
from unittest import mock

# This module lives in the test/ package; make the project root importable so
# `import prompt_optimizer` resolves whether run as a script or as -m test.….
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Registry needs a SQL store; use a throwaway sqlite db.
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:////tmp/po_test.db")

import mlflow  # noqa: E402
from mlflow.entities.model_registry import PromptVersion  # noqa: E402
from mlflow.genai import scorers as S  # noqa: E402
from mlflow.genai.optimize import GepaPromptOptimizer, MetaPromptOptimizer  # noqa: E402

import prompt_optimizer as po  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ---------------------------------------------------------------------------
# A fake endpoint: records the chat calls, never touches the network.
# ---------------------------------------------------------------------------
class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        joined = " | ".join(m["content"] for m in messages)
        msg = types.SimpleNamespace(content=f"OUT<{joined}>")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeEndpoint:
    provider = "openai"
    task_model = "glm-5.1"
    judge_model = "glm-5.1"
    reflection_model = "glm-5.1"
    base_url = "https://fake.endpoint/v1"
    api_key = "sk-fake"

    def __init__(self) -> None:
        self._client = FakeClient()

    @property
    def client(self):
        return self._client

    def uri(self, model):
        return f"{self.provider}:/{model}"

    def lm_kwargs(self, **extra):
        return {"api_base": self.base_url, "api_key": self.api_key, **extra}

    def gepa_kwargs(self, **extra):
        reflection = {"api_base": self.base_url, "api_key": self.api_key}
        reflection.update(extra.pop("reflection_lm_kwargs", {}))
        return {"reflection_lm_kwargs": reflection, **extra}

    def preflight(self, verbose=False):
        return []


def make_opt():
    return po.PromptOptimizer(FakeEndpoint())


# ---------------------------------------------------------------------------
# 1. Prompt coercion
# ---------------------------------------------------------------------------
def test_coerce_prompts():
    a = po._coerce_prompts("hello {{x}}")
    check("coerce str -> single user prompt", len(a) == 1 and a[0].role == "user")
    b = po._coerce_prompts([{"role": "system", "template": "s"}, po.Prompt("u", "user")])
    check("coerce mixed list", len(b) == 2 and b[0].role == "system" and b[1].template == "u")
    check("p_variables finds double-brace vars",
          po.p_variables("a {{foo}} b {{ bar }}") == {"foo", "bar"})


# ---------------------------------------------------------------------------
# 2. Config validation contracts
# ---------------------------------------------------------------------------
def test_validation():
    def raises(cfg):
        try:
            cfg.validate()
            return False
        except po.ConfigError:
            return True

    check("training w/o data rejected",
          raises(po.OptimizationConfig("t", "p {{s}}", po.Mode.TRAINING_DATA)))
    check("few-shot w/o criteria rejected",
          raises(po.OptimizationConfig("t", "p {{s}}", po.Mode.FEW_SHOT,
                                       examples=[{"s": "x"}])))
    check("bad label_scorer rejected",
          raises(po.OptimizationConfig("t", "p {{s}}", po.Mode.TRAINING_DATA,
                                       training_data=[{"s": "x", "expected_output": "y"}],
                                       label_scorer="nope")))
    check("unfillable template var rejected",
          raises(po.OptimizationConfig("t", "p {{missing}}", po.Mode.TRAINING_DATA,
                                       training_data=[{"s": "x", "expected_output": "y"}],
                                       input_keys=["s"])))
    check("missing target column rejected",
          raises(po.OptimizationConfig("t", "p {{s}}", po.Mode.TRAINING_DATA,
                                       training_data=[{"s": "x"}], input_keys=["s"],
                                       target_key="label")))
    # valid ones should not raise
    ok = po.OptimizationConfig("t", "p {{s}}", po.Mode.TRAINING_DATA,
                               training_data=[{"s": "x", "label": "y"}],
                               input_keys=["s"], target_key="label")
    ok.validate()
    check("valid training config accepted", True)
    zs = po.OptimizationConfig("t", "p", po.Mode.ZERO_SHOT, task_description="do a thing")
    zs.validate()
    check("valid zero-shot config accepted", True)


# ---------------------------------------------------------------------------
# 3. train_data normalization
# ---------------------------------------------------------------------------
def test_train_data():
    opt = make_opt()
    cfg = po.OptimizationConfig("t", "p {{s}}", po.Mode.TRAINING_DATA,
                                training_data=[{"s": "hi", "label": "A"}],
                                input_keys=["s"], target_key="label")
    rows = opt._build_train_data(cfg)
    check("training row shape",
          rows == [{"inputs": {"s": "hi"}, "expectations": {"expected_response": "A"}}],
          str(rows))

    cfg2 = po.OptimizationConfig("t", [po.Prompt("{{service}} {{location}}", "system")],
                                 po.Mode.FEW_SHOT, examples=[{"service": "uc", "location": "TX"}],
                                 input_keys=["service", "location"], quality_criteria="be good")
    rows2 = opt._build_train_data(cfg2)
    check("few-shot row keeps inputs, no expectations",
          rows2 == [{"inputs": {"service": "uc", "location": "TX"}}], str(rows2))

    check("zero-shot has empty train_data",
          opt._build_train_data(po.OptimizationConfig("t", "p", po.Mode.ZERO_SHOT)) == [])

    # pass-through of already-normalized rows
    cfg3 = po.OptimizationConfig("t", "p {{s}}", po.Mode.TRAINING_DATA,
                                 training_data=[{"inputs": {"s": "x"},
                                                 "expectations": {"expected_response": "Z"}}])
    check("pre-normalized row passed through",
          opt._build_train_data(cfg3)[0]["expectations"]["expected_response"] == "Z")

    # static prompt (no {{variables}}) few-shot: inputs must NOT be empty
    cfg4 = po.OptimizationConfig("t", "Generate keywords.", po.Mode.FEW_SHOT,
                                 examples=[{"expected_output": "kw one\nkw two"}],
                                 quality_criteria="be good")
    rows4 = opt._build_train_data(cfg4)
    check("static-prompt few-shot injects a non-empty input (no MLflow crash)",
          bool(rows4[0]["inputs"]) and rows4[0]["expectations"]["expected_response"].startswith("kw"),
          str(rows4))
    # gold folding happens upstream of the judge; check it via the built-in path,
    # which exposes .guidelines (the default now uses our custom scorer instead).
    cfg4b = po.OptimizationConfig("t", "Generate keywords.", po.Mode.FEW_SHOT,
                                  examples=[{"expected_output": "kw one\nkw two"}],
                                  quality_criteria="be good", use_builtin_judge=True)
    sc4 = opt._build_scorers(cfg4b)
    folded = any("examples of good output" in g.lower()
                 for g in (sc4[0].guidelines if isinstance(sc4[0].guidelines, list) else [sc4[0].guidelines]))
    check("static-prompt few-shot folds gold examples into judge guidelines", folded)

    # validation: few-shot with neither criteria nor gold is rejected clearly
    try:
        po.OptimizationConfig("t", "Generate.", po.Mode.FEW_SHOT,
                              examples=[{"note": "x"}]).validate()
        check("few-shot with no criteria and no gold rejected", False)
    except po.ConfigError:
        check("few-shot with no criteria and no gold rejected", True)

    # validation: training with a static prompt (no variable) is rejected clearly
    try:
        po.OptimizationConfig("t", "Classify it.", po.Mode.TRAINING_DATA,
                              training_data=[{"expected_output": "A"}]).validate()
        check("training with no {{variable}} rejected", False)
    except po.ConfigError:
        check("training with no {{variable}} rejected", True)


# ---------------------------------------------------------------------------
# 4. scorer + optimizer selection per door
# ---------------------------------------------------------------------------
def test_scorers_and_optimizer():
    opt = make_opt()

    td = po.OptimizationConfig("t", "p {{s}}", po.Mode.TRAINING_DATA,
                               training_data=[{"s": "x", "label": "y"}],
                               input_keys=["s"], target_key="label", label_scorer="correctness")
    sc = opt._build_scorers(td)
    check("training default -> custom label judge (no MLflow gateway)",
          len(sc) == 1 and getattr(sc[0], "name", "") == "label_match"
          and not isinstance(sc[0], S.Correctness))

    td.label_scorer = "equivalence"
    eq = opt._build_scorers(td)[0]
    check("training equivalence -> custom label judge", getattr(eq, "name", "") == "label_match")

    # opting into MLflow's built-in label judge still works
    td.use_builtin_judge = True
    b = opt._build_scorers(td)[0]
    check("training builtin -> Equivalence", isinstance(b, S.Equivalence))
    check("builtin judge runs in JSON mode (parse-safe verdict)",
          (getattr(b, "inference_params", {}) or {}).get("response_format") == {"type": "json_object"},
          str(getattr(b, "inference_params", None)))

    fs = po.OptimizationConfig("t", "p {{s}}", po.Mode.FEW_SHOT, examples=[{"s": "x"}],
                               input_keys=["s"], quality_criteria=["a", "b"])
    fsc = opt._build_scorers(fs)
    check("few-shot default -> robust custom guidelines judge",
          len(fsc) == 1 and getattr(fsc[0], "name", "") == "guidelines"
          and not isinstance(fsc[0], S.Guidelines))
    # opting into the built-in judge still works
    fs_b = po.OptimizationConfig("t", "p {{s}}", po.Mode.FEW_SHOT, examples=[{"s": "x"}],
                                 input_keys=["s"], quality_criteria=["a", "b"], use_builtin_judge=True)
    check("few-shot use_builtin_judge -> MLflow Guidelines",
          isinstance(opt._build_scorers(fs_b)[0], S.Guidelines))

    fs2 = po.OptimizationConfig("t", "p {{s}}", po.Mode.FEW_SHOT, examples=[{"s": "x", "guidelines": "g"}],
                                input_keys=["s"], per_row_criteria=True)
    check("few-shot per-row -> ExpectationsGuidelines",
          isinstance(opt._build_scorers(fs2)[0], S.ExpectationsGuidelines))

    zs = po.OptimizationConfig("t", "p", po.Mode.ZERO_SHOT)
    check("zero-shot -> no scorers", opt._build_scorers(zs) == [])

    # custom scorers: instance + built-in name that needs no required args
    custom = po.OptimizationConfig("t", "p {{s}}", po.Mode.TRAINING_DATA,
                                   training_data=[{"s": "x", "label": "y"}],
                                   input_keys=["s"], target_key="label",
                                   custom_scorers=[S.Safety(), "RelevanceToQuery"])
    cs = opt._build_scorers(custom)
    check("custom scorers: instance + resolved-by-name both present",
          len(cs) == 3 and any(isinstance(x, S.Safety) for x in cs)
          and any(isinstance(x, S.RelevanceToQuery) for x in cs), str([type(x).__name__ for x in cs]))

    # a name that needs required config must fail with a clear ConfigError
    bad = po.OptimizationConfig("t", "p {{s}}", po.Mode.TRAINING_DATA,
                                training_data=[{"s": "x", "label": "y"}],
                                input_keys=["s"], target_key="label",
                                custom_scorers=["Guidelines"])
    try:
        opt._build_scorers(bad)
        check("under-specified scorer name raises ConfigError", False)
    except po.ConfigError:
        check("under-specified scorer name raises ConfigError", True)

    check("training -> GepaPromptOptimizer",
          isinstance(opt._build_optimizer(td), GepaPromptOptimizer))
    check("zero-shot -> MetaPromptOptimizer",
          isinstance(opt._build_optimizer(zs), MetaPromptOptimizer))

    gepa = opt._build_optimizer(td)
    gk = gepa.gepa_kwargs
    check("gepa_kwargs carries endpoint creds + temperature",
          gk["reflection_lm_kwargs"]["api_base"] == "https://fake.endpoint/v1"
          and gk["reflection_lm_kwargs"]["temperature"] == 1.0
          and gk["candidate_selection_strategy"] == "pareto", str(gk))
    check("reflection lm caps reasoning + timeout",
          gk["reflection_lm_kwargs"].get("reasoning_effort") == "low"
          and gk["reflection_lm_kwargs"].get("timeout") == 90, str(gk["reflection_lm_kwargs"]))


# ---------------------------------------------------------------------------
# 5. register + predict_fn end-to-end against the real registry,
#    including MLflow's name-keyed candidate injection.
# ---------------------------------------------------------------------------
def test_predict_fn_and_injection():
    opt = make_opt()
    cfg = po.OptimizationConfig(
        "unit_predict_demo",
        [po.Prompt("SYS {{service}} in {{location}}", "system"), po.Prompt("Go!", "user")],
        po.Mode.FEW_SHOT, examples=[{"service": "uc", "location": "TX"}],
        input_keys=["service", "location"], quality_criteria="good",
    )
    registered = opt._register_prompts(cfg)
    check("two prompts registered", len(registered) == 2)
    check("multi-prompt names are unique",
          registered[0]["name"] != registered[1]["name"]
          and registered[0]["name"].endswith("_system"))

    predict_fn = opt._build_predict_fn(registered, "glm-5.1", cfg)

    # 5a. baseline: format substitutes only declared vars; extra inputs ignored
    out = predict_fn(service="urgent care", location="Austin", junk="ignore me")
    calls = opt.endpoint.client.calls
    msgs = calls[-1]["messages"]
    check("predict_fn passes reasoning_effort + timeout to the task call",
          calls[-1]["kwargs"].get("reasoning_effort") == "minimal"
          and calls[-1]["kwargs"].get("timeout") == 90, str(calls[-1]["kwargs"]))
    check("message roles/order preserved",
          [m["role"] for m in msgs] == ["system", "user"], str(msgs))
    check("declared vars filled, undeclared ignored",
          msgs[0]["content"] == "SYS urgent care in Austin" and msgs[1]["content"] == "Go!",
          str(msgs))
    check("predict_fn returns model output string", out.startswith("OUT<"), out)

    # 5b. candidate injection: patch PromptVersion.template by name, as MLflow does
    sys_name = registered[0]["name"]
    candidates = {sys_name: "CANDIDATE {{service}} @ {{location}}"}
    orig = PromptVersion.template

    def _patched(self):
        if self.name in candidates:
            return candidates[self.name]
        return orig.fget(self)

    with mock.patch.object(PromptVersion, "template", property(_patched)):
        predict_fn(service="uc", location="TX")
    inj_msgs = opt.endpoint.client.calls[-1]["messages"]
    check("candidate text is picked up via name-keyed patch",
          inj_msgs[0]["content"] == "CANDIDATE uc @ TX", str(inj_msgs))


# ---------------------------------------------------------------------------
# 6. result mapping + deployable shape
# ---------------------------------------------------------------------------
def test_result_mapping():
    opt = make_opt()
    cfg = po.OptimizationConfig(
        "resmap",
        [po.Prompt("SYS {{s}}", "system"), po.Prompt("U", "user")],
        po.Mode.FEW_SHOT, examples=[{"s": "x"}], input_keys=["s"], quality_criteria="g",
    )
    registered = opt._register_prompts(cfg)

    # fake optimizer result: PromptVersion-like objects keyed by the same names
    fake_prompts = [
        types.SimpleNamespace(name=registered[0]["name"], template="OPT SYS {{s}}", variables=["s"]),
        types.SimpleNamespace(name=registered[1]["name"], template="OPT U", variables=[]),
    ]
    fake_result = types.SimpleNamespace(
        optimized_prompts=fake_prompts, optimizer_name="GepaPromptOptimizer",
        initial_eval_score=0.4, final_eval_score=0.9,
        initial_eval_score_per_scorer={"guidelines": 0.4},
        final_eval_score_per_scorer={"guidelines": 0.9},
    )
    res = opt._build_result(cfg, registered, "glm-5.1", fake_result)
    check("result maps optimized templates back to roles",
          res.prompts[0].role == "system" and res.prompts[0].template == "OPT SYS {{s}}"
          and res.prompts[1].role == "user", res.summary())
    check("improvement computed", abs(res.improvement - 0.5) < 1e-9)

    dep = res.to_deployable()
    check("deployable has model + ordered messages",
          dep["model"] == "glm-5.1"
          and [m["role"] for m in dep["messages"]] == ["system", "user"]
          and dep["messages"][0]["content"] == "OPT SYS {{s}}", str(dep))
    check("deployable collects variables", dep["variables"] == ["s"], str(dep["variables"]))
    import json
    json.loads(res.to_json())   # must be serializable
    check("result serializes to json", True)


def test_output_constraints():
    C = po.check_output_constraints
    # json + required fields
    check("json ok with fields", C('{"a":1,"b":2}', {"format":"json","json_fields":["a","b"]}))
    check("json missing field fails", not C('{"a":1}', {"format":"json","json_fields":["a","b"]}))
    check("json invalid fails", not C('not json', {"format":"json"}))
    # lines / csv
    check("lines ok", C("one\ntwo", {"format":"lines"}))
    check("lines empty fails", not C("   ", {"format":"lines"}))
    check("csv single line ok", C("a,b,c", {"format":"csv"}))
    check("csv multiline fails", not C("a,b\nc,d", {"format":"csv"}))
    # enum (case-insensitive, whole answer)
    enum = {"allowed_values":["METHODS","RESULTS"]}
    check("enum match (case-insensitive)", C("methods", enum))
    check("enum non-member fails", not C("foobar", enum))
    check("enum skipped when format=json", C('{"x":1}', {"format":"json","allowed_values":["A"]}))
    # instruction text
    instr = po.constraint_instruction({"format":"json","json_fields":["kw"],"allowed_values":["A"]})
    check("instruction mentions JSON + keys", "JSON" in instr and "kw" in instr)
    check("empty constraints -> empty instruction", po.constraint_instruction(None) == "")
    # validation rejects a bad format
    try:
        po.OptimizationConfig("t","p {{s}}",po.Mode.TRAINING_DATA,
                              training_data=[{"s":"x","expected_output":"y"}],input_keys=["s"],
                              output_constraints={"format":"xml"}).validate()
        check("bad constraint format rejected", False)
    except po.ConfigError:
        check("bad constraint format rejected", True)


def test_constraints_wire_into_scorer_and_prompt():
    opt = make_opt()
    cfg = po.OptimizationConfig("t","p {{s}}",po.Mode.TRAINING_DATA,
        training_data=[{"s":"x","label":"METHODS"}], input_keys=["s"], target_key="label",
        output_constraints={"allowed_values":["METHODS","RESULTS"]})
    scs = opt._build_scorers(cfg)
    check("constraint scorer appended", any(getattr(s,"name","")=="output_constraints" for s in scs),
          str([getattr(s,"name","?") for s in scs]))
    # the constraint instruction is folded into the registered prompt
    reg = opt._register_prompts(po.OptimizationConfig("t2",
        [po.Prompt("You classify {{s}}.","system"), po.Prompt("Go","user")],
        po.Mode.TRAINING_DATA, training_data=[{"s":"x","label":"A"}], input_keys=["s"], target_key="label",
        output_constraints={"allowed_values":["A","B"]}))
    import mlflow
    sys_tmpl = mlflow.genai.load_prompt(reg[0]["uri"]).template
    check("constraint clause folded into system prompt",
          "Output requirements" in sys_tmpl and "one of: A, B" in sys_tmpl, sys_tmpl[-120:])


def test_tighten_pass():
    import types as _t
    def make_ep(reply):
        ep = FakeEndpoint()
        cli = _t.SimpleNamespace()
        def create(model, messages, **kw):
            return _t.SimpleNamespace(choices=[_t.SimpleNamespace(message=_t.SimpleNamespace(content=reply))])
        cli.completions = _t.SimpleNamespace(create=create)
        cli.chat = _t.SimpleNamespace(completions=cli.completions)
        ep._client = cli
        return ep
    orig = "Please carefully and verbosely generate the answer for {{s}}, thinking hard about it."
    # good compression: shorter and keeps {{s}}
    opt = po.PromptOptimizer(make_ep("Answer for {{s}}."))
    res = po.OptimizationResult("n","few_shot","GEPA","openai","glm-5.1",
        [po.OptimizedPrompt("n","user",orig,["s"])])
    opt._tighten(po.OptimizationConfig("n","x {{s}}",po.Mode.FEW_SHOT,examples=[{"s":"a"}],quality_criteria="g",tighten_prompt=True), res)
    check("tighten applies shorter, variable-preserving prompt",
          res.tightened and res.prompts[0].template == "Answer for {{s}}.", res.prompts[0].template)
    # bad compression: drops the {{s}} variable -> must revert
    opt2 = po.PromptOptimizer(make_ep("Answer."))
    res2 = po.OptimizationResult("n","few_shot","GEPA","openai","glm-5.1",
        [po.OptimizedPrompt("n","user",orig,["s"])])
    opt2._tighten(po.OptimizationConfig("n","x {{s}}",po.Mode.FEW_SHOT,examples=[{"s":"a"}],quality_criteria="g",tighten_prompt=True), res2)
    check("tighten reverts when a variable is lost",
          (not res2.tightened) and res2.prompts[0].template == orig)


def test_extract_json():
    ej = po.extract_json
    check("extract pure json", ej('{"pass": true}') == {"pass": True})
    check("extract fenced json", ej('```json\n{"pass": false}\n```') == {"pass": False})
    check("extract prose-then-json (the glm-5.1 case)",
          ej('Let\'s think step by step. 1. yes 2. yes\n{"pass": true, "rationale": "ok"}') ==
          {"pass": True, "rationale": "ok"})
    check("extract json-then-trailing-prose",
          ej('{"result": "yes"}\nThat is my answer.') == {"result": "yes"})
    check("extract nested braces", ej('noise {"a": {"b": 1}} tail')["a"] == {"b": 1})
    check("no json -> None", ej("there is no json here") is None)


def test_verdict_to_score():
    v = po.verdict_to_score
    check("per_guideline partial credit", v({"per_guideline": [True, True, False, True]}, 4) == 0.75)
    check("per_guideline all true", v({"per_guideline": [True, True]}, 2) == 1.0)
    check("key-agnostic: verdict=yes", v({"verdict": "yes"}, 4) == 1.0)
    check("key-agnostic: passed=false", v({"passed": False}, 4) == 0.0)
    check("key-agnostic: compliant=true", v({"compliant": True}, 4) == 1.0)
    check("legacy pass key still works", v({"pass": True}, 4) == 1.0)
    check("empty/None -> 0", v({}, 4) == 0.0 and v(None, 4) == 0.0)
    check("results[] alias", v({"results": [True, False]}, 2) == 0.5)


def test_robust_guidelines_judge():
    import types as _t

    def scripted_endpoint(replies):
        ep = FakeEndpoint()
        seq = list(replies)
        calls = []

        def create(model, messages, **kw):
            calls.append({"messages": messages, "kwargs": kw})
            content = seq[min(len(calls) - 1, len(seq) - 1)]
            return _t.SimpleNamespace(choices=[_t.SimpleNamespace(
                message=_t.SimpleNamespace(content=content))])
        comp = _t.SimpleNamespace(create=create)
        ep._client = _t.SimpleNamespace(chat=_t.SimpleNamespace(completions=comp),
                                        completions=comp, _calls=calls)
        return ep, calls

    cfg = po.OptimizationConfig("t", "p {{s}}", po.Mode.FEW_SHOT, examples=[{"s": "x"}],
                                input_keys=["s"], quality_criteria=["be concise"])

    # by default (use_builtin_judge=False) few-shot uses OUR scorer, not MLflow's Guidelines
    from mlflow.genai import scorers as S
    opt0 = make_opt()
    sc = opt0._build_scorers(cfg)[0]
    check("few-shot default -> robust custom judge (not MLflow Guidelines)",
          getattr(sc, "name", "") == "guidelines" and not isinstance(sc, S.Guidelines))

    # clean JSON verdict -> pass
    ep, _ = scripted_endpoint(['{"pass": true, "rationale": "meets all"}'])
    j = po.PromptOptimizer(ep)._build_guidelines_scorer(cfg, ["be concise"])
    check("robust judge: clean pass", j(inputs={"s": "x"}, outputs="short") == 1.0)

    # the exact failing shape: narration THEN valid JSON -> still parses (fail verdict)
    ep2, _ = scripted_endpoint(
        ['Let\'s think step by step. The response is too long.\n{"pass": false, "rationale": "verbose"}'])
    j2 = po.PromptOptimizer(ep2)._build_guidelines_scorer(cfg, ["be concise"])
    check("robust judge: parses prose-then-json (MLflow would fail here)",
          j2(inputs={"s": "x"}, outputs="long...") == 0.0)

    # first reply unparseable -> repair retry uses the second reply
    ep3, calls3 = scripted_endpoint(['no json at all', '{"pass": true}'])
    j3 = po.PromptOptimizer(ep3)._build_guidelines_scorer(cfg, ["be concise"])
    r3 = j3(inputs={"s": "x"}, outputs="x")
    check("robust judge: repair retry recovers", r3 == 1.0 and len(calls3) == 2, f"score={r3} calls={len(calls3)}")

    # json mode is requested on the judge call
    check("robust judge calls in json_object mode",
          calls3[0]["kwargs"].get("response_format") == {"type": "json_object"}
          and calls3[0]["kwargs"].get("temperature") == 0)


def test_lines_constraint_tightened():
    C = po.check_output_constraints
    # decorations that mean "prose", not a clean line list, are rejected
    check("lines: numbered list rejected", not C("1. one\n2. two", {"format": "lines"}))
    check("lines: paren-numbered rejected", not C("1) one\n2) two", {"format": "lines"}))
    check("lines: bullet list rejected", not C("- one\n- two", {"format": "lines"}))
    check("lines: markdown heading rejected", not C("# Keywords\none\ntwo", {"format": "lines"}))
    check("lines: quoted line rejected", not C('"one"\n"two"', {"format": "lines"}))
    check("lines: clean list still ok", C("urgent care near me\nurgent care open now", {"format": "lines"}))
    # exact line_count is enforced
    lc = {"format": "lines", "line_count": 3}
    check("lines: exact count ok", C("a\nb\nc", lc))
    check("lines: too few fails", not C("a\nb", lc))
    check("lines: too many fails", not C("a\nb\nc\nd", lc))
    # instruction text reflects the count
    instr = po.constraint_instruction(lc)
    check("lines instruction states the count", "exactly 3" in instr, instr)


def test_constraint_gate():
    import types as _t

    def scripted_endpoint(replies):
        ep = FakeEndpoint()
        seq, calls = list(replies), []

        def create(model, messages, **kw):
            calls.append({"messages": messages, "kwargs": kw})
            content = seq[min(len(calls) - 1, len(seq) - 1)]
            return _t.SimpleNamespace(choices=[_t.SimpleNamespace(
                message=_t.SimpleNamespace(content=content))])
        comp = _t.SimpleNamespace(create=create)
        ep._client = _t.SimpleNamespace(chat=_t.SimpleNamespace(completions=comp),
                                        completions=comp, _calls=calls)
        return ep, calls

    cfg = po.OptimizationConfig(
        "t", "p {{s}}", po.Mode.FEW_SHOT, examples=[{"s": "x"}], input_keys=["s"],
        quality_criteria=["be concise"],
        output_constraints={"format": "lines", "line_count": 3}, constraint_gate=True)

    # gate FAIL: wrong line count -> score 0 and the judge is NOT called
    ep, calls = scripted_endpoint(['{"pass": true}'])
    j = po.PromptOptimizer(ep)._build_guidelines_scorer(cfg, ["be concise"])
    r_fail = j(inputs={"s": "x"}, outputs="only one line")
    check("gate fail -> 0.0 without a judge call", r_fail == 0.0 and len(calls) == 0,
          f"score={r_fail} calls={len(calls)}")

    # gate PASS: compliant output -> judge runs and its verdict stands
    ep2, calls2 = scripted_endpoint(['{"per_guideline": [true]}'])
    j2 = po.PromptOptimizer(ep2)._build_guidelines_scorer(cfg, ["be concise"])
    r_pass = j2(inputs={"s": "x"}, outputs="a\nb\nc")
    check("gate pass -> judge scores normally", r_pass == 1.0 and len(calls2) == 1,
          f"score={r_pass} calls={len(calls2)}")

    # with the gate on, no SEPARATE averaged output_constraints scorer is added
    scs = po.PromptOptimizer(FakeEndpoint())._build_scorers(cfg)
    names = [getattr(s, "name", "") for s in scs]
    check("gate -> constraints not a separate averaged scorer",
          "output_constraints" not in names and "guidelines" in names, str(names))

    # without the gate, the separate constraint scorer IS added (unchanged behavior)
    cfg_no_gate = po.OptimizationConfig(
        "t", "p {{s}}", po.Mode.FEW_SHOT, examples=[{"s": "x"}], input_keys=["s"],
        quality_criteria=["be concise"], output_constraints={"format": "lines"})
    names2 = [getattr(s, "name", "") for s in po.PromptOptimizer(FakeEndpoint())._build_scorers(cfg_no_gate)]
    check("no gate -> separate constraint scorer present", "output_constraints" in names2, str(names2))

    # validation guards
    def rejects(**kw):
        try:
            po.OptimizationConfig("t", "p {{s}}", **kw).validate()
            return False
        except po.ConfigError:
            return True

    check("gate without constraints rejected",
          rejects(mode=po.Mode.FEW_SHOT, examples=[{"s": "x"}], input_keys=["s"],
                  quality_criteria=["c"], constraint_gate=True))
    check("gate in TRAINING_DATA rejected",
          rejects(mode=po.Mode.TRAINING_DATA, training_data=[{"s": "x", "expected_output": "y"}],
                  input_keys=["s"], output_constraints={"format": "lines"}, constraint_gate=True))
    check("gate with builtin judge rejected",
          rejects(mode=po.Mode.FEW_SHOT, examples=[{"s": "x"}], input_keys=["s"],
                  quality_criteria=["c"], output_constraints={"format": "lines"},
                  constraint_gate=True, use_builtin_judge=True))


def _usage_obj(pt, ct, rt=0):
    import types as _t
    return _t.SimpleNamespace(
        prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
        completion_tokens_details=_t.SimpleNamespace(reasoning_tokens=rt))


class UsageClient:
    """Fake OpenAI client that attaches a usage block to every completion."""

    def __init__(self, pt=10, ct=5, rt=3, content=None):
        self.calls = []
        self.chat = types.SimpleNamespace(completions=self)
        self._u = (pt, ct, rt)
        self._content = content

    def create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        joined = " | ".join(m["content"] for m in messages)
        msg = types.SimpleNamespace(content=self._content or f"OUT<{joined}>")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg)], usage=_usage_obj(*self._u))


class UsageEndpoint(FakeEndpoint):
    def __init__(self, pt=10, ct=5, rt=3, content=None):
        self._client = UsageClient(pt, ct, rt, content)


def test_usage_accounting():
    # -- _usage_fields parsing (dict and object; reasoning under details) --------
    f = po._usage_fields({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                          "completion_tokens_details": {"reasoning_tokens": 4}})
    check("usage_fields dict form", f == {"input": 10, "output": 5, "reasoning": 4, "total": 15}, str(f))
    f2 = po._usage_fields(_usage_obj(7, 3, 2))
    check("usage_fields object+details", f2 == {"input": 7, "output": 3, "reasoning": 2, "total": 10}, str(f2))
    check("usage_fields None -> None", po._usage_fields(None) is None)
    check("usage_fields all-zero -> None",
          po._usage_fields({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}) is None)
    check("usage_fields infers total when absent",
          po._usage_fields({"prompt_tokens": 6, "completion_tokens": 4})["total"] == 10)

    # -- UsageMeter aggregation (per role + grand total) -------------------------
    m = po.UsageMeter()
    m.record("task", _usage_obj(10, 5, 3))
    m.record("task", _usage_obj(10, 5, 3))
    m.record("judge", _usage_obj(8, 2, 0))
    d = m.as_dict()
    check("meter task rolled up", d["by_role"]["task"] == {"calls": 2, "input": 20, "output": 10, "reasoning": 6, "total": 30}, str(d["by_role"]["task"]))
    check("meter judge reasoning zero", d["by_role"]["judge"]["reasoning"] == 0)
    check("meter grand total",
          d["total"] == {"calls": 3, "input": 28, "output": 12, "reasoning": 6, "total": 40}, str(d["total"]))
    check("meter ignores unparseable usage", (po.UsageMeter().record("task", None) or True))

    # -- task usage captured through predict_fn ----------------------------------
    opt = po.PromptOptimizer(UsageEndpoint(10, 5, 3))
    opt._usage = po.UsageMeter()
    cfg = po.OptimizationConfig(
        "unit_usage_task", [po.Prompt("SYS {{x}}", "system"), po.Prompt("Go", "user")],
        po.Mode.FEW_SHOT, examples=[{"x": "a"}], input_keys=["x"], quality_criteria="g")
    reg = opt._register_prompts(cfg)
    opt._build_predict_fn(reg, "glm-5.1", cfg)(x="a")
    td = opt._usage.as_dict()
    check("task usage captured", td["by_role"].get("task") == {"calls": 1, "input": 10, "output": 5, "reasoning": 3, "total": 15}, str(td))

    # -- judge usage captured through the guidelines scorer ----------------------
    opt2 = po.PromptOptimizer(UsageEndpoint(8, 2, 1, content='{"per_guideline": [true]}'))
    opt2._usage = po.UsageMeter()
    cfg2 = po.OptimizationConfig("u2", "p {{x}}", po.Mode.FEW_SHOT, examples=[{"x": "a"}],
                                 input_keys=["x"], quality_criteria=["be good"])
    opt2._build_guidelines_scorer(cfg2, ["be good"])(inputs={"x": "a"}, outputs="hello")
    jd = opt2._usage.as_dict()
    check("judge usage captured (single clean call)",
          jd["by_role"].get("judge") == {"calls": 1, "input": 8, "output": 2, "reasoning": 1, "total": 10}, str(jd))

    # -- scorer built with no meter set doesn't crash (eval_prompt path) ----------
    opt3 = po.PromptOptimizer(UsageEndpoint(content='{"per_guideline": [true]}'))
    opt3._build_guidelines_scorer(cfg2, ["be good"])(inputs={"x": "a"}, outputs="hi")
    check("no-meter scorer path is safe", opt3._usage is None)


def test_reflection_length_control():
    # the brevity template bakes in the ceiling and keeps GEPA's required placeholders
    t = po.concise_reflection_template(600)
    check("template carries the ceiling number", "600" in t)
    check("template keeps <curr_param>/<side_info>", "<curr_param>" in t and "<side_info>" in t)
    # gepa itself must accept the override (validates placeholders)
    try:
        from gepa.strategies.instruction_proposal import InstructionProposalSignature as S
        S.validate_prompt_template(t)
        check("gepa accepts our override template", True)
    except Exception as e:  # noqa: BLE001
        check("gepa accepts our override template", False, f"{type(e).__name__}: {e}")

    # wired into gepa_kwargs only when a ceiling is set
    base = dict(mode=po.Mode.FEW_SHOT, examples=[{"s": "a"}], input_keys=["s"], quality_criteria="g")
    opt = make_opt()
    off = opt._build_optimizer(po.OptimizationConfig("t", "p {{s}}", **base))
    check("no reflection template by default",
          "reflection_prompt_template" not in getattr(off, "gepa_kwargs", {}))
    on = opt._build_optimizer(po.OptimizationConfig("t", "p {{s}}", reflection_max_chars=500, **base))
    tmpl = getattr(on, "gepa_kwargs", {}).get("reflection_prompt_template", "")
    check("brevity template wired into gepa_kwargs when set",
          "500" in tmpl and "<curr_param>" in tmpl, tmpl[:40])

    # validation guards
    def rejects(**kw):
        try:
            po.OptimizationConfig("t", "p {{s}}", **kw).validate(); return False
        except po.ConfigError:
            return True
    check("non-positive reflection_max_chars rejected",
          rejects(reflection_max_chars=0, **base))
    ok = True
    try:
        po.OptimizationConfig("t", "p {{s}}", reflection_max_chars=800, **base).validate()
    except po.ConfigError:
        ok = False
    check("positive reflection_max_chars accepted", ok)

    # observability: prompt_chars + echoed cap + to_dict + summary verdict
    r = po.OptimizationResult(
        name="t", mode="few_shot", optimizer="GepaPromptOptimizer", provider="openai",
        task_model="glm-5.1", prompts=[po.OptimizedPrompt("t.sys", "system", "abcde", [])],
        reflection_max_chars=3)
    check("prompt_chars measures final length", r.prompt_chars == 5, str(r.prompt_chars))
    d = r.to_dict()
    check("to_dict exposes prompt_chars + cap",
          d["prompt_chars"] == 5 and d["reflection_max_chars"] == 3, str({k: d[k] for k in ("prompt_chars", "reflection_max_chars")}))
    check("summary flags OVER when above the cap", "OVER" in r.summary())


def test_holdout_evaluation():
    # fill_template: substitutes known vars, leaves unknown intact
    check("fill substitutes vars", po.fill_template("SYS {{x}} and {{ y }}", {"x": "a", "y": "b"}) == "SYS a and b")
    check("fill leaves unknown placeholders", po.fill_template("keep {{z}}", {}) == "keep {{z}}")

    # evaluate_prompts: runs task per input, judges each, means the scores
    opt = po.PromptOptimizer(UsageEndpoint(content='{"per_guideline": [true]}'))
    cfg = po.OptimizationConfig("hold", "p {{x}}", po.Mode.FEW_SHOT, examples=[{"x": "a"}],
                                input_keys=["x"], quality_criteria=["be good"])
    prompts = [po.OptimizedPrompt("s", "system", "SYS {{x}}", ["x"]),
               po.OptimizedPrompt("u", "user", "Go", [])]
    r = opt.evaluate_prompts(cfg, prompts, [{"x": "a"}, {"x": "b"}])
    check("held-out scored every input", r["n"] == 2 and len(r["outputs"]) == 2, str(r["n"]))
    check("held-out mean from judge", r["mean"] == 1.0, str(r["mean"]))
    # the task prompt was filled with the row's value
    first_task_msgs = opt.endpoint.client.calls[0]["messages"]
    check("held-out fills the task prompt", first_task_msgs[0]["content"] == "SYS a", str(first_task_msgs[:1]))


def test_tracking_off_and_tracing_restored():
    """GP keeps its own results: MLflow runs and traces are off unless asked for (or a custom scorer
    may read the trace), and a run leaves the process-wide tracing state as it found it."""
    import mlflow
    from mlflow.tracing.provider import is_tracing_enabled
    import prompt_optimizer as po
    check("enable_tracking defaults to False",
          po.OptimizationConfig.__dataclass_fields__["enable_tracking"].default is False)
    for start in (True, False):                     # the caller's state, before a run
        (mlflow.tracing.enable if start else mlflow.tracing.disable)()
        for on in (True, False):                    # what the run needs
            before = po._set_tracing(on)
            check(f"tracing set to {on} (was {start})", is_tracing_enabled() is on and before is start)
            po._set_tracing(before)
            check(f"tracing restored to {start}", is_tracing_enabled() is start)
    mlflow.tracing.enable()
    # optimize() restores it even when the run raises
    import types
    fake = types.SimpleNamespace(validate=lambda: None)
    opt = po.PromptOptimizer.__new__(po.PromptOptimizer)
    opt.endpoint = types.SimpleNamespace(task_model="t", reflection_model="r", judge_model="j")
    for name in ("_register_prompts", "_build_predict_fn", "_build_train_data", "_build_scorers", "_build_optimizer"):
        setattr(opt, name, lambda *a, **k: [])
    cfg = types.SimpleNamespace(validate=lambda: None, task_model=None, reflection_model=None, judge_model=None,
                                mode=po.Mode.FEW_SHOT, enable_tracking=False, custom_scorers=())
    raised = False
    try:
        import unittest.mock as um
        with um.patch("mlflow.genai.optimize_prompts", side_effect=RuntimeError("boom")):
            opt.optimize(cfg)
    except RuntimeError:
        raised = True
    check("a failed run restores tracing", raised and is_tracing_enabled() is True)


def main() -> int:
    for fn in (test_coerce_prompts, test_validation, test_train_data,
               test_scorers_and_optimizer, test_predict_fn_and_injection,
               test_result_mapping, test_output_constraints,
               test_constraints_wire_into_scorer_and_prompt, test_tighten_pass,
               test_extract_json, test_verdict_to_score, test_robust_guidelines_judge,
               test_lines_constraint_tightened, test_constraint_gate,
               test_usage_accounting, test_reflection_length_control,
               test_holdout_evaluation, test_tracking_off_and_tracing_restored):
        print(f"\n== {fn.__name__} ==")
        fn()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())

