"""One place to configure an OpenAI-compatible endpoint for MLflow prompt optimization.

MLflow reaches three model roles through three different code paths, each with its
own configuration channel:

    task model   predict_fn                -> openai SDK      -> explicit base_url
    reflection   Gepa/MetaPromptOptimizer  -> litellm         -> kwargs, then env
    judge        Correctness & friends     -> mlflow gateway  -> env ONLY

``configure()`` sets the environment for the paths that have no parameter, and
returns an :class:`Endpoint` that hands you the right object or dict for the paths
that do.

    from llm_endpoint import configure

    ep = configure()                       # reads env, sets OPENAI_* as a side effect

    ep.client                              # openai.OpenAI, explicitly pointed
    ep.task_model                          # "deepseek-v4-flash-0731-fast"
    ep.reflection_uri                      # "openai:/deepseek-v4-flash-0731-fast"
    ep.lm_kwargs()                         # -> MetaPromptOptimizer(lm_kwargs=...)
    ep.gepa_kwargs()                       # -> GepaPromptOptimizer(gepa_kwargs=...)

Side effect, deliberate and documented: ``configure()`` writes OPENAI_API_KEY,
OPENAI_BASE_URL and OPENAI_API_BASE into ``os.environ``. The judge path reads no
other channel, so this is not optional.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


def time_calls(client, model: str, messages: list[dict], n: int = 5,
               timeout: int = 60) -> tuple[list[float], str | None]:
    """Time ``n`` sequential chat calls; stop at the first error. Returns
    ``(latencies, error_or_None)``. Shared by the diagnose script and the webapp's
    latency-standings probe so both measure identically — kept here (a production
    module) rather than in a test module so nothing production imports from tests."""
    lats: list[float] = []
    err: str | None = None
    for _ in range(n):
        t0 = time.time()
        try:
            client.chat.completions.create(model=model, messages=messages, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            lats.append(time.time() - t0)
            break
        lats.append(time.time() - t0)
    return lats, err

DEFAULT_KEY_VAR = "CHEAPERINFERENCE_API_KEY"
DEFAULT_URL_VAR = "CHEAPERINFERENCE_BASE_URL"
DEFAULT_MODEL = "deepseek-v4-flash-0731-fast"


class EndpointConfigError(RuntimeError):
    """Raised when the endpoint cannot be configured from the environment."""


def _mask(secret: str) -> str:
    if not secret:
        return "<unset>"
    return f"{secret[:6]}…{secret[-4:]}" if len(secret) > 14 else "…"


@dataclass(frozen=True)
class Endpoint:
    """A configured OpenAI-compatible endpoint plus the per-role plumbing."""

    base_url: str
    api_key: str = field(repr=False)
    task_model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_MODEL
    reflection_model: str = DEFAULT_MODEL
    provider: str = "openai"
    timeout: float = 300.0

    # -- role A: task model -------------------------------------------------

    @property
    def client(self):
        """An openai.OpenAI pointed at this endpoint, explicitly (not via env)."""
        import openai

        # dataclass is frozen; cache on the instance via object.__setattr__
        cached = self.__dict__.get("_client")
        if cached is None:
            cached = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            object.__setattr__(self, "_client", cached)
        return cached

    def complete(self, messages: list[dict[str, str]], model: str | None = None, **kwargs) -> str:
        """Convenience one-shot chat call against the task model."""
        resp = self.client.chat.completions.create(
            model=model or self.task_model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content or ""

    # -- MLflow model URIs --------------------------------------------------

    def uri(self, model: str) -> str:
        """MLflow model URI. The colon-slash form is required by _parse_model_uri,
        and the 'openai' prefix is what routes litellm down the OpenAI-compatible
        path where api_base is honoured -- keep it for non-OpenAI models too."""
        return f"{self.provider}:/{model}"

    @property
    def task_uri(self) -> str:
        return self.uri(self.task_model)

    @property
    def judge_uri(self) -> str:
        return self.uri(self.judge_model)

    @property
    def reflection_uri(self) -> str:
        return self.uri(self.reflection_model)

    # -- role B: reflection model ------------------------------------------

    def lm_kwargs(self, **extra: Any) -> dict[str, Any]:
        """For ``MetaPromptOptimizer(lm_kwargs=...)``.

        Merged last into litellm.completion kwargs by MLflow, so api_base/api_key
        here override the hardcoded None MLflow passes for plain openai:/ URIs.
        Extra headers retain reflection attribution unless explicitly overridden.
        """
        headers = {"x-gepa-purpose": "reflection", **(extra.pop("extra_headers", None) or {})}
        return {"api_base": self.base_url, "api_key": self.api_key,
                "extra_headers": headers, **extra}

    def gepa_kwargs(self, **extra: Any) -> dict[str, Any]:
        """For ``GepaPromptOptimizer(gepa_kwargs=...)``.

        ``reflection_lm_kwargs`` is not in MLflow's override set, so it reaches
        gepa.optimize -> gepa.lm.LM(**kwargs) -> litellm.completion. Any other
        gepa.optimize argument can be passed through as **extra.
        """
        reflection = self.lm_kwargs(**{
            "num_retries": 0, **extra.pop("reflection_lm_kwargs", {})})
        return {"reflection_lm_kwargs": reflection, **extra}

    # -- diagnostics --------------------------------------------------------

    def summary(self) -> str:
        return (
            f"endpoint   {self.base_url}\n"
            f"api key    {_mask(self.api_key)}\n"
            f"task       {self.task_uri}\n"
            f"judge      {self.judge_uri}\n"
            f"reflection {self.reflection_uri}"
        )

    def preflight(self, verbose: bool = True) -> list[str]:
        """Hit each role once. Returns a list of failure strings (empty == all good).

        Worth running before any optimization: a misrouted judge otherwise fails
        partway into a paid run rather than at startup.
        """
        probe = [{"role": "user", "content": "Reply with the single word OK."}]
        failures: list[str] = []

        def report(role: str, ok: bool, detail: str) -> None:
            if verbose:
                print(f"  [{'PASS' if ok else 'FAIL'}] {role:22s} {detail}")

        # A. task model -- openai SDK
        try:
            report("task (openai sdk)", True, repr(self.complete(probe)[:40]))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"task: {type(exc).__name__}: {exc}")
            report("task (openai sdk)", False, str(exc)[:90])

        # B. reflection -- MLflow's own litellm wrapper (MetaPromptOptimizer path)
        try:
            from mlflow.genai.utils.llm_utils import _call_llm

            resp = _call_llm(
                self.reflection_uri, probe, inference_params=self.lm_kwargs()
            )
            report("reflection (mlflow)", True, repr(resp.choices[0].message.content[:40]))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"reflection/mlflow: {type(exc).__name__}: {exc}")
            report("reflection (mlflow)", False, str(exc)[:90])

        # C. reflection -- GEPA's litellm wrapper (GepaPromptOptimizer path)
        try:
            from gepa.lm import LM

            lm = LM(f"{self.provider}/{self.reflection_model}", **self.lm_kwargs())
            report("reflection (gepa)", True, repr(lm(probe)[:40]))
        except ImportError:
            report("reflection (gepa)", True, "gepa not installed, skipped")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"reflection/gepa: {type(exc).__name__}: {exc}")
            report("reflection (gepa)", False, str(exc)[:90])

        # D. judge -- MLflow gateway resolution, env vars only
        try:
            from mlflow.metrics.genai.model_utils import score_model_on_payload

            out = score_model_on_payload(self.judge_uri, "Reply with the single word OK.")
            report("judge (mlflow env)", True, repr(str(out)[:40]))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"judge: {type(exc).__name__}: {exc}")
            report("judge (mlflow env)", False, str(exc)[:90])

        if verbose:
            print(
                "\nall routes reach the endpoint"
                if not failures
                else f"\n{len(failures)} route(s) failed"
            )
        return failures


def configure(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    task_model: str = DEFAULT_MODEL,
    judge_model: str | None = None,
    reflection_model: str | None = None,
    key_var: str = DEFAULT_KEY_VAR,
    url_var: str = DEFAULT_URL_VAR,
    env_file: str | None = ".env",
    set_openai_env: bool = True,
) -> Endpoint:
    """Resolve the endpoint from arguments or environment and wire up MLflow.

    Args:
        base_url / api_key: explicit values; fall back to ``url_var`` / ``key_var``.
        task_model: model for predict_fn. judge_model and reflection_model default
            to it.
        env_file: dotenv file to load first, if present. Values already exported in
            the shell win. Pass None to skip.
        set_openai_env: write OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_API_BASE.
            Leave True unless you are deliberately targeting real OpenAI for the
            judge -- it is the only channel that path reads.

    Raises:
        EndpointConfigError: if the key or base URL cannot be resolved.
    """
    if env_file:
        _load_dotenv(env_file)

    base_url = base_url or os.environ.get(url_var)
    api_key = api_key or os.environ.get(key_var)

    if not api_key:
        raise EndpointConfigError(
            f"No API key. Set ${key_var} or pass api_key= explicitly."
        )
    if not base_url:
        raise EndpointConfigError(
            f"No base URL. Set ${url_var} or pass base_url= explicitly."
        )
    base_url = base_url.rstrip("/")

    if set_openai_env:
        _export_openai_env(base_url, api_key)

    return Endpoint(
        base_url=base_url,
        api_key=api_key,
        task_model=task_model,
        judge_model=judge_model or task_model,
        reflection_model=reflection_model or task_model,
    )


def _export_openai_env(base_url: str, api_key: str) -> None:
    """Write the three vars MLflow's judge path and litellm read.

    Both spellings, because the readers disagree on precedence:
        openai SDK -> OPENAI_BASE_URL only
        litellm    -> OPENAI_BASE_URL, then OPENAI_API_BASE
        mlflow     -> OPENAI_API_BASE, then OPENAI_BASE_URL
    A stale OPENAI_API_BASE left over from another project would otherwise
    outrank OPENAI_BASE_URL for the judge only, producing a run where two roles
    hit your endpoint and one silently does not.
    """
    for var in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        existing = os.environ.get(var)
        if existing and existing.rstrip("/") != base_url:
            _logger.warning("Overriding %s: %s -> %s", var, existing, base_url)
        os.environ[var] = base_url

    existing_key = os.environ.get("OPENAI_API_KEY")
    if existing_key and existing_key != api_key:
        _logger.warning("Overriding OPENAI_API_KEY (%s)", _mask(existing_key))
    os.environ["OPENAI_API_KEY"] = api_key


def _load_dotenv(path: str) -> None:
    """Minimal .env reader; shell-exported values take precedence."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


if __name__ == "__main__":
    ep = configure()
    print(ep.summary())
    print()
    raise SystemExit(1 if ep.preflight() else 0)
