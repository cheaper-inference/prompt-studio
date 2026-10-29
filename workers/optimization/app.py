"""Private deterministic worker boundary for pinned research implementations.

The original SoT prompts/exemplars are vendored under their MIT license. The
original DistilBERT router selects the paradigm; no generic shortening prompt is
substituted. Only the four operations in the public artifact manifest are supported.
"""
from __future__ import annotations

import hmac
import hashlib
import json
import math
import multiprocessing
import os
import threading
import time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from operations import configured_operations

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).parent
os.environ["TIKTOKEN_CACHE_DIR"] = str(ROOT / "models" / "tiktoken")
ARTIFACTS = json.loads((ROOT / "artifacts.json").read_text())
LOCK = threading.Lock()
MODEL_PROCESS = None
MODEL_PIPE = None

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def authenticate(authorization: Annotated[str | None, Header()] = None):
    token = os.environ.get("OPTIMIZATION_WORKER_TOKEN", "")
    if not token or not hmac.compare_digest(authorization or "", "Bearer " + token):
        raise HTTPException(401, "Unauthorized")


class Input(BaseModel):
    texts: list[str] = Field(default_factory=list, max_length=32)
    rate: float = Field(default=0.7, ge=0.1, le=1)
    question: str = Field(default="", max_length=16000)
    text: str = Field(default="", max_length=16000)
    left: str = Field(default="", max_length=16000)
    right: str = Field(default="", max_length=16000)
    maxCostUsd: str | None = Field(default=None, pattern=r"^\d+(\.\d+)?$")
    idempotencyKey: str | None = Field(default=None, max_length=300)


def model_path(operation):
    path = ROOT / "models" / operation
    if not path.is_dir():
        raise HTTPException(503, "Pinned artifact is not installed")
    return str(path)


@lru_cache(maxsize=4)
def load(operation):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = model_path(operation)
    if operation == "llmlingua2":
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        tokenizer_file = ROOT / "models" / "tiktoken" / hashlib.sha1(url.encode()).hexdigest()
        if not tokenizer_file.is_file() or hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7":
            raise HTTPException(503, "Pinned token-counting artifact is not installed")
        from llmlingua import PromptCompressor
        return PromptCompressor(model_name=path, use_llmlingua2=True, device_map=os.environ.get("OPTIMIZATION_DEVICE", "cpu"))
    if operation == "embed":
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(path, device=os.environ.get("OPTIMIZATION_DEVICE", "cpu"), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval()
    return tokenizer, model


def bounded_tokens(tokenizer, *texts):
    encoded = tokenizer(*texts, return_tensors="pt", truncation=False)
    # Reject truncation: a classifier must see the entire question/candidate.
    if encoded["input_ids"].shape[-1] > min(tokenizer.model_max_length, 512):
        raise HTTPException(422, "Input exceeds the pinned classifier context")
    return encoded


def execute(operation, value):
    import torch
    artifact = ARTIFACTS[operation]
    evidence = {"artifact": artifact["model"], "artifactRevision": artifact["revision"]}
    tokens = 0
    if operation == "llmlingua2":
        model = load(operation)
        output = []
        for text in value.texts:
            result = model.compress_prompt(text, rate=value.rate, force_tokens=["\n", ".", "?", "!"], drop_consecutive=True)
            output.append(result["compressed_prompt"])
            tokens += int(result.get("origin_tokens", 0))
        return {"texts": output}, evidence, tokens
    if operation == "sot":
        tokenizer, model = load(operation)
        encoded = bounded_tokens(tokenizer, value.question)
        with torch.inference_mode():
            label = model(**encoded).logits.argmax(-1).item()
        directory = ROOT / "artifacts" / "sot"
        labels = json.loads((directory / "label_mapping.json").read_text())
        paradigm = next(k for k, v in labels.items() if v == label)
        filenames = {"chunked_symbolism": "ChunkedSymbolism", "conceptual_chaining": "ConceptualChaining", "expert_lexicons": "ExpertLexicons"}
        system = (directory / f"EN_{filenames[paradigm]}_SystemPrompt.md").read_text()
        exemplars = json.loads((directory / "exemplars.json").read_text())["EN"][paradigm]
        messages = [{"role": "system", "content": system}]
        for ex in exemplars:
            messages.extend([{"role": "user", "content": ex["question"]}, {"role": "assistant", "content": ex["answer"]}])
        messages.append({"role": "user", "content": value.question})
        evidence.update({"paradigm": paradigm, "sourceRevision": artifact["sourceRevision"], "language": "EN"})
        return {"messages": messages}, evidence, encoded["input_ids"].numel()
    if operation == "embed":
        model = load(operation)
        tokens = len(model.tokenizer.encode(value.text))
        if tokens > model.max_seq_length:
            raise HTTPException(422, "Input exceeds the embedding context")
        return {"vector": model.encode(value.text, normalize_embeddings=True).tolist()}, evidence, tokens
    tokenizer, model = load("verify")
    scores = []
    # The original model card fixes label order: contradiction, entailment, neutral.
    for left, right in [(value.left, value.right), (value.right, value.left)]:
        encoded = bounded_tokens(tokenizer, left, right)
        tokens += encoded["input_ids"].numel()
        with torch.inference_mode():
            scores.append(torch.softmax(model(**encoded).logits, -1)[0, 1].item())
    evidence.update({"bidirectionalEntailment": scores, "threshold": 0.98, "probabilistic": True})
    return {"verified": all(s >= 0.98 for s in scores)}, evidence, tokens


def _model_loop(connection):
    """Keep weights warm in one killable process; no unbounded timed-out work."""
    while True:
        try:
            operation, payload = connection.recv()
        except EOFError:
            return
        try:
            connection.send({"result": execute(operation, Input(**payload))})
        except HTTPException as error:
            connection.send({"error": (error.status_code, error.detail)})
        except Exception:
            # Do not leak passage text, filesystem paths or service secrets.
            connection.send({"error": (500, "Pinned model execution failed")})


def _dispatch(operation, value, timeout):
    global MODEL_PROCESS, MODEL_PIPE
    deadline = time.monotonic() + timeout
    if MODEL_PROCESS is None or not MODEL_PROCESS.is_alive():
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        MODEL_PROCESS = ctx.Process(target=_model_loop, args=(child,), daemon=True)
        MODEL_PROCESS.start()
        child.close()
        MODEL_PIPE = parent
    try:
        MODEL_PIPE.send((operation, value.model_dump()))
        if not MODEL_PIPE.poll(max(0, deadline - time.monotonic())):
            MODEL_PROCESS.terminate()
            MODEL_PROCESS.join(timeout=5)
            MODEL_PIPE.close()
            MODEL_PROCESS = None
            MODEL_PIPE = None
            raise HTTPException(504, "Worker budget or execution deadline reached; reconcile compute receipt")
        result = MODEL_PIPE.recv()
    except (EOFError, BrokenPipeError):
        raise HTTPException(503, "Worker interrupted; reconcile compute receipt")
    if "error" in result:
        raise HTTPException(*result["error"])
    return result["result"]


@app.post("/v1/{operation}", dependencies=[Depends(authenticate)])
def run(operation: str, value: Input):
    if operation not in ARTIFACTS:
        raise HTTPException(404, "Operation not enabled")
    try:
        enabled = configured_operations(ARTIFACTS)
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    if operation not in enabled:
        raise HTTPException(404, "Operation not enabled")
    if sum(map(len, value.texts)) > 128000:
        raise HTTPException(413, "Context limit exceeded")
    try:
        configured_rate = os.environ.get("OPTIMIZATION_COMPUTE_USD_PER_SECOND")
        rate = Decimal(configured_rate) if configured_rate else None
        timeout = min(110.0, float(os.environ.get("OPTIMIZATION_MAX_EXECUTION_SECONDS", "110")))
        if not math.isfinite(timeout) or timeout <= 0 or (rate is not None and (not rate.is_finite() or rate < 0)):
            raise ValueError()
        if value.maxCostUsd is not None:
            if rate is None:
                raise HTTPException(503, "Budgeted execution requires configured compute pricing")
            budget = Decimal(value.maxCostUsd)
            if budget <= 0:
                raise HTTPException(422, "Positive compute budget required")
            if rate > 0:
                timeout = min(timeout, float(budget / rate))
    except (ValueError, ArithmeticError):
        raise HTTPException(503, "Invalid worker metering configuration")
    deadline = time.monotonic() + timeout
    # Models have mutable caches; serialized execution also bounds RAM/GPU use.
    with LOCK:
        if time.monotonic() >= deadline:
            raise HTTPException(504, "Worker queue deadline reached before model execution")
        started = time.monotonic()
        result, evidence, tokens = _dispatch(operation, value, deadline - started)
    elapsed = time.monotonic() - started
    cost = format(rate * Decimal(str(elapsed)), ".12f") if rate is not None else None
    return {**result, "evidence": evidence, "usage": {"inputTokens": tokens, "outputTokens": 0, "latencyMs": round(elapsed * 1000), "costUsd": cost, "source": "worker_compute_meter" if rate is not None else "worker_compute_unpriced"}}
