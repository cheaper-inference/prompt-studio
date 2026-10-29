# Optimization research worker

This repository owns the authenticated Python research worker and its pinned
artifacts. It runs separately from the GEPA web and durable job processes: its
dependencies, Dockerfile and startup command are contained in this directory.
The repository's root Dockerfile and GEPA service startup do not start it.

Client applications should keep the worker URL and authentication token in
server-side configuration. Model weights stay with the worker.

Use Python 3.11 on Linux x86_64. The direct versions are in `requirements.txt`;
the full transitive, hashed resolution is `requirements.lock`. Model repositories
are pinned to immutable revisions in `artifacts.json`.
`artifacts.json` in this directory is the canonical artifact manifest. Client
adapters and receipts must agree on the exact model revision. SoT's model
revision and separate template `sourceRevision` must both retain their provenance.

The SoT English prompts, exemplars and license are vendored unchanged from the
[authors' repository](https://github.com/SimonAytes/SoT/tree/94c6b8d17f3fe7f7fe536ddc868e257bdfe9d654).

Run the following setup from `workers/optimization/`:

1. Create an isolated Python 3.11 environment and run
   `pip install --require-hashes -r requirements.lock`.
2. Choose `OPTIMIZATION_OPERATIONS` and run `python download_artifacts.py` in
   that environment. This is the explicit network/model-download step.
   Weights are ignored by Git. For an isolated
   container, mount the populated `models/` directory read-only at `/worker/models`.
3. Set a strong `OPTIMIZATION_WORKER_TOKEN`, keep the same
   `OPTIMIZATION_OPERATIONS` selection for the service, and set
   `OPTIMIZATION_COMPUTE_USD_PER_SECOND` to the reviewed worker compute price.
   An omitted price produces **unknown**, never zero, cost. Unknown costs block
   evaluation completion and promotion. `OPTIMIZATION_DEVICE` defaults to `cpu`.
4. Start `uvicorn app:app --host 127.0.0.1 --port 8091 --workers 1 --no-access-log`.
   Put TLS and a private network boundary in front of a remote worker. Set the
   corresponding server-side URL/token on your client application. Requests never
   download models; offline Hugging Face mode is enabled by default.

To build its standalone image from the repository root, use this directory as
the Docker context:

```sh
docker build -t gepa-optimization-worker workers/optimization
```

The worker's `.dockerignore` excludes downloaded weights, tokenizer data, local
credentials and caches. Mount the prepared `models/` directory at
`/worker/models` read-only. Build and provision this service independently; adding these
sources to this repository does not deploy it or change the existing GEPA image.

When unset, `OPTIMIZATION_OPERATIONS` defaults to `llmlingua2,sot,embed,verify`.
Both setup and serving trim whitespace, ignore empty comma-separated entries,
and remove duplicates. An explicitly blank value disables all operations.
Unknown names fail with a list of supported operations before any downloads;
invalid service configuration rejects execution.

The endpoints are `/v1/llmlingua2`, `/v1/sot`, `/v1/embed` and
`/v1/verify`. The worker reads revisions from `artifacts.json`; client applications
should use matching artifact metadata. SoT's template source commit is recorded
separately as `sourceRevision`.
Every successful response has `evidence.artifactRevision` and
`usage` with worker token counts, latency and compute price. These are overhead,
not estimates of the downstream model's billed token savings. The SoT worker
uses the original DistilBERT classifier and original full English exemplars;
the adapter limits it to explicitly approved English tasks without existing
system instructions, tools or structured output. Compression operates only on
hash-designated context passages; the surrounding request is preserved.

## Offline contract tests

From this directory, install only the small test dependency set and run:

```sh
python -m pip install fastapi==0.116.1 pydantic==2.11.7 httpx==0.28.1
python -m unittest discover -s . -p 'test_*.py' -v
```

The repository's `Optimization worker CI` workflow runs the same suite on Python
3.11. Model execution, model downloads and external requests are mocked; the
tests do not require PyTorch, Transformers, model weights or credentials. These
checks do not execute paid inference or deploy the service. Validate model
behavior separately against representative held-out data before use.
