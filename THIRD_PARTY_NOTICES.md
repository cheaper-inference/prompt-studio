# Third-party notices

Prompt Studio's original code is licensed under the [MIT License](LICENSE).
Third-party code, documentation, prompts, examples, dependencies, and model
weights retain their respective copyrights and licenses. The root license does
not replace their terms.

## GEPA

Prompt Studio builds on [GEPA](https://github.com/gepa-ai/gepa), the reflective
prompt optimizer, through MLflow's optimization integration. Its Python
dependency is pinned to `gepa==0.1.4` in `requirements.txt`.

GEPA is MIT licensed, copyright © 2025 Lakshya A Agrawal. Its full notice is
preserved in [licenses/GEPA-LICENSE](licenses/GEPA-LICENSE). The corresponding
upstream license is available at
[GEPA v0.1.4 LICENSE](https://github.com/gepa-ai/gepa/blob/v0.1.4/LICENSE).
Retain upstream notices with any redistributed GEPA material.
The vendored `gepa-optimize-anything` agent skill under `.claude/skills/`
also identifies GEPA as its upstream project and MIT as its license.

## SoT prompts and examples

The English prompts, exemplars, and supporting assets in
`workers/optimization/artifacts/sot/` are vendored from
[SimonAytes/SoT](https://github.com/SimonAytes/SoT/tree/94c6b8d17f3fe7f7fe536ddc868e257bdfe9d654).
Their MIT notice, copyright (c) 2025 Simon Aytes, remains in
[the vendored LICENSE](workers/optimization/artifacts/sot/LICENSE).
The upstream source revision is recorded in the worker's `artifacts.json`.

## Dependencies and downloaded model artifacts

Python dependencies are installed separately from the requirement files and
retain their own license terms. Preserve their notices when redistributing
installed packages or container images.

Model weights are not bundled in this source release. The worker's
[artifact manifest](workers/optimization/artifacts.json) records model identifiers,
pinned revisions, and license metadata. Check the actual upstream terms before
downloading, using, modifying, or redistributing each artifact; manifest metadata
is not a grant of rights.
