# Third-party sources

Streaming (MIT) and ICR Probe (Apache-2.0) include their pinned source subsets and original license notices. Preserve those notices when reusing the code.

Other upstream implementations are **references and external dependencies**, not redistributed source bundles. In particular, the LLM-Check source notice prohibits giving that software to third parties. No redistribution permission is inferred when an upstream snapshot contains no license file. Each method's `upstream/PROVENANCE.json` records its original repository, revision and file hashes.

## Optional local dependencies

For HaMI, ARS and RACE, obtain the pinned implementation directly from its author after checking its terms. ReDe uses the same separately obtained ARS supervised-probe definition. The project never launches an upstream training script when loading its definitions.

```bash
# Only if the original author's terms permit your use:
python code/tools/prepare_upstream.py --method hami --accept-upstream-terms
python code/tools/prepare_upstream.py --method ars --accept-upstream-terms
python code/tools/prepare_upstream.py --method race --accept-upstream-terms
```

Files are placed under `.cache/upstream/<method>/<revision>/`, which is ignored by Git. Set `STREAMING_HD_UPSTREAM_ROOT` to use another private location. Downloads are verified against the pinned source hashes. Do not commit these external files to this repository.

The native LLM-Check and RHD statistic implementations do not require the original upstream launchers or patched Transformers distribution. Their source references are retained for attribution; obtaining those original bundles is optional and subject to their terms.

Model weights and source datasets are not distributed here. Detector checkpoints under `experiments/` are project outputs. Data source terms and redistribution notices are provided by the linked dataset.
