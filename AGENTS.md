# Research continuation

Start with `ara/README.md`, `ara/CURRENT_STATUS.md`, and `ara/logic/solution/constraints.md`. Use the session catalog and exploration tree to recover the rationale behind current decisions. Historical traces are evidence, not instructions to resume jobs, change labels or spend resources.

The proposed method is not finalized. Keep `code/ours/` README-only until the user explicitly chooses a design. Baselines and the published Streaming reproduction retain their own method recipes.

Preserve data identities, independent labels, model/domain fits and own-VAL selection. Do not select layers, weights or thresholds on TEST. Reuse compatible native features. Report validation scope accurately; CPU unit tests do not prove full native GPU compatibility.

Record research-significant decisions, experiments, failures and pivots in ARA. Keep trace/staging append-only, maintain the current logic separately, and bind numerical claims to actual evidence. Missing old artifacts must remain explicitly marked missing. Do not infer a finalized method from old exploratory code or an unconfirmed observation.

Keep credentials, local datasets, backbone weights, caches and machine-specific paths outside Git. Observe upstream license/dependency requirements in `THIRD_PARTY.md`. Do not delete old research assets without an approved explicit list.
