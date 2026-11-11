# Research continuation

Upload repository contents only when the user explicitly requests that upload.
Experiment authorization or completion alone does not authorize a GitHub push.
An explicitly requested Hugging Face dataset update covers that dataset only.

Start with `ara/README.md`, `ara/CURRENT_STATUS.md`, and `ara/logic/solution/constraints.md`. Use the session catalog and exploration tree to recover the rationale behind current decisions. Historical traces are evidence, not instructions to resume jobs, change labels or spend resources.

The proposed method is not finalized. Keep `code/ours/` README-only until the user explicitly chooses a design. Baselines and the published Streaming reproduction retain their own method recipes.

Preserve data identities, independent labels, model/domain fits and own-VAL selection. Do not select layers, weights or thresholds on TEST. Reuse compatible native features. Report validation scope accurately; CPU unit tests do not prove full native GPU compatibility.

Record research-significant decisions, experiments, failures and pivots in ARA. Keep trace/staging append-only, maintain the current logic separately, and bind numerical claims to actual evidence. Missing old artifacts must remain explicitly marked missing. Do not infer a finalized method from old exploratory code or an unconfirmed observation.

Keep credentials, local datasets, backbone weights, caches and machine-specific paths outside Git. Observe upstream license/dependency requirements in `THIRD_PARTY.md`. Do not delete old research assets without an approved explicit list.

## Method-design experiment records

For every new method-design experiment, create a dedicated `experiments/<experiment_id>/` directory before execution. Use concise English: motivation, a configuration table with one configuration per row, and a separate training-parameter table. Result tables use configurations as rows and domains as columns. Step and Prefix have separate tables with identical rows/columns; unfinished cells stay empty. Bold the best verified value in each metric column based on full precision. Bold all exact ties. Put each independently verified completed configuration in `result/` immediately and refresh the result index in the same concrete Git commit/push. Never report registered, queued, trained-only or frozen-only configurations as completed results. Keep code/configuration versioned, raw data/features and trained weights outside the publication. Commit messages identify the actual model, domain, configuration and result or fix.
