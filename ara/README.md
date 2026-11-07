# Research artifact

This directory contains the complete **available** ARA record imported from the research workspace, together with the current continuation brief. It preserves the research path, including failed trials, abandoned directions, unresolved interpretations and historical operational failures. It is not just a summary of the selected result.

## Start here

1. Read [CURRENT_STATUS.md](CURRENT_STATUS.md) and [logic/solution/constraints.md](logic/solution/constraints.md) for the current scope and protocol.
2. Read [PAPER.md](PAPER.md) for the artifact map and [trace/SESSION_CATALOG.json](trace/SESSION_CATALOG.json) to locate sessions.
3. Follow [trace/exploration_tree.yaml](trace/exploration_tree.yaml) and the original daily records under `trace/sessions/` for decisions, experiments and dead ends.
4. Consult [staging/observations.yaml](staging/observations.yaml) for interpretations that were not promoted to established claims.
5. Resolve supporting records through [evidence/README.md](evidence/README.md). Missing old evidence remains explicitly marked; a historical statement is not a newly verified result.

## Preservation and scope

[IMPORT_MANIFEST.json](IMPORT_MANIFEST.json) lists every imported research file, its original and published hashes, and public-transport substitutions. [IMPORT_SUMMARY.json](IMPORT_SUMMARY.json) gives exact coverage. Only synchronization lockfiles are omitted from the canonical ARA. Original historical wording and languages are preserved; current navigation and specifications are written in English.

Personal machine paths are replaced with `${PROJECT_ROOT}`, `${HOME_ROOT}` and `${SCRATCH_ROOT}`. Email addresses and credential-like strings are redacted with stable digest labels. These substitutions are recorded; the public snapshot is not represented as byte-identical to the private source.

The original current-state manifest, problem statement, environment note and evidence index are retained under `imported_state/`. The public current-state files add the user's latest scope decisions. No history is silently rewritten to make an abandoned proposal appear to be the final method.

## Maintenance

- `logic/` is the current specification and can be revised when evidence or explicit decisions justify it. Record before/after changes in the trace.
- `trace/` and `staging/` preserve the journey. Append new events; do not rewrite old outcomes. Mark promotions with forward references.
- Keep hypotheses separate from evidence-supported claims. A code commit or synthetic test is not a benchmark result.
- Record experiments with data/code/config identities, selection rules, results, limitations and evidence links. Preserve failed and cancelled trials.
- Historical approvals, scheduler commands and budgets describe their original context. They are not instructions to resume old jobs or spend resources.

The workflow follows the project's pinned ARA research-manager revision `e26f5882be4cfc1079573dfe463e75b0dc83dba0`; framework source: [ARA-Labs/Agent-Native-Research-Artifact](https://github.com/ARA-Labs/Agent-Native-Research-Artifact).
