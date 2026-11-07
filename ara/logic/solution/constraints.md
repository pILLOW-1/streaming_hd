# Current constraints

- Use the agreed local data schema, original retained Step indices and independent Step/Prefix/Answer labels.
- Preserve the selected experiment's exact per-cell split membership when using its weights/results.
- Fit independently by generator/domain. Choose weights and layers with own VAL AUROC, then AP; choose decision thresholds with own VAL F1max. Never tune on TEST.
- Native feature identity includes model/tokenizer revision, full input token IDs, reasoning start and Step spans. Reuse compatible features; never silently truncate or replace unavailable attention.
- Keep `code/ours/` as a README-only placeholder until the researcher finalizes a method.
- Keep theory unpublished until suitable results exist.
- Keep external datasets, backbone weights, private credentials and machine-specific paths out of Git.
- Only detector weights accompany the published Streaming result. Upstream dependencies follow their own terms.
- Cleanup of old research assets requires a reviewed explicit target list and researcher approval.

Provenance: explicit researcher directions, recorded in the session trace and the selected experiment protocol. Earlier conflicting experimental defaults remain historical.
