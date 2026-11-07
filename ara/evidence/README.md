# Evidence index

The existing `tables/` and JSON records are retained as historical evidence. They concern the datasets/protocols named in their original records and must not be relabeled as results on the selected split.

- [Original evidence index](../imported_state/evidence/README.md).
- [Current Streaming result](../../experiments/streaming_622_p70/result/RESULTS.md).
- [Current metric values](../../experiments/streaming_622_p70/result/metrics.json).
- [Current split/data identities](../../experiments/streaming_622_p70/README.md).
- [External reference availability](EXTERNAL_REFERENCES.json).

`EXTERNAL_REFERENCES.json` inventories original-workspace references mentioned by the imported ARA. Available-but-unbundled files require the original workspace; missing references remain marked `not_found_in_original_workspace`. Some are directory prefixes or historical paths, not individual result files. No missing result is recreated from prose and no missing audit is labeled passed.

The complete available **record** is published; this does not claim that every historical experiment artifact still exists or is included in Git. Large feature banks, actual datasets and unpublished theory are not embedded in ARA.
