# RHD with corrected official attention-layer selection

The prior port read attention from its depth anchors. This successor follows the actual released offset rule after native-depth mapping: `ell - min(mapped_anchors) + 1`. Genuine global full-attention layers are selected separately; Gemma sliding and Qwen linear layers are excluded, with nearest preceding full layer or first full layer as fallback. This is an explicit cross-backbone adaptation of the original Qwen7B/14B implementation.

Depth/logit semantics, four component formulas, both candidate families, seven weights, complete ownVAL selection and threshold, and selected TEST once are unchanged. Missing Step depth/entropy/masks are obtained from the same necessary corrected-attention forward; no hidden-state bank is copied or persisted. Accepted results include independent component witnesses, IDs/labels, candidate/threshold and saved-score metric checks.

See [configuration](config.json) and [results](result/accepted_cells.json). Historical prior-attention records remain in the original random622 result directory. No data, model weights, feature banks or predictions are published.
