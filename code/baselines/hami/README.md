# HaMI

HaMI treats a generated token sequence as a multiple-instance bag and learns to
score its most informative tokens. The model and MIL loss are loaded from the
pinned official implementation; provenance and retrieval instructions are in
`upstream/PROVENANCE.json` and the repository's `THIRD_PARTY.md`.

For Step-level evaluation, each bag contains every native token of the current
Step in its causal context. Each model/domain is fitted independently. Layers and
weights are selected on complete VAL AUROC, with VAL AP as the tie criterion;
TEST is evaluated after selection and threshold freezing.

## Branch and token storage

The current recipe uses the official `ori` branch: unaugmented native tokens from
the current Step. Feature manifests declare `hami_mode: ori`; no semantic
probability or additional sampling is required. The separate `se` branch retains
its original `(1 + semantic_probability)` augmentation and requires real sampled
semantic-cluster probabilities. A branch mismatch is an error.

Ragged token bags are read lazily. Exact BF16 values may be stored as uint16 bit
patterns when `token_dtype: bfloat16_bits` is declared; these bits are reinterpreted
as BF16 before conversion to the detector's FP32 inputs. This avoids a full token
bank copy in RAM and preserves every token value. Other float feature banks remain
supported.
