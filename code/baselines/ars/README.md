# ARS

The pinned upstream source, projection objective and supervised probe are recorded in `upstream/PROVENANCE.json`. Obtain the original dependency as described in the repository's third-party instructions.

## Auxiliary agreement in the selected experiment

The current 6:2:2 experiment has an explicit amendment: generated variants with an invalid official Yes/No response are isolated from auxiliary projection training. Every original trajectory remains in supervised training and evaluation. Valid `True` and `False` agreement labels are retained. Original judgments and failed responses are preserved.

`auxiliary_selection.select_auxiliary` validates an ordered list of judgment records, each containing an original reference and four generated variants. It returns retained source positions, original trajectory indices, Boolean agreements and isolation records. Use these positions to filter `variants_x`, `variant_original_id` and `variant_agreement`. Keep the complete original `x` arrays, supervised labels and split IDs.

Projection groups can contain fewer than five observations. The projection loss, optimizer, epoch budget and downstream probe retain the experiment's original recipe. Computational failures remain subject to their separate recovery checks. A full identity audit and a frozen selection hash precede each independent model/domain fit.
