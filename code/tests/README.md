# Verification

Run `python -m unittest discover -s code/tests -v` from the repository root. The suite needs PyTorch and NumPy; it does not download models or call paid services. It covers equations, gradients, metrics, checkpoints, identities, feature rejection and portable CPU execution.

[Validation record](../../ara/VALIDATION.json) binds the verified implementation hashes. Native full-model GPU extraction and complete method benchmarks remain separate checks.

The two external-dependency checks run when the pinned private HaMI/ARS sources are installed; otherwise they explicitly skip. The release validation ran both checks.
