# Environment

Python 3.11 is recommended. The exported detector files were loaded and checked with PyTorch 2.8.0+cu128. Use a CUDA-compatible PyTorch installation for your GPU, for example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements.txt
```

For CPU checks, a CPU build of PyTorch 2.8.0 is sufficient. Native extraction requires a GPU and local backbone snapshots, whose identities are recorded in `configs/model_revisions.json`. The original extraction metadata records Transformers 5.17.0 and Tokenizers 0.23.2. Exact numerical equality across different hardware/runtime builds is not promised. Preserve the model/tokenizer revision, native input IDs, dtype and pooling policy.

`requirements.txt` pins scientific packages for which recorded versions are available. Dataset transport and plotting dependencies are intentionally unpinned; this is not advertised as a complete lockfile. The original runtime used a vLLM v0.30.0 image with separate detector dependencies; the multi-gigabyte image is not stored in Git.

Plain command-line tools are scheduler-independent. On a cluster, run GPU commands inside your scheduler allocation. Configure CUDA_VISIBLE_DEVICES according to that scheduler. No local credentials, PBS owner, hostname or compute-node path is required by the portable Streaming workflow.

## RACE auxiliary runtime

```bash
python -m pip install -r environment/requirements-race.txt
python -m spacy download en_core_web_trf
```

The RACE wrapper checks that all three auxiliary models, the spaCy model and the extraction prompt are loaded. Set their local paths in `race_assets`; the repository does not distribute them. The optional dependency file is not a version-locked environment. HaMI/ARS upstream preparation scripts include their original dependency specifications in the externally obtained pinned source directories (see THIRD_PARTY.md).
