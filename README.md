# Streaming HD

A research codebase for **hallucination detection across reasoning trajectories**.

Streaming HD studies hallucinations at three complementary levels: individual reasoning **Steps**, cumulative **Prefixes**, and final **Answers**. It brings together a multi-model, multi-domain dataset, baseline reproductions, and research records for investigating how hallucinations emerge, persist, and are corrected during reasoning.

## Repository structure

| Directory | Contents |
|---|---|
| [`code/`](code/) | Baseline implementations, feature extraction, utilities, tests, and a placeholder for the proposed method. |
| [`data/`](data/) | Dataset access and format documentation. |
| [`experiments/`](experiments/) | Experiment protocols, results, and detector checkpoints. |
| [`configs/`](configs/) | Method configurations and local path templates. |
| [`environment/`](environment/) | Dependencies and environment setup. |
| [`ara/`](ara/) | Research history and design decisions. |
| [`theory/`](theory/) | Reserved for theoretical analyses. |

## Getting started

Start with the [environment setup](environment/README.md), then follow the [code guide](code/README.md). Dataset conventions and access are described in the [data guide](data/README.md).

## Acknowledgments

This repository builds on prior hallucination-detection methods and their released implementations. Source attribution and applicable licenses are recorded in [THIRD_PARTY.md](THIRD_PARTY.md).
