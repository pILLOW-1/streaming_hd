# v1 + conditional ranking

Start from [v1](../v1/README.md). Add 0.1 times a pairwise logistic loss to the conditional logit. Sample up to 32 trajectories per update. Within each, compare all inherited vs recovered positions with no current Step error after an observed earlier error, allowing multiple earlier errors. Average pairs within trajectories and then across trajectories. Match the v1 BCE batch sequence and training update count per layer. Select on composed Prefix VAL AUROC, AP, and lower BCE, then evaluate the frozen winner once on TEST.

For recovery AUROC, compare the conditional scores on the same recoverable within-trajectory pairs, then average the trajectory AUROCs. Recover the v1 conditional probability from the saved composed score and frozen Step probability wherever the latter is below one. See the [run contract](../../code/kappa_ranking/contract.json) and [implementation](../../code/kappa_ranking/ranking.py).
