# Stage Summary: How Step, Prefix and Answer Relate (2026-11-16)

This note connects the derived relation, the three motivation analyses, the resulting view of the task and the research plan. It assumes the labels follow the annotation protocol. Notation follows [STEP_PREFIX_NOTATION.md](STEP_PREFIX_NOTATION.md).

## 1. Task and labels

At reasoning step $t$ a detector reads only the problem $\mathbf{Q}$ and the steps generated so far $\mathbf{S}_{1:t}$, with the model's internal representations on them. It has to predict three labels:

- **Step** $y^{\mathrm{S}}_t$: the step introduces a new committed error. A valid inference from an inherited false premise, or a restatement of an inherited error, has $y^{\mathrm{S}}_t=0$.
- **Prefix** $y^{\mathrm{P}}_t$: the reasoning so far contains at least one unresolved error. An error stays unresolved while the Prefix still endorses it or a conclusion that is still in use depends on it. It is resolved only after explicit correction, retraction or replacement, with no remaining dependent state.
- **Answer** $y^{\mathrm{A}}$: the final answer is wrong, judged from the answer alone.

The only cross-label constraint is $y^{\mathrm{S}}_t=1\Rightarrow y^{\mathrm{P}}_t=1$.

## 2. Derived relation

Each position is new ($y^{\mathrm{S}}_t=1$), inherited ($y^{\mathrm{S}}_t=0,\ y^{\mathrm{P}}_t=1$) or clean ($y^{\mathrm{P}}_t=0$). A clean position is fresh if no earlier position of its trajectory has $y^{\mathrm{P}}=1$, and recovered otherwise. Hence the detector probabilities satisfy

```math
\pi^{\mathrm{P}}_t=\pi^{\mathrm{S}}_t+(1-\pi^{\mathrm{S}}_t)\,\kappa_t,\qquad
\pi^{\mathrm{A}}=\pi^{\mathrm{P}}_T\,\alpha_1+(1-\pi^{\mathrm{P}}_T)\,\alpha_0 .
```

Here $\kappa_t$ is the probability that an earlier error is still unresolved given no new error at step $t$, and $\alpha_c$ is the probability of a wrong answer given that the terminal Prefix label is $c$. All symbols are defined in the notation document.

The relation links the three levels in one chain: introduction ($\pi^{\mathrm{S}}$), an earlier error remaining unresolved ($\kappa$), read-out ($\alpha_1,\alpha_0$). Equivalently, $1-\pi^{\mathrm{P}}_t=(1-\pi^{\mathrm{S}}_t)(1-\kappa_t)$: the reasoning is clean only if no error is introduced now and no earlier error remains unresolved.

## 3. What the three analyses show about each term

| Analysis | Term tested | Result | Meaning |
|---|---|---|---|
| [prefix_answer_endpoint](../experiments/prefix_answer_endpoint_20261110/README.md) (15,697 trajectories) | $\alpha_1,\alpha_0$ | 35.5% of Answers are correct after an erroneous final Prefix; 18.6% are wrong after a clean one; 25.2% disagree overall | The Answer is a lossy read-out of the final reasoning state, not a copy |
| [step_prefix_composition](../experiments/step_prefix_composition_20261111/README.md) (15,697 trajectories, 1.45M positions) | $\pi^{\mathrm{S}}$ vs $(1-\pi^{\mathrm{S}})\kappa$ | 84.7% of Prefix positives are inherited; 72–88% in every combination | Most Prefix errors are earlier errors that remain unresolved; a Step label sees only the moment of introduction |
| [prefix_recovery_diagnostic](../experiments/prefix_recovery_diagnostic_20261112/README.md) (five external detectors, TEST) | detectors' estimate of $\kappa$ | Inherited vs fresh ≈ 0.93 AUROC; inherited vs recovered ≈ 0.66; within-trajectory mean 0.51–0.58 | Errors also get resolved; detectors capture whether an error has occurred, not whether it is still unresolved |

## 4. What the task is

**From the relation and the analyses.** Step, Prefix and Answer are three observations of one latent error state:

| Level | Observes | Misses |
|---|---|---|
| Step | increments: a new error enters | resolution and re-endorsement |
| Prefix | the state: some error is unresolved | which error is unresolved |
| Answer | the effect at the output | the process |

An error is introduced, stays unresolved or is resolved, and is finally read out, with loss, into the answer.

**From the protocol's definitions.** The levels interact in both directions: a new error opens the state, and whether a step counts as a new error is judged relative to the inherited state. Whether an error is unresolved is defined by endorsement and dependence, so it is relational, between a past error and the current reasoning, and non-monotone.

**Statement.**

> Streaming hallucination detection tracks how an error state evolves during reasoning from three complementary, incomplete observations: how errors are introduced, whether they remain unresolved, and how they reach the answer. The largest and hardest part is deciding **whether past errors are still unresolved now**, not merely whether an error has occurred.

**Mechanism (interpretation).** Errors arise at steps and propagate along the dependence structure of the reasoning. The three labels read this structure at different positions and times. The structure is not annotated, so this layer motivates the view but is not a verified result.

## 5. What earlier methods miss

- **White-box hallucination detectors** (Streaming, SAPLMA, Global Mean, LLM-Check, ICR Probe) score the Prefix from a summary of the visible history. This summary retains evidence that an error has occurred but does not represent whether it was later resolved. Within a trajectory they separate the first error from earlier positions, but not unresolved errors from recovered positions.
- **Reasoning-error detectors** such as process reward models aggregate Step scores by minimum or product, or locate only the first error. Both are monotone in the history.
- **Shared limitation.** The Prefix is effectively scored as whether an error has occurred. Since a re-endorsed error has $y^{\mathrm{S}}_t=0$, Step evidence alone can tell neither that an error has been resolved nor that it has become unresolved again.

## 6. Research plan

**One framework, explored in stages.** The three detectors output the three error probabilities and are tied together by the relation of Section 2.

The Step read-out and the Prefix term $\kappa$ are explored separately, each with its upstream part frozen, and are joined only through this relation. They are not trained jointly: Prefix labels add no information about the Step label once it is observed, and the paired runs so far show no gain for Step from Prefix gradients.

| Line | What is explored | Frozen input | Goal |
|---|---|---|---|
| Step | The read-out that turns the tokens of a step into $\pi^{\mathrm{S}}_t$ | none | Exceed every Step baseline |
| Prefix | The design of $\kappa_t$ | $\pi^{\mathrm{S}}_t$ from the Step line | Separate inherited from recovered positions; after composing with the Step read-out, exceed every Prefix baseline |
| Answer | How $\alpha_1,\alpha_0$ are trained | $\pi^{\mathrm{P}}_T$ from the Prefix line | Exceed every Answer baseline |

**Step.** Find a read-out that is better than the last token, fixed pooling and learned token selection used by the baselines. No single baseline is best across generators and domains, and averaging their scores is consistently higher than the best single one, so the existing read-outs each capture part of the evidence. Candidate inputs are explored one at a time against fixed pooling under one training recipe.

**Prefix.** $\kappa_t$ answers one question: given no new error at step $t$, is an earlier error still unresolved? It has two targets, checked in this order.

1. *Inherited versus recovered.* Within a trajectory, positions whose error is still unresolved must score above recovered positions, whose earlier error has been resolved. This is the within-trajectory recovery AUROC of the [recovery diagnostic](../experiments/prefix_recovery_diagnostic_20261112/README.md), on which every external detector is near chance.
2. *Overall Prefix detection.* After composing $\kappa_t$ with the frozen Step read-out, the Prefix score must exceed every Prefix baseline.

$\kappa$ is trained with the Prefix cross-entropy of the composed score on all positions, with no gradient into the Step read-out. This matches inference, where $\kappa_t$ is also used at positions whose new error the Step read-out is unsure about.

**Answer.** The terminal Prefix score of the finished Prefix line is plugged into the Answer relation. With constant $\alpha_1,\alpha_0$ the Answer score is a monotone function of $\pi^{\mathrm{P}}_T$ and equals plain Prefix transfer, so any gain must come from $\alpha_1,\alpha_0$ that depend on the trajectory. How to train them is still open: each is supervised only by the trajectories with one value of the terminal Prefix label, which leaves few examples of the minority outcome in some generator/domain combinations.

**Protocol shared by the three lines.**

- One data pipeline and one set of splits; each line saves its scores for every position so that the lines can be composed without retraining.
- Design choices use VAL only. A finalized design is evaluated once on TEST.
- Several seeds per configuration; differences are judged against seed variation and trajectory-bootstrap intervals.
- Prefix results are reported overall, by position type (new, inherited, recovered, fresh) and within trajectories.

**Merge point.** When the three lines are finalized, the full chain is composed from our own Step, Prefix and Answer parts and compared with the baselines on all three levels. A jointly fine-tuned version is run once as an ablation.

**Open questions.**

- How to train $\alpha_1$ and $\alpha_0$.
- How the Prefix state evolves over time. The only label constraint links Step and Prefix at the same position, so a recursion over steps needs an explicit assumption on whether a resolved error can become unresolved again without a new Step error.
