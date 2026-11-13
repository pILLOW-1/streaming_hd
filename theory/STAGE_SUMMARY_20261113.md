# Stage Summary: How Step, Prefix and Answer Relate (2026-11-13)

This note connects the derived relation, the three motivation analyses and the resulting view of the task. It assumes the labels follow the annotation protocol. Notation follows [STEP_PREFIX_NOTATION.md](STEP_PREFIX_NOTATION.md).

## 1. Task and labels

At reasoning Step $t$ a detector sees only $X_{\le t}$ and outputs three judgements:

- **Step** $Y_t$: the Step introduces a new committed error. A valid inference from an inherited false premise, or a restatement of an inherited error, has $Y_t=0$.
- **Prefix** $C_t$: the reasoning so far contains at least one unresolved active error. An error stays active while the Prefix still endorses it or an active conclusion depends on it. It is resolved only after explicit correction, retraction or replacement, with no remaining dependent state.
- **Answer** $A$: the final answer is wrong, judged from the answer alone.

The only cross-label constraint is $Y_t=1\Rightarrow C_t=1$.

## 2. Derived relation

Each position is new ($Y=1$), inherited ($Y=0,C=1$) or clean ($C=0$). Hence

```math
\pi^C_t=\pi^Y_t+(1-\pi^Y_t)\,\kappa_t,\qquad \kappa_t=P(C_t=1\mid Y_t=0,X_{\le t}),
```

```math
\pi^A=\pi^C_T\,a^1+(1-\pi^C_T)\,a^0,\qquad a^c=P(A=1\mid C_T=c,X).
```

The relation links the three levels in one chain: introduction ($\pi^Y$), carry-over ($\kappa$), read-out ($a^1,a^0$). Equivalently, $1-\pi^C_t=(1-\pi^Y_t)(1-\kappa_t)$: the reasoning is clean only if no error is introduced now and no earlier error is carried over.

## 3. What the three analyses show about each term

| Analysis | Term tested | Result | Meaning |
|---|---|---|---|
| [prefix_answer_endpoint](../experiments/prefix_answer_endpoint_20261110/README.md) (15,697 trajectories) | $a^1,a^0$ | 35.5% of Answers are correct after an erroneous final Prefix; 18.6% are wrong after a clean one; 25.2% disagree overall | The Answer is a lossy read-out of the final reasoning state, not a copy |
| [step_prefix_composition](../experiments/step_prefix_composition_20261111/README.md) (TRAIN+VAL, 1.07M positions) | $\pi^Y$ vs $(1-\pi^Y)\kappa$ | 84.7% of Prefix positives are inherited; 72–88% in every combination | The erroneous state persists mostly by carry-over; a Step label sees only the moment of introduction |
| [prefix_recovery_diagnostic](../experiments/prefix_recovery_diagnostic_20261112/README.md) (five external detectors, VAL) | detectors' estimate of $\kappa$ | Inherited vs never-erroneous ≈ 0.96 AUROC; inherited vs recovered ≈ 0.74; within-trajectory mean 0.50–0.58 | The state also closes; detectors capture whether an error has occurred, not whether it is still active |

## 4. What the task is

**From the relation and the analyses.** Step, Prefix and Answer are three observations of one latent error state:

| Level | Observes | Misses |
|---|---|---|
| Step | increments: a new error enters | resolution and re-endorsement |
| Prefix | the state: some error is active | which error is active |
| Answer | the effect at the output | the process |

The state is introduced, carried over, closed and finally read out, with loss, into the answer.

**From the protocol's definitions.** Let $A_{it}=1$ when the error introduced at Step $i$ is still active at Step $t$. Then

```math
C_t=\bigvee_{i\le t}A_{it},\qquad A_{tt}=Y_t .
```

Step and Prefix are the same quantity at $i=t$ and over all $i\le t$. The levels interact in both directions: a new error opens the state, and whether a Step counts as a new error is judged relative to the inherited state. Activity is defined by endorsement and dependence, so it is relational, between a past error and the current reasoning, and non-monotone.

**Statement.**

> Streaming hallucination detection tracks how an error state evolves during reasoning from three complementary, incomplete observations: how errors are introduced, whether they remain active, and how they reach the answer. The largest and hardest part is deciding **whether past errors are still active now**, not merely whether an error has occurred.

**Mechanism (interpretation).** Errors arise at Steps and propagate along the dependence structure of the reasoning. The three labels read this structure at different positions and times. The structure is not annotated, so this layer motivates the view but is not a verified result.

## 5. What earlier methods miss

- **White-box hallucination detectors** (Streaming, SAPLMA, Global Mean, LLM-Check, ICR Probe) score the Prefix from a summary of the visible history. This summary retains evidence that an error has occurred but does not represent whether it was later resolved. Within a trajectory they separate the first error from earlier positions, but not still-active errors from corrected ones.
- **Reasoning-error detectors** such as process reward models aggregate Step scores by minimum or product, or locate only the first error. Both are monotone in the history.
- **Shared limitation.** The Prefix is effectively scored as whether an error has occurred. Since a re-endorsed error has $Y=0$, Step evidence alone can neither close the state nor reopen it.
