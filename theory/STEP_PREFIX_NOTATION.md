# Step and Prefix: Notation and Error Probabilities

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $Q$ | The problem, including options or supplied materials |
| $s_1,\dots,s_T$ | Reasoning steps, segmented by the annotation manual |
| $X_{\le t}$ | Information visible up to step $t$: $Q$, $s_1,\dots,s_t$, and the model's internal signals (hidden states) on them |
| $X_t$ | Information of step $t$ only: $s_t$ and its internal signals |
| $Y_t\in\{0,1\}$ | Step label: whether step $t$ introduces a new committed error that cannot be attributed solely to inherited erroneous premises |
| $C_t\in\{0,1\}$ | Prefix label: whether, after step $t$, an unresolved error remains, i.e. one that is still endorsed or used and has not been explicitly retracted or replaced |
| $\pi^Y_t$ | Step error probability: $\pi^Y_t=P(Y_t=1\mid X_{\le t})$ |
| $\pi^C_t$ | Prefix error probability: $\pi^C_t=P(C_t=1\mid X_{\le t})$ |

**Meaning of probability.** Probabilities are taken over the data distribution. The randomness has three sources: the sampled problem, the reasoning produced by the generator, and the labels assigned by the Judge. Whatever is not covered by $X_{\le t}$ is the source of uncertainty in $\pi^Y_t$ and $\pi^C_t$.

**Manual constraint.** The only hard constraint in the manual is

$$
Y_t=1 \;\Rightarrow\; C_t=1 .
$$

---

## 2. Relation between the two error probabilities

### 2.1 Three mutually exclusive outcomes per step

By the manual constraint, the pair $(Y_t, C_t)$ takes exactly one of three values:

| Outcome | Labels | Probability |
|---|---|---|
| Step $t$ introduces a new error | $Y_t=1$ | $\pi^Y_t$ |
| No new error, but an earlier error is still active | $Y_t=0,\ C_t=1$ | $\psi_t$ |
| Clean | $C_t=0$ | $1-\pi^Y_t-\psi_t$ |

where

$$
\psi_t = P(Y_t=0,\,C_t=1\mid X_{\le t}).
$$

The three outcomes are exhaustive and mutually exclusive, so $\pi^Y_t+\psi_t+(1-\pi^Y_t-\psi_t)=1$.

Step is the first outcome; Prefix is the first two:

$$
\boxed{\;\pi^C_t = \pi^Y_t + \psi_t\;}
$$

All three probabilities are computable: every position's labels $(Y_t, C_t)$ fall into exactly one of the three classes.

### 2.2 Derivation over $Y_t$

$$
\begin{aligned}
\pi^C_t &= P(C_t=1\mid X_{\le t})\\[2pt]
&= P(C_t=1,\,Y_t=1\mid X_{\le t}) + P(C_t=1,\,Y_t=0\mid X_{\le t}) && \text{total probability over } Y_t\\[2pt]
&= P(Y_t=1\mid X_{\le t}) + P(C_t=1,\,Y_t=0\mid X_{\le t}) && \{Y_t=1\}\subseteq\{C_t=1\}\text{ (manual constraint)}\\[2pt]
&= \pi^Y_t + \psi_t && \text{definitions}
\end{aligned}
$$
