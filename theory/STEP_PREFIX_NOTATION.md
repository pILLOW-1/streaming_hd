# Step, Prefix and Answer: Notation and Error Probabilities

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $Q$ | The problem, including options or supplied materials |
| $s_1,\dots,s_T$ | Reasoning steps, segmented by the annotation manual |
| $X_{\le t}$ | Information visible up to step $t$: $Q$, $s_1,\dots,s_t$, and the model's internal signals (hidden states) on them |
| $X_t$ | Information of step $t$ only: $s_t$ and its internal signals |
| $Y_t\in\lbrace 0,1\rbrace$ | Step label: whether step $t$ introduces a new committed error that cannot be attributed solely to inherited erroneous premises |
| $C_t\in\lbrace 0,1\rbrace$ | Prefix label: whether, after step $t$, an unresolved error remains, i.e. one that is still endorsed or used and has not been explicitly retracted or replaced |
| $\pi^Y_t$ | Step error probability: $\pi^Y_t=P(Y_t=1\mid X_{\le t})$ |
| $\pi^C_t$ | Prefix error probability: $\pi^C_t=P(C_t=1\mid X_{\le t})$ |
| $X_A$ | Information of the final answer: its text and internal signals |
| $A\in\lbrace 0,1\rbrace$ | Answer label: whether the final answer is wrong; the Answer Judge sees only the final answer |
| $\pi^A$ | Answer error probability: $\pi^A=P(A=1\mid X_{\le T}, X_A)$ |

**Meaning of probability.** Probabilities are taken over the data distribution. The randomness has three sources: the sampled problem, the reasoning produced by the generator, and the labels assigned by the Judge. Whatever is not covered by $X_{\le t}$ is the source of uncertainty in $\pi^Y_t$ and $\pi^C_t$.

**Manual constraints.** The label definitions impose two hard constraints:

```math
Y_t=1 \;\Rightarrow\; C_t=1 ,
```

```math
C_{t-1}=0 \;\Rightarrow\; C_t=Y_t ,\qquad C_0=0 .
```

The second holds because, when the prefix before step $t$ has no unresolved error, any unresolved error after step $t$ cannot be inherited, so it is a new error of step $t$. The empty prefix is clean.

---

## 2. Relation between the two error probabilities

### 2.1 Three mutually exclusive outcomes per step

By the first manual constraint, the pair $(Y_t, C_t)$ takes exactly one of three values:

| Outcome | Labels | Probability |
|---|---|---|
| Step $t$ introduces a new error | $Y_t=1$ | $\pi^Y_t$ |
| No new error, but an earlier error is still active | $Y_t=0,\ C_t=1$ | $\psi_t$ |
| Clean | $C_t=0$ | $1-\pi^Y_t-\psi_t$ |

where

```math
\psi_t = P(Y_t=0,\,C_t=1\mid X_{\le t}).
```

The three outcomes are exhaustive and mutually exclusive, so $\pi^Y_t+\psi_t+(1-\pi^Y_t-\psi_t)=1$.

Step is the first outcome; Prefix is the first two:

```math
\boxed{\;\pi^C_t = \pi^Y_t + \psi_t\;}
```

All three probabilities are computable: every position's labels $(Y_t, C_t)$ fall into exactly one of the three classes.

### 2.2 Derivation over $Y_t$

```math
\begin{aligned}
\pi^C_t &= P(C_t=1\mid X_{\le t})\\[2pt]
&= P(C_t=1,\,Y_t=1\mid X_{\le t}) + P(C_t=1,\,Y_t=0\mid X_{\le t}) && \text{total probability over } Y_t\\[2pt]
&= P(Y_t=1\mid X_{\le t}) + P(C_t=1,\,Y_t=0\mid X_{\le t}) && \{Y_t=1\}\subseteq\{C_t=1\}\text{ (first constraint)}\\[2pt]
&= \pi^Y_t + \psi_t && \text{definitions}\\[2pt]
&= \pi^Y_t + P(Y_t=0\mid X_{\le t})\,P(C_t=1\mid Y_t=0,\,X_{\le t}) && \text{chain rule}\\[2pt]
&= \pi^Y_t + (1-\pi^Y_t)\,\kappa_t && \text{definitions}
\end{aligned}
```

where

```math
\kappa_t = P(C_t=1\mid Y_t=0,\,X_{\le t})
```

is the probability that an earlier error is still active, given that step $t$ introduces no new error.

```math
\boxed{\;\pi^C_t = \pi^Y_t + (1-\pi^Y_t)\,\kappa_t\;}
```

---

## 3. Answer error probability

The Answer error probability is conditioned on all visible information, the reasoning and the final answer:

```math
\pi^A = P(A=1\mid X_{\le T}, X_A).
```

It is linked to the Prefix label at the end of reasoning, $C_T$. Below, "$\cdot$" abbreviates the condition $X_{\le T}, X_A$.

### 3.1 Four outcomes for the answer

The manual imposes no hard constraint between $C_T$ and $A$, so all four combinations can occur:

| Outcome | Labels | Probability |
|---|---|---|
| An unresolved error remains, and the answer is wrong | $C_T=1,\ A=1$ | $q^A a^1$ |
| An unresolved error remains, but it does not reach the answer | $C_T=1,\ A=0$ | $q^A(1-a^1)$ |
| Reasoning is clean, yet the answer is wrong | $C_T=0,\ A=1$ | $(1-q^A)\,a^0$ |
| Reasoning is clean, and the answer is correct | $C_T=0,\ A=0$ | $(1-q^A)(1-a^0)$ |

where

```math
q^A = P(C_T=1\mid \cdot),\qquad a^1 = P(A=1\mid C_T=1,\,\cdot),\qquad a^0 = P(A=1\mid C_T=0,\,\cdot).
```

### 3.2 Derivation over $C_T$

```math
\begin{aligned}
\pi^A &= P(C_T=1,\,A=1\mid \cdot) + P(C_T=0,\,A=1\mid \cdot) && \text{total probability over } C_T\\[2pt]
&= P(C_T=1\mid \cdot)\,P(A=1\mid C_T=1,\,\cdot) + P(C_T=0\mid \cdot)\,P(A=1\mid C_T=0,\,\cdot) && \text{chain rule}\\[2pt]
&= q^A a^1 + (1-q^A)\,a^0 && \text{definitions}\\[2pt]
&\approx \pi^C_T\, a^1 + (1-\pi^C_T)\,a^0 && X_A \text{ adds little information about } C_T
\end{aligned}
```

```math
\boxed{\;\pi^A \approx \pi^C_T\, a^1 + (1-\pi^C_T)\,a^0\;}
```

The approximation holds because the answer mostly restates the conclusion of the reasoning; it fails when the answer itself revises the reasoning.

Prefix has the same form over $Y_t$, $\pi^C_t=\pi^Y_t\cdot 1+(1-\pi^Y_t)\,\kappa_t$, where the first conditional is fixed to $1$ by the manual constraint $Y_t=1\Rightarrow C_t=1$. Answer has no such constraint, so both conditionals $a^1$ and $a^0$ remain free.
