# Step, Prefix and Answer: Notation and Error Probabilities

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $X_0$ | The problem, including options or supplied materials, with the model's internal signals (hidden states) on it |
| $S_t$ | Text of reasoning step $t$, $t=1,\dots,T$, segmented by the annotation manual |
| $X_t$ | Observation of step $t$: the text $S_t$ and the model's internal signals on it |
| $X_{0:t}$ | Observed history up to step $t$: $X_{0:t}=(X_0,X_1,\dots,X_t)$ |
| $Y_t\in\lbrace 0,1\rbrace$ | Step label: whether step $t$ introduces a new committed error that cannot be attributed solely to inherited erroneous premises |
| $C_t\in\lbrace 0,1\rbrace$ | Prefix label: whether, after step $t$, an unresolved error remains, i.e. one that is still endorsed or used and has not been explicitly retracted or replaced |
| $\pi^Y_t$ | Step error probability: $\pi^Y_t=P(Y_t=1\mid X_{0:t})$ |
| $\pi^C_t$ | Prefix error probability: $\pi^C_t=P(C_t=1\mid X_{0:t})$ |
| $X_A$ | Observation of the final answer: its text and the model's internal signals on it |
| $A\in\lbrace 0,1\rbrace$ | Answer label: whether the final answer is wrong; the Answer Judge sees only the final answer |
| $\pi^A$ | Answer error probability: $\pi^A=P(A=1\mid X_{0:T}, X_A)$ |

**Convention.** Upper-case letters are random variables and lower-case letters are their values. Every conditional probability names its conditioning variables explicitly. A quantity such as $\pi^Y_t=P(Y_t=1\mid X_{0:t})$ is itself a random variable, a function of the history; its value at an observed history is $\pi^Y_t(x_{0:t})=P(Y_t=1\mid X_{0:t}=x_{0:t})$. A detector is any function of the same variables, for example $q_t=q(X_{0:t})$.

**Meaning of probability.** Probabilities are taken over the data distribution. The randomness has three sources: the sampled problem, the reasoning produced by the generator, and the labels assigned by the Judge. Whatever is not determined by $X_{0:t}$ is the source of uncertainty in $\pi^Y_t$ and $\pi^C_t$.

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
\psi_t = P(Y_t=0,\,C_t=1\mid X_{0:t}).
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
\pi^C_t &= P(C_t=1\mid X_{0:t})\\[2pt]
&= P(C_t=1,\,Y_t=1\mid X_{0:t}) + P(C_t=1,\,Y_t=0\mid X_{0:t}) && \text{total probability over } Y_t\\[2pt]
&= P(Y_t=1\mid X_{0:t}) + P(C_t=1,\,Y_t=0\mid X_{0:t}) && \{Y_t=1\}\subseteq\{C_t=1\}\text{ (first constraint)}\\[2pt]
&= \pi^Y_t + \psi_t && \text{definitions}\\[2pt]
&= \pi^Y_t + P(Y_t=0\mid X_{0:t})\,P(C_t=1\mid Y_t=0,\,X_{0:t}) && \text{chain rule}\\[2pt]
&= \pi^Y_t + (1-\pi^Y_t)\,\kappa_t && \text{definitions}
\end{aligned}
```

where

```math
\kappa_t = P(C_t=1\mid Y_t=0,\,X_{0:t})
```

is the probability that an earlier error is still active, given that step $t$ introduces no new error.

```math
\boxed{\;\pi^C_t = \pi^Y_t + (1-\pi^Y_t)\,\kappa_t\;}
```

---

## 3. Answer error probability

The Answer error probability conditions on the whole trajectory, the reasoning and the final answer. Write $Z=(X_{0:T},X_A)$ for these conditioning variables:

```math
\pi^A = P(A=1\mid Z).
```

It is linked to the Prefix label at the end of reasoning, $C_T$.

### 3.1 The answer adds no information about the terminal Prefix label

The Prefix label is assigned from the problem and the reasoning steps only; the Prefix Judge does not see the final answer. Given the reasoning, the remaining randomness of $C_T$ is the Judge's own, which is independent of the answer. Hence

```math
C_T \perp X_A \mid X_{0:T},
\qquad\text{so}\qquad
P(C_T=1\mid Z)=P(C_T=1\mid X_{0:T})=\pi^C_T .
```

### 3.2 Four outcomes for the answer

The manual imposes no hard constraint between $C_T$ and $A$, so all four combinations can occur:

| Outcome | Labels | Probability given $Z$ |
|---|---|---|
| An unresolved error remains, and the answer is wrong | $C_T=1,\ A=1$ | $\pi^C_T\, a^1$ |
| An unresolved error remains, but it does not reach the answer | $C_T=1,\ A=0$ | $\pi^C_T(1-a^1)$ |
| Reasoning is clean, yet the answer is wrong | $C_T=0,\ A=1$ | $(1-\pi^C_T)\,a^0$ |
| Reasoning is clean, and the answer is correct | $C_T=0,\ A=0$ | $(1-\pi^C_T)(1-a^0)$ |

where

```math
a^1 = P(A=1\mid C_T=1,\,Z),\qquad a^0 = P(A=1\mid C_T=0,\,Z).
```

### 3.3 Derivation over $C_T$

```math
\begin{aligned}
\pi^A &= P(C_T=1,\,A=1\mid Z) + P(C_T=0,\,A=1\mid Z) && \text{total probability over } C_T\\[2pt]
&= P(C_T=1\mid Z)\,P(A=1\mid C_T=1,\,Z) + P(C_T=0\mid Z)\,P(A=1\mid C_T=0,\,Z) && \text{chain rule}\\[2pt]
&= \pi^C_T\, a^1 + (1-\pi^C_T)\,a^0 && \text{Section 3.1 and definitions}
\end{aligned}
```

```math
\boxed{\;\pi^A = \pi^C_T\, a^1 + (1-\pi^C_T)\,a^0\;}
```

Prefix has the same form over $Y_t$, $\pi^C_t=\pi^Y_t\cdot 1+(1-\pi^Y_t)\,\kappa_t$, where the first conditional is fixed to $1$ by the manual constraint $Y_t=1\Rightarrow C_t=1$. Answer has no such constraint, so both conditionals $a^1$ and $a^0$ remain free.
