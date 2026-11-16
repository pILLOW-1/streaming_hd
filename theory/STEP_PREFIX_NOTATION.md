# Step, Prefix and Answer: Notation and Error Probabilities

---

## 1. Learning setup

**LLMs and token sequences.** We model an LLM as a conditional distribution $\mathbb{P}_{\theta}(\cdot)$ over token sequences, where $\theta$ denotes the model parameters. Given an input token sequence $\mathbf{Q}=[x_1,\dots,x_k]$, representing the problem together with any options or supplied materials, the LLM generates autoregressively a reasoning token sequence $\mathbf{R}=[x_{k+1},\dots,x_{k+m}]$ followed by a final-answer token sequence $\mathbf{A}=[x_{k+m+1},\dots,x_{k+m+l}]$. Each token is predicted from the preceding tokens, $\mathbb{P}_{\theta}(x_j\mid x_1,\dots,x_{j-1})$ for $j=k+1,\dots,k+m+l$.

**Reasoning steps and prefixes.** The annotation manual segments the reasoning into $T$ consecutive steps. With boundaries $k=i_0<i_1<\dots<i_T=k+m$, step $t$ is the token sequence

```math
\mathbf{S}_t=[x_{i_{t-1}+1},\dots,x_{i_t}],\qquad t=1,\dots,T ,
```

so that $\mathbf{R}=[\mathbf{S}_1,\dots,\mathbf{S}_T]$. The prefix up to step $t$ is $\mathbf{S}_{1:t}=[\mathbf{S}_1,\dots,\mathbf{S}_t]$, and $\mathbf{S}_{1:T}=\mathbf{R}$.

**Internal representations.** Let $L$ be the number of layers of $\mathbb{P}_{\theta}$ and $d$ its hidden dimension. For the input $(\mathbf{Q},\mathbf{S}_{1:t})$ with $i_t$ tokens, the hidden representation at layer $\ell\in\lbrace 1,\dots,L\rbrace$ is $\mathbf{H}^{\ell}_{\theta}(\mathbf{Q},\mathbf{S}_{1:t})\in\mathbb{R}^{i_t\times d}$, and $\mathbf{h}^{\ell}_{\theta,j}(\mathbf{Q},\mathbf{S}_{1:t})\in\mathbb{R}^{d}$ is the representation of the $j$-th token. For a fixed model these representations are functions of the token sequences.

**Labels.** Three binary labels are assigned by a Judge following the annotation manual; the value $1$ always denotes an error.

| Label | Level | Meaning |
|---|---|---|
| $y^{\mathrm{S}}_t\in\lbrace 0,1\rbrace$ | Step | The step introduces a new committed error that cannot be attributed solely to inherited erroneous premises |
| $y^{\mathrm{P}}_t\in\lbrace 0,1\rbrace$ | Prefix | After the step an unresolved error remains, i.e. one that is still endorsed or used and has not been explicitly retracted or replaced |
| $y^{\mathrm{A}}\in\lbrace 0,1\rbrace$ | Answer | The final answer $\mathbf{A}$ is wrong; the Answer Judge sees only the final answer |

All labels are written $y$; the superscript names the level: $\mathrm{S}$ for Step, $\mathrm{P}$ for Prefix and $\mathrm{A}$ for Answer. Every label is a value in $\lbrace 0,1\rbrace$ fixed by the trajectory; it is not a probability.

**Manual constraint.** The label definitions impose one hard constraint: a step that introduces a new error leaves an unresolved error.

```math
y^{\mathrm{S}}_t=1 \;\Rightarrow\; y^{\mathrm{P}}_t=1 .
```

**Domain and dataset.** Let $Q$ be a random variable representing problems and $\mathbb{P}_{Q}$ its distribution. For a sampled problem $\mathbf{Q}\sim\mathbb{P}_{Q}$ the LLM generates $(\mathbf{R},\mathbf{A})\sim\mathbb{P}_{\theta}(\cdot\mid\mathbf{Q})$, and the Judge assigns the labels. This yields the dataset

```math
\mathcal{D}=\Big\lbrace\big(\mathbf{Q}^{n},\,\mathbf{R}^{n},\,\mathbf{A}^{n},\,y^{\mathrm{S},n}_{1:T_n},\,y^{\mathrm{P},n}_{1:T_n},\,y^{\mathrm{A},n}\big)\Big\rbrace_{n=1}^{N},
```

where $T_n$ is the number of steps of the $n$-th trajectory.

**Problem 1 (Streaming hallucination detection).** Given the dataset $\mathcal{D}$ constructed from $\mathbb{P}_{Q}$ and the LLM $\mathbb{P}_{\theta}$, the goal is to learn three detectors $D^{\mathrm{S}}$, $D^{\mathrm{P}}$ and $D^{\mathrm{A}}$ whose output probabilities match the labels.

- While the reasoning is generated, at every step $t$ the Step detector $D^{\mathrm{S}}$ and the Prefix detector $D^{\mathrm{P}}$ read $(\mathbf{Q},\mathbf{S}_{1:t})$ and the internal representations of $\mathbb{P}_{\theta}$ on them.
- After the final answer is generated, the Answer detector $D^{\mathrm{A}}$ reads $(\mathbf{Q},\mathbf{R},\mathbf{A})$ and the internal representations of $\mathbb{P}_{\theta}$ on them.

**Detector probabilities.** The labels are $0$ or $1$, so there is no separate true probability to estimate. All probabilities in this document are outputs of the detectors. Each detector defines a predictive distribution over labels given what it has read: $\mathbb{P}_{D^{\mathrm{S}}}$ over $y^{\mathrm{S}}_t$; $\mathbb{P}_{D^{\mathrm{P}}}$ jointly over $(y^{\mathrm{S}}_t,y^{\mathrm{P}}_t)$, taking the probability of $y^{\mathrm{S}}_t$ from the Step detector and giving zero probability to the pair excluded by the manual constraint; and $\mathbb{P}_{D^{\mathrm{A}}}$ jointly over $(y^{\mathrm{P}}_T,y^{\mathrm{A}})$, taking the probability of $y^{\mathrm{P}}_T$ from the Prefix detector. The three error probabilities are

```math
\pi^{\mathrm{S}}_t=\mathbb{P}_{D^{\mathrm{S}}}(y^{\mathrm{S}}_t=1\mid \mathbf{Q},\mathbf{S}_{1:t}),\qquad
\pi^{\mathrm{P}}_t=\mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{P}}_t=1\mid \mathbf{Q},\mathbf{S}_{1:t}),\qquad
\pi^{\mathrm{A}}=\mathbb{P}_{D^{\mathrm{A}}}(y^{\mathrm{A}}=1\mid \mathbf{Q},\mathbf{R},\mathbf{A}).
```

---

## 2. Relation between the Step and Prefix error probabilities

### 2.1 Three mutually exclusive outcomes per step

By the manual constraint, the pair $(y^{\mathrm{S}}_t, y^{\mathrm{P}}_t)$ takes exactly one of three values:

| Outcome | Labels | Probability under $D^{\mathrm{P}}$ |
|---|---|---|
| The step introduces a new error | $y^{\mathrm{S}}_t=1$ | $\pi^{\mathrm{S}}_t$ |
| No new error, but an earlier error is still unresolved | $y^{\mathrm{S}}_t=0,\ y^{\mathrm{P}}_t=1$ | $\psi_t$ |
| Clean | $y^{\mathrm{P}}_t=0$ | $1-\pi^{\mathrm{S}}_t-\psi_t$ |

where

```math
\psi_t = \mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{S}}_t=0,\,y^{\mathrm{P}}_t=1\mid \mathbf{Q},\mathbf{S}_{1:t}).
```

The three outcomes are exhaustive and mutually exclusive. Step is the first outcome; Prefix is the first two:

```math
\boxed{\;\pi^{\mathrm{P}}_t = \pi^{\mathrm{S}}_t + \psi_t\;}
```

All three probabilities can be supervised directly: every position's labels $(y^{\mathrm{S}}_t, y^{\mathrm{P}}_t)$ fall into exactly one of the three classes.

### 2.2 Derivation over $y^{\mathrm{S}}_t$

All probabilities in this derivation are conditional on $(\mathbf{Q},\mathbf{S}_{1:t})$, which is omitted for brevity.

```math
\begin{aligned}
\pi^{\mathrm{P}}_t &= \mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{P}}_t=1)\\[2pt]
&= \mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{P}}_t=1,\,y^{\mathrm{S}}_t=1) + \mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{P}}_t=1,\,y^{\mathrm{S}}_t=0) && \text{total probability over } y^{\mathrm{S}}_t\\[2pt]
&= \mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{S}}_t=1) + \mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{P}}_t=1,\,y^{\mathrm{S}}_t=0) && \text{manual constraint}\\[2pt]
&= \pi^{\mathrm{S}}_t + \mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{S}}_t=0)\,\mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{P}}_t=1\mid y^{\mathrm{S}}_t=0) && \text{definition of } \pi^{\mathrm{S}}_t \text{ and chain rule}\\[2pt]
&= \pi^{\mathrm{S}}_t + (1-\pi^{\mathrm{S}}_t)\,\kappa_t && \text{definition of } \kappa_t
\end{aligned}
```

where

```math
\kappa_t = \mathbb{P}_{D^{\mathrm{P}}}(y^{\mathrm{P}}_t=1\mid y^{\mathrm{S}}_t=0,\,\mathbf{Q},\mathbf{S}_{1:t})
```

is the probability that an earlier error is still unresolved, given that step $t$ introduces no new error.

```math
\boxed{\;\pi^{\mathrm{P}}_t = \pi^{\mathrm{S}}_t + (1-\pi^{\mathrm{S}}_t)\,\kappa_t\;}
```

---

## 3. Relation between the Prefix and Answer error probabilities

The Answer error probability is linked to the Prefix label at the end of reasoning, $y^{\mathrm{P}}_T$.

### 3.1 Four outcomes for the answer

The manual imposes no hard constraint between $y^{\mathrm{P}}_T$ and $y^{\mathrm{A}}$, so all four combinations can occur:

| Outcome | Labels | Probability under $D^{\mathrm{A}}$ |
|---|---|---|
| An unresolved error remains, and the answer is wrong | $y^{\mathrm{P}}_T=1,\ y^{\mathrm{A}}=1$ | $\pi^{\mathrm{P}}_T\, \alpha_1$ |
| An unresolved error remains, but it does not reach the answer | $y^{\mathrm{P}}_T=1,\ y^{\mathrm{A}}=0$ | $\pi^{\mathrm{P}}_T(1-\alpha_1)$ |
| Reasoning is clean, yet the answer is wrong | $y^{\mathrm{P}}_T=0,\ y^{\mathrm{A}}=1$ | $(1-\pi^{\mathrm{P}}_T)\,\alpha_0$ |
| Reasoning is clean, and the answer is correct | $y^{\mathrm{P}}_T=0,\ y^{\mathrm{A}}=0$ | $(1-\pi^{\mathrm{P}}_T)(1-\alpha_0)$ |

where, for $c\in\lbrace 0,1\rbrace$,

```math
\alpha_c = \mathbb{P}_{D^{\mathrm{A}}}(y^{\mathrm{A}}=1\mid y^{\mathrm{P}}_T=c,\,\mathbf{Q},\mathbf{R},\mathbf{A})
```

is the probability of a wrong answer given the terminal Prefix label.

### 3.2 Derivation over $y^{\mathrm{P}}_T$

All probabilities in this derivation are conditional on $(\mathbf{Q},\mathbf{R},\mathbf{A})$, which is omitted for brevity.

```math
\begin{aligned}
\pi^{\mathrm{A}} &= \mathbb{P}_{D^{\mathrm{A}}}(y^{\mathrm{A}}=1)\\[2pt]
&= \mathbb{P}_{D^{\mathrm{A}}}(y^{\mathrm{P}}_T=1,\,y^{\mathrm{A}}=1) + \mathbb{P}_{D^{\mathrm{A}}}(y^{\mathrm{P}}_T=0,\,y^{\mathrm{A}}=1) && \text{total probability over } y^{\mathrm{P}}_T\\[2pt]
&= \mathbb{P}_{D^{\mathrm{A}}}(y^{\mathrm{P}}_T=1)\,\alpha_1 + \mathbb{P}_{D^{\mathrm{A}}}(y^{\mathrm{P}}_T=0)\,\alpha_0 && \text{chain rule and the definition of } \alpha_c\\[2pt]
&= \pi^{\mathrm{P}}_T\, \alpha_1 + (1-\pi^{\mathrm{P}}_T)\,\alpha_0 && \text{definition of } \pi^{\mathrm{P}}_T
\end{aligned}
```

```math
\boxed{\;\pi^{\mathrm{A}} = \pi^{\mathrm{P}}_T\, \alpha_1 + (1-\pi^{\mathrm{P}}_T)\,\alpha_0\;}
```

Prefix has the same form over $y^{\mathrm{S}}_t$, $\pi^{\mathrm{P}}_t=\pi^{\mathrm{S}}_t\cdot 1+(1-\pi^{\mathrm{S}}_t)\,\kappa_t$, where the first conditional is fixed to $1$ by the manual constraint. Answer has no such constraint, so both conditionals $\alpha_1$ and $\alpha_0$ remain free.
