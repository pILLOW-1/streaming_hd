# A2 with post-error ranking

Keep original A2's fixed exponential readouts, prediction-weighted exclusive history and three-class cross-entropy. The only added component is a TRAIN loss.

Let H indicate an earlier observed Prefix=1, excluding the current position. Within H=1, N=(Step=1, Prefix=1), U=(0,1), and R=(0,0). True history is used only to group TRAIN positions for the loss; it never enters model inputs.

For final output logits sS=log(pN/(pU+pR)) and sP=log((pN+pU)/pR), add:

$$L=L_{A2}+\frac{\lambda}{4}\left[R_S(N,U)+R_S(N,R)+R_P(N,R)+R_P(U,R)\right].$$

Each R(A,B) averages softplus(sB-sA) over every available A×B pair in the original shuffled batch, including cross-trajectory pairs. An absent term contributes differentiable zero; the divisor remains four. Prefix ranking uses the final Prefix output, not the conditional head.

Choose lambda from {0.1, 0.3, 1.0} jointly with the layer and checkpoint using the original joint VAL rule. These are hyperparameter candidates for one method. Lambda=0 strictly reuses accepted original A2; it is not refitted. [Shared training and evaluation](training.md).
