# Last token for Step and Prefix

For Step t, use its final token representation for both readouts:

$$e_t=c_t=h_{t,\mathrm{last}},\qquad q_t=\sigma(w_Y^\top e_t+b_Y).$$

Keep the exclusive, trajectory-local A2 memory over all earlier original Steps:

$$g_t=\sum_{i<t}\mathrm{stopgrad}(q_i)e_i,\quad \kappa_t=\sigma(b+\theta^\top c_t+w^\top g_t),\quad p_t^P=q_t+(1-q_t)\kappa_t.$$

Three class probabilities are q, (1-q)κ and (1-q)(1-κ). Optimize their original C18 cross-entropy jointly. Only historical q is detached. [Shared training protocol](training.md).

Logic: L34, epoch 5. Factuality: L47, epoch 3. Both completed 64-layer selection and one selected TEST.
