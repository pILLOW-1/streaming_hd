# LLM-Check

The raw attention statistic is the sum over heads of the mean log causal attention diagonal, matching upstream `get_attn_eig_prod`. The official evaluation notebook **negates** this statistic for hallucination-positive labels. Therefore the released recipe uses `score_sign: -1` before VAL layer selection and threshold calibration.

Source: [official evaluation notebook at the pinned revision](https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection/blob/2f3bf9ea6db19e60a416090f77694816c92a9146/check_scores_fava_annotated.ipynb).

Prefix adaptation: retain native prompt conditioning and include prompt tokens in the full visible-prefix statistic. Only genuine attention layers are eligible; linear-attention blocks are excluded. Select the layer using own VAL AUROC, then VAL AP; select the operating threshold by own VAL F1max. Never choose score direction using TEST.
