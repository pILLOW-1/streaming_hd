# Qwen Step–Prefix 关系联合检测：固定 Readout、预测头与损失

## Motivation

单独检测 Step 或 Prefix 时，容易忽略两者的标签逻辑关系：新错误一定使当前 Prefix 含有未解决错误。我们用一个三状态分布联合表达“新错误／无新错误但旧错误仍活跃／干净”，令 Step 概率为第一类、Prefix 概率为前两类之和。本实验先研究固定读出下的可检测性，以及线性/MLP、边缘 BCE/三类 CE 的影响；不预设哪种组合更好，也不据此声称已经优于独立检测。

## Configuration

| 项目 | 固定设置 |
|---|---|
| 模型与数据 | 冻结 Qwen3.8-27B（当前模型资产），Math、Logic、Science、Multi-hop、Factuality 各独立训练；沿用当前622_p70 TRAIN/VAL/TEST及标签，不重新划分 |
| Readout | Last token；Streaming global mean；Streaming Step指数聚合 exp(3u)；Streaming Prefix指数聚合 exp(.003j)；全部原始FP32向量，不做L2归一化 |
| 可见信息 | 截至当前Step的原始native因果上下文；聚合排除prompt/最终Answer，保留原始推理中的符号与间隔，严格应用原BPE边界override |
| 预测头 | Linear D→3；MLP D→256→128→64→3，ReLU；无额外归一化/Dropout |
| 输出 | Softmax(N,I,K)；p(Step)=p(N)，p(Prefix)=p(N)+p(I) |
| 损失 | 等权边缘BCE：0.5[BCE(Y,pN)+BCE(C,pN+pI)]；或纯三分类CE，(1,1)→N、(0,1)→I、(0,0)→K；实际发现冲突则停止该域并报告，不改标签/删题 |
| 训练 | Adam lr=1e-3、weight_decay=0；batch256有效位置；每epoch同seed42不放回打乱，位置等权、无类别重采样；最多100epochs、patience10 |
| 选权重与层 | 每配置独立训练native层1–64；完整VAL的Step/Prefix AUROC算术平均最大，完全并列依次看平均AP、更低平均边缘BCE；仍并列保留并报告，不任意选层 |
| 阈值与TEST | 两任务各自VAL F1max阈值，依次precision及更大阈值打破并列，严格score>threshold；联合checkpoint冻结后各TEST一次；仅保存预测做独立CPU验收，不额外模型replay |
| 规模 | 4readouts ×2heads ×2losses ×5domains=80配置，分别输出Step/Prefix结果；不包含Answer、多模型或额外种子 |

MLP结构参考SAPLMA；本实验的Adam学习率、batch、训练预算和early stopping是统一设计选择，不声称完整复现SAPLMA训练recipe。早停“改善”使用上述完整VAL联合选择顺序。

## Result protocol

当前已验收 **0 / 80**。每个配置通过保存预测的ID/标签/覆盖、有限值、概率归一性、阈值及CPU指标验收后，立即写入`result/<domain>/<readout>/<head>/<loss>.json`并更新`result/index.json`及本页。报告AUROC、AP、TPR@1%FPR、ACC、F1、precision、recall、FPR、confusion、正负覆盖、选层/epoch/阈值与配置/数据/checkpoint/预测hash；缺项不填造。注册、排队、训练或VAL冻结不计完成。

代码位于`code/ours/step_prefix_relation_22/`，配置在本目录`config.json`。运行日志和训练权重保留在HPC独立run，特征在受保护assets/scratch；不上传原始样本、标签数组、特征或权重。严格hash兼容复用已有Last-token/Global-mean缓存，只补缺失池化。每个配置的64层在一个连续PBS入口执行，独立参数/optimizer/RNG，断点续接；不启动额外服务或远程Codex会话。


## 已完成配置

| Domain | Readout | Head | Loss | Layer | Step AUROC/AP | Prefix AUROC/AP |
|---|---|---|---|---:|---:|---:|

尚无已验收结果；未完成配置不填入数值。
