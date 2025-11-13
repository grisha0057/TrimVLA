# TrimVLA

## 引言

- TrimVLA 是一个个人研究项目。项目受到理想汽车自动驾驶团队 [LightVLA](https://github.com/LiAutoAD/LightVLA) 的启发，探索另一种面向 VLA 模型的视觉 token 剪枝算法。
- 项目最初 fork 自 [LightVLA](https://github.com/LiAutoAD/LightVLA) 仓库。
- 项目处于非常初期的状态。受限于计算资源，暂未进行下一阶段的实验。
- 在小数据集、相同超参配置、早期训练阶段，TrimVLA 表现出了不输 LightVLA 的性能潜力。

## 实验结果

- **测试数据集**：Libero-spatial。共十个任务，每个任务尝试50次，共500次。
- **评估指标**：总体成功率。

### LightVLA vs TrimVLA 总体成功率对比

| Checkpoint | LightVLA 成功率 | 保留Token数 (mean/min/max) | TrimVLA 成功率 | 保留Token数 (我们手动设置最小保留10%, 即51个) |
|------------|-----------------|---------------------------|----------------|----------------------------------------------|
| Step 0     | 几乎是0         | -                         | 89.6% (448/500)| 51 (固定)                                     |
| Step 200   | 几乎是0         | -                         | 93.6% (468/500)| 51 (固定)                                     |
| Step 400   | 90.0% (450/500) | 55.2 / 24 / 107           | 94.4% (472/500)| 51 (固定)                                     |
| Step 800   | 68.8% (344/500) | 55.5 / 27 / 105           | 90.8% (454/500)| 51 (固定)                                     |

## 训练配置

- **硬件**：2 ✖️ 单卡48GB显存4090

### 基础超参数（TrimVLA, LightVLA 共享）

| 参数类别 | 参数名称 | 值 | 说明 |
|---------|---------|-----|------|
| 批次配置 | `batch_size` | 4 | |
| 批次配置 | `grad_accumulation_steps` | 4 | 有效批次 = 4 x 4 x 2 = 32 |
| 训练步数 | `step` | 800 | 总共训练800步 |
| 学习率 | `lr` | 1e-4 | 全程无衰减 |
| LoRA配置 | `lora_rank` | 8 | |
| LoRA配置 | `lora_dropout` | 0.0 | |
| 数据增强 | `image_aug` | True | |

### TrimVLA 相关超参数

| 参数类别 | 参数名称 | 值 | 说明 |
|---------|---------|-----|------|
| 图像-文本相关性聚合函数 | `prune_prompt_aggregation` | `logsumexp` | Logsumexp 在聚合相关性分数时,相比max, mean更平滑 |
| LogSumExp聚合的温度 | `prune_logsumexp_temperature` | `1.0` | |
| 最小保留比例 | `min_keep_ratio` | `0.1` | 最少保留10%的token,即51个,略少于LightVLA 的 |
| 累积概率 | `coverage` | `0.15` | 我们设置 coverage=min_keep_ratio + 0.05 |
| Gumble-softmax 温度 | `tau` | `1 -> 0.3 (600 steps)` | Gumble-softmax 温度衰减,共600steps |

## 关键发现

- **早期性能差异**：
  - LightVLA 在Step 0和Step 200时成功率几乎为0%，需要更多训练才能收敛。
  - TrimVLA 从Step 0开始就表现出较高的成功率（89.6%），收敛速度更快。

- **中期性能对比（Step 400）**：
  - LightVLA 达到90.0%成功率，保留平均55.2个tokens（范围：24-107）。
  - TrimVLA 达到94.4%成功率，固定保留51个tokens。
  - TrimVLA 在相似 token 数下可能表现更优。

- **"长期"训练（Step 800）**：
  - 两者都存在参数调度问题，性能掉点不能说明问题。目前还没有进行更多的调参。

- **Token保留策略**：
  - LightVLA采用动态token保留，平均约55个。
  - TrimVLA采用固定token保留策略（51个）。
  - 两者在token使用量上相近。

## 结论

在当前训练配置下，TrimVLA表现出以下潜能：

- ✅ 更快的收敛速度（Step 0即可达到89.6%）。
- ✅ 良好的峰值性能（Step 400达到94.4%）。
- ✅ 可以灵活设定保留的token数量，精确控制推理成本。

## TrimVLA 方法

整体流程：在进入第一层 Transformer 之前，对视觉 patch 和文本 prompt 进行相关性计算，根据覆盖率约束决定保留的 patch 数量，并在训练通过 gumble softmax 实现可微分优化。

### 1. 相关性分数计算

- **归一化**：对视觉 patch 隐状态与提示 token 隐状态分别施加 RMSNorm，统一数值尺度。
- **Q/K 投影**：复用第一层自注意力的Q, K投影矩阵，将视觉 patch 映射为 queries，将提示 token 映射为 keys。
- **注意力打分**：对每个head，计算每个 patch 对每个 token 的注意力分数。
- **Prompt 聚合**：对整个prompt的所有 token 聚合分数，使用带温度的 LogSumExp（更平滑、可整合多词线索，相比于 max, mean 更优）。
- **Multi-head 聚合**：对各注意力头的聚合结果取平均，得到每个视觉 patch 对当前 prompt 的相关性分数。

### 2. 基于覆盖率确定保留 patch 数

- **概率化**：对于每个视觉patch，将相关性分数经 softmax 得到保留概率 p.
- **覆盖率约束**：按 p 降序排序并做前缀累计，选取最小 k，使得累计概率达到目标覆盖率 coverage。
- **最小保留约束**：对 k 施加最小保留比例（min_keep_ratio）约束，防止极端情况。我们发现，通常这个值觉得决定了保留的 patch 数。比如 coverage=0.15，min_keep_ratio=0.1，那么最终保留的 patch 数是 512 * 0.1 = 51。

### 3. 训练侧（Gumbel-Softmax直通估计）

- 先按覆盖率策略求得 k 与 Top‑k 索引。对分数加入 Gumbel 噪声，温度 tau 控制平滑度。
- 前向使用硬 Top‑k 掩码实现离散选择，反向用直通估计。
- **稳定性措施**：冻结用于打分的第一层 Q/K 的 LoRA 适配器，抑制训练早期打分分布的抖动，让训练更稳。

### 4. 推理侧（硬剪枝）

- 执行硬剪枝：根据选定的 k 与 Top‑k 索引，物理删除未保留的视觉 patch，减少后续层的计算与显存占用。同步更新 position_ids 与 attention_mask 保持序列结构一致性。

## 研究局限性

1. 训练步数太短，batch size 不够大，超参数没有充分调优，只能初步验证模型的潜力。
2. 目前只在 Libero-Spatial 数据集上进行了训练和评测，没有在 Libero 的另外三个数据集上训练评测。
3. 用全量 Libero-Spatial 数据进行finetune并且在相同的数据集上评测在线成功率，是一种in-domain的评测，虽然在机器人领域比较常见，但是 ood 的表现仍是未知。尤其是在 base model openvla-oft 已经在 Libero 上表现非常好的情况下，算法的有效性和泛化性还需要进一步验证。

## 讨论

1. 为什么 TrimVLA 在step 0的表现就已经很好（保留10%的token，成功率89.6%）？我认为可能的原因是TrimVLA与 OpenVLA 训练/推理流程的对齐更好，可直接用于 LIBERO 这样的任务。
2. TODO: 与lightvla的比较
3. TODO: 在自动驾驶场景可能的应用

## checkpoints：
- **Huggingface**: https://huggingface.co/grisha0057
