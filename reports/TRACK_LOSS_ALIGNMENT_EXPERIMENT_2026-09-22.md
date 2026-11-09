# Track loss 与正式 EPE 对齐实验（2026-09-22）

## 1. 为什么做

此前五目标训练中的 Track 项与正式评测不完全一致。旧目标把所有可见点混在一起，并用 20 px 的平滑上限压缩大误差；正式评测则先删掉有效点不足 20 个的目标帧，再对其余 `known & visible` 点计算未截断 EPE。因此先做 Track-only 对照，确认 loss 本身能否把正式 EPE 推坏，再决定是否接回五目标。

## 2. 两个 Track 目标

旧目标（`capped_all`）：

\[
D_{\text{old}}=20\cdot\operatorname{mean}\left[\tanh\left(\frac{\lVert \hat p-p^{gt}\rVert_2}{20}\right)\right].
\]

新目标（`eval_epe`）：

\[
D_{\text{track}}=
\operatorname{mean}_{t:n_t\geq20,\;j\in\text{known}\cap\text{visible}}
\lVert \hat p_{t,j}-p^{gt}_{t,j}\rVert_2.
\]

优化方向都是最大化 damage。新目标与 `eval_universal_four_tasks.track_stats` 的筛帧和 EPE 定义一致。

## 3. 300 步 A/B 设置

- 场景：`rgbd_dataset_freiburg3_sitting_xyz`
- 训练池：序列前半段 608 帧；每步随机抽 10 帧
- 载体：v5 显示器世界坐标四边形；平均覆盖率约 3.3%
- 初始化：同一 warning 纹理
- 任务：仅 Track
- EOT：关闭
- 更新数：300；两组使用同一随机种子和抽帧顺序
- 正式未见评测：后半段 608 个唯一帧，121 个 10 帧滑窗；Track 使用正式 reprojection-GT EPE；Camera Pose 拼成完整轨迹后只做一次 Sim(3)

## 4. 固定训练审计结果

固定审计由前半段 4 组互不重叠的 10 帧构成，共 40 帧；它是训练端诊断，不是未见帧结果。

| 纹理 | Track EPE | 相对初始纹理 | 相对 Clean | visF1 |
|---|---:|---:|---:|---:|
| Clean（无 patch） | 2.3060 | — | — | 0.8429 |
| 初始 warning 纹理 | 2.2237 | — | -3.57% | 0.8788 |
| 旧 `capped_all`，300 步 | 2.2273 | +0.16% | -3.41% | 0.8725 |
| 新 `eval_epe`，300 步 | 2.2516 | **+1.25%** | -2.36% | 0.8726 |

解释：新 loss 的优化方向是对的，相比同一初始纹理确实更明显地提高了正式 EPE；但 300 步增幅不足以抵消 warning 初始化本身对 Track 的改善，因此仍未超过 Clean。

## 5. 未见半段正式结果

| 训练目标 | ATE | RPE-t | RPE-r (deg) | AbsRel | Point Acc | Point Comp | Track EPE | visF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.165746 | 0.005169 | 0.334840 | 0.076888 | 0.083655 | 0.099457 | 2.140283 | 0.956871 |
| 初始 warning 纹理（未优化） | 0.173782 | 0.005218 | 0.332778 | 0.076445 | 0.082284 | 0.098693 | 2.114876 | 0.956917 |
| 旧 `capped_all` | 0.171794 | 0.005220 | 0.333537 | 0.076962 | 0.082673 | 0.099186 | 2.121885 | 0.956976 |
| 新 `eval_epe` | 0.172676 | 0.005295 | 0.333488 | 0.077075 | 0.082816 | 0.099522 | 2.122361 | 0.956985 |

初始 warning 纹理本身把 Track EPE 从 2.140283 降到 2.114876，即它在 Track 上先产生了 1.19% 的改善。300 步的新 `eval_epe` 把 EPE 从同一初始纹理的 2.114876 提到 2.122361（+0.35%），说明优化方向在未见帧上也有少量传递；但相对 Clean 仍低 0.84%，尚未形成正式 Track 攻击。因此当前不能把新 Track loss 直接接回五目标并宣称问题解决。

## 6. 自适应收敛的决定性对照

1. **已完成：**初始 warning 纹理在同一未见半段上的正式结果。它确认初始化本来就使 Track 变好，因此必须按“相对初始纹理的优化增量”和“最终是否超过 Clean”两层判断。
2. **已完成：**新 `eval_epe` 的自适应收敛训练跑满 3000 步，每 100 步在固定训练审计集上评估。固定审计的 Track/Clean 比值从 update 0 的 0.975、update 300 的 0.987，升到 update 600 的 1.007、update 1100 的 1.046，最终在 update 3000 达到 **1.463**。四个审计子集的 EPE 均高于各自 Clean，说明训练目标已经充分优化，不是 loss 符号、300 步不足或单个子集偶然拉高。
3. **决策：暂不接回五目标。**训练审计 EPE 虽然明显超过 Clean，但未见半段的 Track EPE 没有恶化；继续把该项加入五目标只能证明训练帧可拟合，不能支撑跨帧 Track 攻击主张。

### update 700 中期检查（不是最终模型）

为避免等到最大 3000 步才发现泛化方向错误，从 checkpoint 导出 update 700 时的最佳审计纹理，并在后半段完整 121 窗上做了一次中期检查：

| 纹理 | 训练审计 Track/Clean | 后半段 Track EPE | 相对初始 warning | 相对 Clean |
|---|---:|---:|---:|---:|
| Clean | 1.000× | 2.140283 | — | — |
| 初始 warning | 0.975× | 2.114876 | — | -1.19% |
| 自适应最佳（update 700） | 1.011× | 2.131841 | +0.80% | -0.39% |

中期结论：延长训练让正式 EPE 的恶化从 300 步时相对初始纹理的 +0.35% 增至 +0.80%，但后半段仍未超过 Clean。它支持“新 loss 有效但跨半段衰减明显”，不支持“Track 攻击已经成功”。自适应训练继续进行；最终只采用早停/最大步数结束后由固定训练审计选出的最佳纹理，不按后半段结果挑 checkpoint。

### 3000 步最终最佳纹理

最终纹理由固定训练审计选出，没有按后半段结果挑 checkpoint。正式评测覆盖后半段 608 个唯一帧、121 个 10 帧滑窗；Camera Pose 拼成完整轨迹后只做一次 Sim(3)。

| 指标（越大越差，visF1 除外） | Clean | Attack | 相对变化 |
|---|---:|---:|---:|
| Camera ATE (m) | 0.165746 | 0.169088 | +2.02% |
| RPE translation (m) | 0.005169 | 0.005202 | +0.62% |
| RPE rotation (deg) | 0.334840 | 0.334005 | -0.25%（改善） |
| Depth AbsRel | 0.076888 | 0.109187 | **+42.01%** |
| Point Acc | 0.083655 | 0.126775 | **+51.54%** |
| Point Comp | 0.099457 | 0.109632 | **+10.23%** |
| Track EPE (px) | 2.140283 | 2.138454 | -0.09%（基本不变） |
| Track visibility F1 | 0.956871 | 0.956866 | -0.0004%（不变） |

最终解释：Track-only 优化不是“什么都没学到”。同一纹理在未见半段明显破坏了 depth 和 point map，却没有让正式 Track EPE 变差。这说明当前 Track 目标在训练帧上可被充分拟合，但该 Track 误差方向没有跨帧迁移；纹理对共享稠密几何表示的副作用反而更容易迁移。训练侧 1.463× 与未见侧 0.999× 的差距排除了“再多跑一点就能解决”的解释。

### 新发现的采样偏差（需要再做一次修复实验）

当前动态采样代码先从 608 帧池中均匀抽 10 帧，再按时间排序，并固定把排序后的第 0 帧作为 Track query。这样目标帧覆盖全池，但 query 并不均匀：它永远是十帧中的最早帧。对 608 选 10 而言，query 的期望索引约为 54，中位索引为 40，约 94.5% 的 query 落在训练池前 25%，95% 分位也只到索引 156（前 25.8%）。

因此本轮结果严格证明的是“偏向训练池前部 query 的 Track 优化不迁移”，还不能证明“均匀 query 训练必然不迁移”。下一轮应先均匀选 query frame，再独立抽 9 个 target frame，并把 query 明确放在输入第 0 位；GT pool 按该 query gather。其余初始化、载体、学习率、自适应审计和后半段评测全部保持不变。这个对照优先级高于继续调 Track 权重或增加步数。

## 7. 结果和日志绝对路径

- 旧 loss 训练审计：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_capped_all_s0_u300/training_track_audit/track_training_audit.csv`
- 新 loss 训练审计：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_s0_u300/training_track_audit/track_training_audit.csv`
- 初始纹理训练审计：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_s0_u300/training_track_audit_initial/track_training_audit.csv`
- 旧 loss 未见结果：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_capped_all_s0_u300/heldout_w10_s5_v5_controlled/four_tasks/four_task_by_scene.csv`
- 新 loss 未见结果：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_s0_u300/heldout_w10_s5_v5_controlled/four_tasks/four_task_by_scene.csv`
- 自适应训练日志：`/mnt/data/wangqq/vggt/logs/track_objective_ablation/eval_epe_adaptivefix_umax3000.log`
- 自适应固定审计：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_adaptivefix_s0_umax3000/geometry_patch/audit_history.jsonl`
- update 700 最佳纹理的中期未见评测：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_adaptivefix_best_interim/heldout_w10_s5_v5_controlled/four_tasks/four_task_by_scene.csv`
- 3000 步最终未见评测：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_adaptivefix_s0_umax3000/heldout_w10_s5_v5_controlled/four_tasks/four_task_by_scene.csv`
- 初始纹理未见结果：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_initial_warning_control/heldout_w10_s5_v5_controlled/four_tasks/four_task_by_scene.csv`
- 初始纹理未见评测日志：`/mnt/data/wangqq/vggt/logs/controlled_baselines/eval/trackonly_v5_k10_firsthalf_sitting_xyz_initial_warning_control.log`

## 8. Track query 均匀采样修复实验（2026-09-23，已完成）

旧动态采样先从 608 帧中抽 10 帧，再按时间排序并把最早帧作为 Track query。20,000 次固定种子模拟显示，旧 query 有 94.58% 落在训练池前四分之一，四个四分位计数为 18915 / 1061 / 24 / 0。

新模式 `uniform_query` 先从完整训练池均匀抽取 query，再从其余帧独立抽 9 张 target/context，并把 query 固定放在 VGGT 输入第 0 位。相同模拟下，四个四分位计数为 5108 / 4898 / 5072 / 4922，前四分之一占 25.54%。固定自适应审计不需要修改：它原有四个 query 为 0 / 152 / 304 / 456，已经均匀覆盖 608 帧。

除 query 采样外，其余设置全部保持不变：`sitting_xyz` 前半段 608 帧、每步 10 帧、Track-only、正式 `eval_epe` loss、warning 初始化、v5 显示器载体、no-EOT、学习率 0.002、自适应调度、最大 3000 步、每 100 步固定审计。训练结束后自动在后半段 608 个唯一帧上做完整四任务测评。

- 单元测试：15/15 通过。
- 采样诊断：`/mnt/data/wangqq/vggt/reports/track_query_sampling_distribution.json`
- 运行目录：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_uniformquery_adaptive_s0_umax3000`
- 训练日志：`/mnt/data/wangqq/vggt/logs/track_objective_ablation/eval_epe_uniformquery_adaptive_umax3000.log`
- 自动测评日志：`/mnt/data/wangqq/vggt/logs/track_objective_ablation/eval_epe_uniformquery_adaptive_eval_watch.log`

训练跑满 3000 步。固定训练审计上的 Track/Clean 从初始的 0.975× 升到 1.409×，四个审计子集全部高于各自 Clean，说明新采样下 Track loss 仍能被充分优化。

后半段正式测评覆盖 608 个唯一帧、121 个 10 帧滑窗，Camera Pose 拼成完整轨迹后只做一次 Sim(3)：

| 指标（除 d1/visF1 外越大越差） | Clean | 均匀 query Attack | 相对变化 |
|---|---:|---:|---:|
| Camera ATE (m) | 0.165746 | 0.174244 | +5.13% |
| RPE translation (m) | 0.005169 | 0.005187 | +0.34% |
| RPE rotation (deg) | 0.334840 | 0.334590 | -0.07%（改善） |
| Depth AbsRel | 0.076888 | 0.097920 | +27.35% |
| Depth δ1 | 0.932311 | 0.891728 | -4.35%（恶化） |
| Point Acc | 0.083655 | 0.108780 | +30.03% |
| Point Comp | 0.099457 | 0.098396 | -1.07%（改善） |
| Track EPE (px) | 2.140283 | 2.133073 | -0.34%（改善） |
| Track visibility F1 | 0.956871 | 0.956860 | -0.001%（基本不变） |

相对同一张未优化 warning 纹理的 Track EPE 2.114876，优化后升到 2.133073（+0.86%），说明优化方向有少量跨半段传递；但它仍未超过 Clean 2.140283，不能构成正式 Track 攻击。旧偏置采样 3000 步的未见 Track EPE 为 2.138454，同样没有超过 Clean；新均匀采样反而略弱。

因此可以排除“Track 失败只是因为 query 有 94.5% 集中在训练池前四分之一”。更准确的结论是：无论 query 偏置还是均匀覆盖，Track-only loss 都能显著拟合训练审计，却不能在后半段未见帧上使正式 Track EPE 超过 Clean。纹理的副作用仍会迁移到 depth 和部分 point 指标，但 Track 目标本身不迁移。

- 最终四任务 CSV：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_uniformquery_adaptive_s0_umax3000/heldout_w10_s5_v5_controlled/four_tasks/four_task_by_scene.csv`
- 固定审计历史：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_uniformquery_adaptive_s0_umax3000/geometry_patch/audit_history.jsonl`

## 9. 完整训练池诊断与多 query 鲁棒目标（2026-09-23）

为区分“只拟合了固定 40 帧审计组合”和“整个训练分布有效、但不能迁移到后半段”，将均匀 query 的 3000 步成品 patch 在前半段全部 608 个训练帧上重新评测。评测使用 121 个 10 帧、步长 5 的窗口；该诊断显式开启 `--allow_training_frames`，正式未见帧评测仍默认禁止训练帧。

| 指标 | Clean | Attack | 相对变化 |
|---|---:|---:|---:|
| Track EPE (px) | 1.42 | 1.46 | 约 +2.8% |
| Depth AbsRel | 0.0817 | 0.1058 | 约 +29.5% |
| Point Acc | 0.1664 | 0.1988 | 约 +19.5% |

训练池 Track EPE 只出现很弱的恶化，而后半段 Track EPE 从 2.140283 降到 2.133073。说明该纹理并非完全只记住固定审计组合，但 Track 方向在训练分布中的一致性本来就很弱，到未见时间段后消失；与此同时 Depth/Point 副作用可以明显迁移。

进一步核查启动链后确认：Python 默认的 `dynamic_sets_per_step=6` 被 per-scene 启动器覆盖成了 1。因此上一轮 uniform-query 训练每个 optimizer update 实际只使用一组 query，并未完成多 query 联合优化。

本轮新增两个严格对照，均使用相同的 warning 初始化、v5 显示器载体、前半段 608 帧、10 帧输入、no-EOT、seed 0、自适应调度和 3000 步上限：

1. `multiquery6_mean`：每步独立抽 6 个均匀 query，六组梯度简单平均；
2. `multiquery6_robust`：每步独立抽 6 个均匀 query，组间对 Track EPE 做 soft-min；每组内部再对合格 target frame 的逐帧 EPE 做 soft-min。两个温度均为 0.25 px。

两步端到端冒烟通过。鲁棒版第 1 步的最弱 query 组获得 52.8% 权重、最强组仅 0.1%；第 2 步最弱组获得 82.6% 权重，说明更新确实集中在当前最难攻击的 query/帧组，而不是退化成普通平均。

- 训练池诊断 CSV：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_eval_epe_uniformquery_adaptive_s0_umax3000/trainpool_w10_s5_v5_controlled/four_tasks/four_task_by_scene.csv`
- 多 query 平均日志：`/mnt/data/wangqq/vggt/logs/track_multiquery_ablation/trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_mean_adaptive_s0_umax3000.log`
- 多 query 鲁棒日志：`/mnt/data/wangqq/vggt/logs/track_multiquery_ablation/trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_robust_adaptive_s0_umax3000.log`

状态：两组正式自适应训练已启动；最终仍须在后半段 608 个唯一帧上评测，不能按训练审计选择结论。

## 10. 30 帧窗口、相邻重叠 10 帧的完整轨迹重评（2026-09-24）

为避免旧版 10 帧窗口、步长 5 的拼接过度依赖短窗口，本轮按已经确定的正式口径重新评测：每个窗口一次向 VGGT 输入 30 帧，常规步长为 20，因此相邻窗口名义上共享 10 帧；利用这些公共帧估计相邻窗口间的 Sim(3)，将所有窗口按时间顺序拼成一条 608 帧轨迹，最后只做一次全局 Sim(3) 对齐并计算 CameraHead ATE/RPE。Depth、Point map 和 Track 仍按原四任务口径在全部窗口上汇总。

本轮没有重新训练纹理，而是使用已有的 10 帧训练成品 `trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_mean_interim_best_u2700`，检验同一张纹理从 10 帧训练输入迁移到 30 帧推理输入后的效果。因此它是“拼接协议验证 + 输入长度迁移诊断”，不能替代后续匹配的 30 帧重新训练实验。

### 10.1 协议检查

- 序列：`rgbd_dataset_freiburg3_sitting_xyz`
- 测试集：后半段 608 个唯一帧，索引 608--1215，与训练帧零交集
- 窗口：30 个，每个 30 帧
- 常规窗口：步长 20，相邻窗口重叠 10 帧
- 尾窗：为保留最后 2 帧，最后一个窗口向前回退 2 帧，因此最后一对窗口重叠 12 帧；没有重复计入最终逐帧指标
- Camera pose 正式来源：CameraHead-only；点图辅助位姿仅保留为诊断，不进入主表
- EOT：关闭，与该纹理原训练设置一致

旧版 10 帧/步长 5 的 CameraHead clean ATE 为 0.165746 m；改成 30 帧/步长 20 后为 0.074348 m，降低 55.1%。这说明更长窗口显著改善了 clean 轨迹估计和跨窗拼接稳定性，旧 10 帧结果确实受到短窗口协议影响。

### 10.2 四任务结果

| 指标 | Clean | Attack | 相对变化 | 是否攻击成功 |
|---|---:|---:|---:|---|
| Camera ATE (m) | 0.074348 | 0.055733 | -25.04% | 否，反而改善 |
| RPE translation (m) | 0.007120 | 0.007422 | +4.24% | 很弱 |
| RPE rotation (deg) | 0.332757 | 0.336677 | +1.18% | 很弱 |
| Depth AbsRel | 0.073427 | 0.142340 | +93.85% | 是 |
| Depth delta1 | 0.933371 | 0.888703 | -4.79% | 是，delta1 越低越差 |
| Point Acc | 0.068765 | 0.153482 | +123.22% | 是 |
| Point Comp | 0.083016 | 0.175004 | +110.81% | 是 |
| Track EPE (px) | 2.683020 | 2.751943 | +2.57% | 仅有弱信号 |
| Track visibility F1 | 0.940644 | 0.940769 | +0.013% | 否，基本不变 |

点图辅助位姿诊断为 clean ATE 0.049705 m、attack ATE 0.049455 m，几乎没有变化。该数字不作为 Camera pose 结果，只用来确认“选择哪一套位姿输出”不会改变本轮位姿攻击失败的结论。

### 10.3 结论

在更稳定的 30 帧窗口协议下，当前纹理对 Depth 和 Point map 仍然有很强的攻击效果，说明这两项不是旧 10 帧拼接造成的假象。Camera pose 的 ATE 从 0.074348 m 降到 0.055733 m，只有 RPE 出现 1%--4% 的轻微恶化，因此不能算位姿攻击成功。

Track EPE 从 2.683020 px 升到 2.751943 px，恶化 2.57%。这比旧 10 帧评测中接近 0 的变化更积极，但仍小于项目此前测得的约 7% 自然波动量级，也没有 visibility F1 的配套恶化，所以目前只能称为弱信号，不能作为稳定 Track 攻击结论。下一步必须用匹配的 30 帧输入重新训练，并补多个种子及未优化纹理基线，才能判断它是否可重复。

### 10.4 文件位置

- 30 帧正式四任务 CSV：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_mean_interim_best_u2700/heldout_w30_s20_v5_controlled/four_tasks/four_task_by_scene.csv`
- CameraHead-only 位姿 CSV：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_mean_interim_best_u2700/heldout_w30_s20_v5_controlled/camera_only_pose/camera_pose_stitched_summary.csv`
- 点图辅助位姿诊断：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_mean_interim_best_u2700/heldout_w30_s20_v5_controlled/four_tasks/dense_point_assisted_pose_diagnostic.csv`
- 完整评测日志：`/mnt/data/wangqq/vggt/logs/controlled_baselines/eval_w30_s20/trackonly_v5_k10_firsthalf_sitting_xyz_multiquery6_mean_interim_best_u2700.log`
- 30 帧评测启动脚本：`/mnt/data/wangqq/vggt/scripts/runs/run_controlled_baseline_eval_w30_s20.sh`

按协议更正要求，当前实验原有的 `heldout_w10_s5_v5_controlled` 目录和对应旧评测日志已删除，共约 55 MB。训练纹理、checkpoint、GT 和数据集均保留，因此旧结果可以重新生成，但已删除文件本身不能直接恢复。
