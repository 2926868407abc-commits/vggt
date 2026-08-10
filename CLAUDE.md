# MIRAGE-3D / VGGT 几何一致物理补丁攻击

## 项目一句话

在 3D 场景的真实平面上定义一块物理补丁，用相机内外参投影到每一帧（跨视角几何一致），
优化其纹理使 VGGT 的 pose / depth / point map 输出退化。目标会议 ICLR。

## 环境与路径

- 仓库根：`/mnt/data/wangqq/vggt`
- 评测仓库：`/mnt/data/wangqq/recons_eval`
- 攻击 env：`/mnt/data/wangqq/conda_envs/vggt/bin/python3`
- 评测 env：`/mnt/data/wangqq/conda_envs/recons_eval/bin/python3`（与攻击 env 隔离，不要混用）

## 关键文件

| 文件 | 作用 |
|---|---|
| `attack_vggt_geometry_tum10.py` | 主攻击脚本（约 3300 行），几何放置 + 纹理优化 |
| `run_geometry_aware_tum10.sh` | 单次实验入口，全部超参走环境变量 |
| `run_tum10_aor_monitor_patch.sh` | 显示器贴面实验组（物理保守设定） |
| `scripts/eval_vggt_tum_pose_for_recons_eval_tum10.py` | ATE/RPE 评测 |
| `scripts/diag_gauge_invariance.py` | 规范不变性诊断（只读，调用 recons_eval 的度量函数） |
| `scripts/eval_gauge_absorbed_frac.py` | 每个 run 的 `ate_ceiling` / `ate_frac_of_ceiling` / `gauge_absorbed_frac`，挂在 `RUN_GAUGE_DIAG` 上 |
| `tests/test_pose_scale_invariant_mse.py` | 尺度不变 pose loss 的单元测试 |
| `tests/test_pose_gauge_losses.py` | pairwise / aligned_residual 两个 loss 的单元测试 |

## 主脚本内的关键函数（行号以当前版本为准，改动后请重新定位）

- `homography_from_points` (~L325) — 平面到图像的单应
- `build_geometry_arrays_for_plane` (~L1425) — 逐帧构建 grid / mask，含 z-buffer 可见性
- `apply_geometry_patch` (~L2181) — 可微渲染合成
- `prepare_texture_for_render` (~L2160) — EOT（目前仅光度）
- `normalize_c2w_to_first` (~L2255) — 轨迹归一化到第 0 帧
- `pose_relative_mse` (~L2339) — **当前 pose loss，有已知缺陷，见下**
- `attack_objective_loss` (~L2370) — loss 分发
- `load_tum_sequence` (~L2426) — 每次迭代重新加载序列与几何
- `patch_regularization_terms` (~L2619) — TV / 可打印 / 自然参考
- `train_geometry_patch` (~L2642) — 训练主循环

## 当前实验设定

- 场景：`rgbd_dataset_freiburg3_sitting_static`（TUM fr3），10 帧
  ⚠️ **这个序列本身是退化的，见已知问题 7。新实验请至少同时跑 sitting_xyz。**
- 载体：右侧电脑显示器屏幕，`PLANE_MODE=depth_manual_quad_surface`
- 主 loss：`ATTACK_LOSS=pose_gt_untargeted`
- 可选 loss：`pose_aligned_residual_mse`（可微 ATE，三序列里两个最强，见问题 1）/
  `pose_scale_invariant_mse`（尺度不变版）/ `pose_pairwise_relative_mse`（**已判定无效**）/
  `feature_l1` / `pose_clean_untargeted` / `pose_{reverse,drift,scale,yaw}_targeted`

### 三序列 loss 对照（1000 步，ATE 占各自 ceiling 的比例，越高越强）

| 序列 | ceiling | clean | 随机 | 旧 `pose_gt_untargeted` | `scale_invariant` | `pairwise` | `aligned_residual` |
|---|---|---|---|---|---|---|---|
| sitting_xyz | 0.2005 | 3.9% | 93.8% | 6.2% | 76.2% | 5.9% | **84.8%** |
| sitting_halfsphere | 0.2519 | 3.2% | 90.5% | **89.2%** | — | — | 44.2%（EOT 关） |
| sitting_static | 0.0310 | 19.5% | 90.3% | 78.5% | 63.6% | 60.9% | **95.2%** |

**没有一个 loss 普遍占优，排名随序列翻转。** 旧 loss 在 sitting_xyz 上几乎无效（6.2%，
clean 才 3.9%），在 halfsphere 上却接近上限。任何单序列得出的 loss 排名都不可信。

## 已知问题（按优先级，正在逐个修）

1. **规范失配（gauge mismatch）** — *尺度部分已修复，帧 0 特权待成对结构解决*
   `pose_relative_mse` 比较的是归一化到第 0 帧的**绝对位姿**，但评测用
   `evo_utils.eval_metrics(align=True, correct_scale=True)`，会做 Sim(3) 对齐。
   后果：训练 loss 上升不保证 ATE 上升。攻击第 0 帧最省力，而对齐恰好消除它。
   尺度完全未处理——`trans_scale` 只是常数量纲归一，不做 pred/GT 尺度对齐。

   **已做（尺度）**：新增 `pose_scale_invariant_mse`（`ATTACK_LOSS` 同名新选项，
   旧 `pose_relative_mse` 原样保留）。pred / target 的相对平移各自按自身 RMS 归一
   后再比较，尺度规范从 loss 中移除；旋转项未动。副作用：平移项被 4/3 封顶。
   测试见 `tests/test_pose_scale_invariant_mse.py`。

   **帧 0 特权（已尝试，方案失败）**：所有相对量仍相对第 0 帧计算。若攻击只扰动
   第 0 帧位姿（`T_0' = T_0·δ`），则每个 `rel_i' = δ⁻¹·rel_i`，N−1 帧全部被污染，
   loss 约放大 N 倍；而 ATE 的最佳拟合只把它当成一个离群帧。

   为此实现了 `pose_pairwise_relative_mse`（45 对帧两两比较，无特权帧，单元测试
   验证了帧 0 扰动下它显著低于帧 0 归一化版）。**但实测无效**：sitting_xyz 上只有
   5.9%，低于旧 loss 的 6.2%。排查过程已排除梯度损坏（‖grad‖=0.224，有限差分吻合）、
   步长（lr 扫 50 倍无变化）、loss 不敏感（扰动响应比 si 更高）、正则压制
   （按实测比值把正则缩小 64 倍，ATE 仅 4.2%→5.9%）。**结论：这个目标函数确实
   推不动攻击，不要再在它上面花时间。** 保留作 ablation 对照行。

   **实际有效的方案是 `pose_aligned_residual_mse`**：在 loss 内部复现评测自己的
   Umeyama Sim(3) 对齐再算残差，除以目标轨迹 RMS 半径，平移项 = ATE/ceiling ∈ [0,1]，
   恰在 ATE 取到最大时饱和。对齐参数用 `no_grad` 求解后 detach（包络定理，一阶精确，
   避开 SVD 梯度病态；测试里与完整 autograd 对照相对误差 < 1e-6）。

   注意区分：loss 对**全局** SE(3) 已经免疫（`T_0⁻¹g⁻¹gT_i = T_0⁻¹T_i`），
   但对**只扰动第 0 帧**不免疫。这两件事不同，别混为一谈。

   量化依据：`scripts/diag_gauge_invariance.py`（只读诊断，不改评测语义）。
   实测 18 组合成 Sim(3) 下 ATE 相对变化 1.35e-14；clean 预测相对 GT 的尺度比逐
   序列在 2.07~4.15 之间；76 个已训练 patch 的轨迹破坏中位数 90.4% 被全局 Sim(3)
   吸收（`gauge_absorbed_frac`）。

2. **untargeted loss 无界**
   `objective = -loss + reg`，平移项可无限增大，优化会跑到纹理 clamp 饱和。
   （`pose_scale_invariant_mse` 下平移项已被 4/3 封顶，但旋转项仍无界；
   旧 `pose_relative_mse` 两项都无界。）

3. **EOT 只有光度，无几何 —— 且已实测会制造训练/测试差距**
   `prepare_texture_for_render` 只有 brightness/contrast/gamma/noise，
   对单应 `H_t` 零扰动；且 EOT 因子全序列共享一个标量，未 per-frame 采样。

   **实测危害**（sitting_halfsphere，补丁覆盖率 0.056，是 sitting_xyz 的 4.8 倍）：
   `pose_aligned_residual_mse` 训练时平移项 0.7366，最终干净渲染只有 0.2671，
   **差距 63.7%**；旋转项从 0.772 掉到 0.0012。`PHYSICAL_EOT=0` 重跑后差距精确降到
   **0.0%**，ATE 从 26.7% 升到 44.2%。同一对照下旧 loss 几乎不变（88.9%→89.2%）。

   即：训练用 `training=True`（加抖动）、最终输出用 `training=False`（不加），
   补丁大时攻击会学会**依赖注入的噪声而非几何**，而这部分不转移。危害程度依 loss
   而异，所以它同时污染 loss 之间的比较。跑对照实验时务必固定 `PHYSICAL_EOT`，
   并检查训练末值与最终预测上重算的 loss 是否一致。

4. **mask 硬二值**
   `mask_bool.astype(np.float32)` 是 0/1 硬边，渲染有锯齿，边界梯度不连续。

5. **几何重复计算**
   `train_geometry_patch` 每个 iteration 都调 `load_tum_sequence`，
   但 `OPTIMIZE_GEOMETRY=0` 时 grid/mask 完全确定，属纯浪费。

6. **init 与 reference 未解耦**
   `natural_reference_texture` 在未显式指定时 fallback 到 `texture_init_image`，
   导致「从 X 出发 + 用 MSE 拉回 X」，两个旋钮实际耦合成一个。

7. **裸 ATE 不可比，且 sitting_static 已经饱和**（当前最高优先级）
   Sim(3) 对齐时尺度自由，预测可被压成一点，此时残差 = GT 绕质心的 RMS 半径；
   而 Umeyama 取最小值，所以这是**硬上界**：`ATE ≤ ate_ceiling`。

   sitting_static 的 ceiling 只有 **0.0310 m**。实测 200 次**纯随机**预测的平均 ATE
   是 0.0281（**90.4% of ceiling**），而历史最强攻击是 0.0285（91.7%）——
   **和噪声无法区分**。clean 是 19.5%。这个序列上 ATE 几乎没有区分度，
   过去所有在它上面得出的 loss 排名都是在饱和区比较。

   各序列余量（clean 占 ceiling 比例，越小越好）：walking_halfsphere 2.1%、
   sitting_halfsphere 3.2%、sitting_xyz 3.9%、walking_xyz 5.5%、walking_rpy 8.6%、
   walking_static 18.8%、sitting_static 19.5%、sitting_rpy 20.1%。
   `*_static` / `*_rpy` 都是退化的。

   **要求：一律报告 `ate_frac_of_ceiling`，不要只报裸 ATE。** 管线已自动产出
   （`RUN_GAUGE_DIAG=1` → `tum10-gauge-<model>.csv`），接近无信息水平时会告警。

8. **正则权重与 loss 量级耦合**
   `objective = -attack + reg`。正则梯度是常数（同一初始纹理下实测 1.15e-5），
   攻击梯度随 loss 量级变化，所以 `TV_WEIGHT` / `PRINTABILITY_WEIGHT=0.001`
   这组按旧 loss 调出来的值，换个量级不同的 loss 就等于换了镣铐强度。

   实测 `|g_attack|/|g_reg|`（sitting_xyz）：旧 loss 93.3、aligned 37.0、
   scale_inv 1.86、pairwise 1.40。比值接近 1 时优化器收敛到 `(-attack + reg)` 的
   **真驻点**——这正是 pairwise 换任何 lr 都停在同一位置的原因（lr 只改到达速度，
   不改驻点）。同一比值在 sitting_static 上是 40.7，所以 pairwise 在那儿能训、
   在 xyz 上不能。

   **做 loss ablation 前先量这个比值并按 loss 对齐正则权重**，否则比的是梯度尺度
   不是目标函数。测量方法见本轮诊断（复刻一次训练迭代，分别对 attack 和 reg 求
   `d/d texture`）。

9. **单元测试覆盖不到训练回路**
   `tests/test_pose_*.py` 在 float64、合成轨迹、脱离训练回路下验证 loss 数学，
   10/10 全绿却完全没能预测 pairwise 训不动。**新 loss 除数学测试外，还需要一次
   20 步冒烟并检查 `|g_attack|/|g_reg|` 与训练/测试一致性**，否则"测试全绿但训练
   不动"会重复发生。

## 工作约定

- **改动要小而可验证。** 一次只改一个问题，不要顺手重构无关代码。
- **不改评测语义。** 评测脚本只读不写；诊断需要新指标时新建脚本，不要改现有 eval。
- **新 loss 必须配单元测试。** 尤其涉及 SE(3)/Sim(3) 的，测试要能验证不变性。
  数学测试不够，还要跑 20 步冒烟并检查 `|g_attack|/|g_reg|`，见已知问题 9。
- **下一步不要再加 loss。** 当前瓶颈是评测协议不是目标函数：单序列、裸 ATE、
  EOT 训练/测试差距、正则权重与 loss 量级耦合（问题 3/7/8）。先把协议固定下来
  （多序列 + `ate_frac_of_ceiling` + 固定 EOT + 对齐正则比值），再重新评判 loss。
- **旧 loss 全部保留**，作为论文 ablation 的对照行，不要删除或原地替换。
- 长实验前先用 `ITERATIONS=20` 冒烟，确认能跑通再放长。
