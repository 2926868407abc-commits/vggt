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
| `tests/test_pose_scale_invariant_mse.py` | 尺度不变 pose loss 的单元测试 |

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
- 载体：右侧电脑显示器屏幕，`PLANE_MODE=depth_manual_quad_surface`
- 主 loss：`ATTACK_LOSS=pose_gt_untargeted`
- 可选 loss：`pose_scale_invariant_mse`（尺度不变版，见已知问题 1）/ `feature_l1` /
  `pose_clean_untargeted` / `pose_{reverse,drift,scale,yaw}_targeted`

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

   **未做（帧 0 特权）**：所有相对量仍相对第 0 帧计算。若攻击只扰动第 0 帧位姿
   （`T_0' = T_0·δ`），则每个 `rel_i' = δ⁻¹·rel_i`，N−1 帧全部被污染，loss 约放大
   N 倍；而 ATE 的最佳拟合只把它当成一个离群帧。**这条失配仍然存在**，是下一步
   「成对相对位姿」（45 对帧两两比较、不再有特权帧）要解决的。

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

3. **EOT 只有光度，无几何**
   `prepare_texture_for_render` 只有 brightness/contrast/gamma/noise，
   对单应 `H_t` 零扰动；且 EOT 因子全序列共享一个标量，未 per-frame 采样。

4. **mask 硬二值**
   `mask_bool.astype(np.float32)` 是 0/1 硬边，渲染有锯齿，边界梯度不连续。

5. **几何重复计算**
   `train_geometry_patch` 每个 iteration 都调 `load_tum_sequence`，
   但 `OPTIMIZE_GEOMETRY=0` 时 grid/mask 完全确定，属纯浪费。

6. **init 与 reference 未解耦**
   `natural_reference_texture` 在未显式指定时 fallback 到 `texture_init_image`，
   导致「从 X 出发 + 用 MSE 拉回 X」，两个旋钮实际耦合成一个。

## 工作约定

- **改动要小而可验证。** 一次只改一个问题，不要顺手重构无关代码。
- **不改评测语义。** 评测脚本只读不写；诊断需要新指标时新建脚本，不要改现有 eval。
- **新 loss 必须配单元测试。** 尤其涉及 SE(3)/Sim(3) 的，测试要能验证不变性。
- **旧 loss 全部保留**，作为论文 ablation 的对照行，不要删除或原地替换。
- 长实验前先用 `ITERATIONS=20` 冒烟，确认能跑通再放长。
