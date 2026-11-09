# 论文 Baseline 迁移到 VGGT：正式实验记录

这份文档只保留进入论文主表的 7 组正式实验。代码冒烟、控制器修复过程和旧实现结果不参与比较，原始流水记录另存为 `baseline_experiment_record_raw_20260906.md`。

## 固定口径

- 主表按各论文自己的摆放、尺寸、初始化、损失和优化器迁移到 VGGT，不强行固定到同一显示器位置。
- 载体平面由 TUM RGB-D 与 GT 位姿在训练前提取并冻结；平面选择不读取攻击分数。
- 所有方法都在 `rgbd_dataset_freiburg3_sitting_halfsphere` 的同一 10 帧上训练和评测。
- 所有方法统一比较 1000 次参数更新，并用同一套四任务指标评测 Camera pose、Depth、Point map 和 Track。
- `baseline_3d2fool_depth` 是“3D²Fool 可见表面适配”；TUM 没有完整车辆网格，不能写成完整 3-D 复现。
- 每组都保存三张图：初始 patch、优化后 patch、优化后 patch 的 10 帧无红边合图。
- 7 份 CSV 的 Clean/Attack 两行均通过独立重渲染检查，`render_ok=True`。

## 正式结果总表

| 方法与摆放 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE均值 |
|---|---:|---:|---:|---:|---:|
| Clean | 0.0082 | 0.4457° | 0.0595 | 0.0225 | 5.69 |
| SPOO－显示器 | 0.0092 | 0.4409° | 0.1606 | 0.1072 | 6.14 |
| SPOO－桌面 | **0.0127** | **0.4969°** | 0.2045 | **0.1096** | **6.36** |
| SPOO－隔板 | 0.0100 | 0.4659° | **0.5656** | 0.0650 | 5.85 |
| APARATE－显示器 | 0.0076 | 0.4374° | 0.0593 | 0.0235 | 5.69 |
| APARATE－桌面 | 0.0080 | 0.4362° | 0.0590 | 0.0225 | 5.70 |
| 3D²Fool－显示器可见表面 | 0.0083 | 0.4321° | 0.0607 | 0.0371 | 5.78 |
| Passive VO－隔板 | 0.0081 | 0.4361° | 0.0939 | 0.0733 | 5.82 |

下面逐项记录完整设置、指标和三张图片的服务器绝对路径。

<!-- BASELINE-RUN:baseline_formal_sourcefaithful_spoo_monitor_s17_u1000:START -->
## baseline_formal_sourcefaithful_spoo_monitor_s17_u1000

- 阶段：正式 1000 更新
- 方法配置：`spoo_monitor` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 450, 600]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 1000
- EOT：关；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_sourcefaithful_spoo_monitor_s17_u1000.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.2680138647556305, 0.7132334113121033, 0.2910550534725189, 0.5243635773658752]`
- SPOO 区域是否已到 1/9 并冻结：`True`；末步面积比例 0.10387

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0092 | 0.4409 | 0.1606 | 0.1072 | 4.57 | 6.14 | 0.710 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_spoo_monitor_s17_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_spoo_monitor_s17_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_sourcefaithful_spoo_monitor_s17_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_sourcefaithful_spoo_monitor_s17_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_sourcefaithful_spoo_table_s17_u1000:START -->
## baseline_formal_sourcefaithful_spoo_table_s17_u1000

- 阶段：正式 1000 更新
- 方法配置：`spoo_table` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：table；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：1.911 × 0.935 m；外接面积 1.7866 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 450, 600]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 1000
- EOT：关；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.09074
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_sourcefaithful_spoo_table_s17_u1000.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.4274442195892334, 0.8074513077735901, 0.1432971954345703, 0.4353444576263428]`
- SPOO 区域是否已到 1/9 并冻结：`True`；末步面积比例 0.11098

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0127 | 0.4969 | 0.2045 | 0.1096 | 4.83 | 6.36 | 0.328 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_spoo_table_s17_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_spoo_table_s17_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_sourcefaithful_spoo_table_s17_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_sourcefaithful_spoo_table_s17_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_sourcefaithful_spoo_partition_s17_u1000:START -->
## baseline_formal_sourcefaithful_spoo_partition_s17_u1000

- 阶段：正式 1000 更新
- 方法配置：`spoo_partition` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：partition；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：2.015 × 1.482 m；外接面积 2.9877 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 450, 600]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 1000
- EOT：关；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.09606
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_sourcefaithful_spoo_partition_s17_u1000.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.11300665140151978, 0.4819926917552948, 0.5276813507080078, 0.7999945878982544]`
- SPOO 区域是否已到 1/9 并冻结：`True`；末步面积比例 0.10048

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0100 | 0.4659 | 0.5656 | 0.0650 | 4.49 | 5.85 | 0.409 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_spoo_partition_s17_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_spoo_partition_s17_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_sourcefaithful_spoo_partition_s17_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_sourcefaithful_spoo_partition_s17_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_sourcefaithful_aparate_monitor_s0_u1000:START -->
## baseline_formal_sourcefaithful_aparate_monitor_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`aparate_monitor` / `baseline_aparate_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；APARATE centre placement with linear object scale 0.2.
- 论文一致性边界：Paper loss/centre/scale; frozen RGB-D carrier mask replaces YOLO object detection.
- 外接物理尺寸：0.144 × 0.116 m；外接面积 0.0167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`Adam`，学习率 0.01，总更新数 1000
- EOT：开；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.0，尺度 0.03，旋转 20.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.00220
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_sourcefaithful_aparate_monitor_s0_u1000.log`
- APARATE 正则：patch 外权重 1.0，TV 0.1，NPS 1.5

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0076 | 0.4374 | 0.0593 | 0.0235 | 4.47 | 5.69 | 0.714 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_aparate_monitor_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_aparate_monitor_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_sourcefaithful_aparate_monitor_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_sourcefaithful_aparate_monitor_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_sourcefaithful_aparate_table_s0_u1000:START -->
## baseline_formal_sourcefaithful_aparate_table_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`aparate_table` / `baseline_aparate_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：table；APARATE centre placement with linear object scale 0.2.
- 论文一致性边界：Paper loss/centre/scale; frozen RGB-D carrier mask replaces YOLO object detection.
- 外接物理尺寸：0.382 × 0.187 m；外接面积 0.0715 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`Adam`，学习率 0.01，总更新数 1000
- EOT：开；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.0，尺度 0.03，旋转 20.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.00209
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_sourcefaithful_aparate_table_s0_u1000.log`
- APARATE 正则：patch 外权重 1.0，TV 0.1，NPS 1.5

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0080 | 0.4362 | 0.0590 | 0.0225 | 4.53 | 5.70 | 0.704 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_aparate_table_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_aparate_table_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_sourcefaithful_aparate_table_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_sourcefaithful_aparate_table_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_sourcefaithful_3d2fool_monitor_visible_surface_s0_u1000:START -->
## baseline_formal_sourcefaithful_3d2fool_monitor_visible_surface_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`3d2fool_monitor_visible_surface` / `baseline_3d2fool_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；Texture the complete visible monitor carrier.
- 论文一致性边界：Visible-surface planar adaptation; TUM has no complete 3-D vehicle mesh.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 128, 128]`
- 优化：`Adam`，学习率 0.01，总更新数 1000
- EOT：开；采样数 1
- EOT 参数：亮度 0.2，对比度 0.1，噪声 0.0，平移 0.0，尺度 0.0，旋转 0.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_sourcefaithful_3d2fool_monitor_visible_surface_s0_u1000.log`
- 3D²Fool 适配：128 种子重复 6 次并裁成 4096；TV 0.1，NPS 5.0，天气强度 0.0

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0083 | 0.4321 | 0.0607 | 0.0371 | 4.51 | 5.78 | 0.527 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_3d2fool_monitor_visible_surface_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_3d2fool_monitor_visible_surface_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_sourcefaithful_3d2fool_monitor_visible_surface_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_sourcefaithful_3d2fool_monitor_visible_surface_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_sourcefaithful_passive_vo_partition_s0_u1000:START -->
## baseline_formal_sourcefaithful_passive_vo_partition_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`passive_vo_partition` / `baseline_vo_mprms`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：partition；Fixed planar screen/partition selected before optimization.
- 论文一致性边界：Paper MPRMS and fixed-screen placement; linear scale is frozen before training.
- 外接物理尺寸：1.328 × 0.977 m；外接面积 1.2974 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`PGD-sign`，学习率 0.00392156862745098，总更新数 1000
- EOT：关；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05273
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_sourcefaithful_passive_vo_partition_s0_u1000.log`
- Passive VO 更新：PGD-sign；固定屏幕位置；不启用 EOT。

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0081 | 0.4361 | 0.0939 | 0.0733 | 4.43 | 5.82 | 0.743 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_passive_vo_partition_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_sourcefaithful_passive_vo_partition_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_sourcefaithful_passive_vo_partition_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_sourcefaithful_passive_vo_partition_s0_u1000:END -->

## 正式实验结论

1. **SPOO 是这组 baseline 中攻击最明显的方法。** 桌面配置把 ATE 从 0.0082 提高到 0.0127、Point-map Acc 从 0.0225 提高到 0.1096；隔板配置把 Depth AbsRel 提高到 0.5656。
2. **原论文方法迁移到 VGGT 后，并不会自动同时破坏四个任务。** SPOO 对深度和点图明显有效，但位姿和跟踪只出现较小变化；APARATE、3D²Fool 和 Passive VO 的多数指标仍接近 Clean。
3. **这张表是“按论文自身规则迁移”的主表，不是等面积、等位置的纯算法消融。** 不同方法的载体、尺寸和初始化属于各自方法的一部分，因此只能回答“整套方法迁移到 VGGT 后效果如何”。
4. 三组 SPOO 的最终区域比例分别为 10.39%、11.10% 和 10.05%，均达到论文约 1/9 的目标并冻结；旧版没有正确缩小区域的 SPOO 结果全部作废。

最终结果包：`/mnt/data/wangqq/vggt/reports/baseline_experiments/formal_bundle_sourcefaithful_20260906`
