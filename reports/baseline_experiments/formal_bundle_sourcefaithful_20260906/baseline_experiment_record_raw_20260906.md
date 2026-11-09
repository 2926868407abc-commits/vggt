# 论文 Baseline 迁移到 VGGT：逐实验记录

这份文档只记录已经实际完成的实验。每个实验单独一节，训练结束后立即写入设置、统一四任务指标和三张图的服务器绝对路径。

## 固定口径

- 主表按各论文自己的摆放、尺寸、初始化、损失和优化器做 VGGT 迁移，不强行固定到同一个显示器位置。
- 载体平面由 TUM RGB-D 与 GT 位姿在训练前提取并冻结；平面选择不读取任何攻击分数。
- 所有方法最终都用同一套 VGGT 四任务评测：Camera pose、Depth、Point map、Track。
- `baseline_aparate_shape` 是老师建议的可学习形状扩展，不冒充原 APARATE。
- `baseline_3d2fool_depth` 写作“3D²Fool 可见表面适配”；TUM 没有完整车辆网格，不能写成完整 3-D 复现。
- 正式主表统一比较 1000 次参数更新；论文原始训练轮数另记在方法说明里。
- 每节固定列出同协议 Clean 与 Attack，并保存三张图：初始 patch、优化后 patch、优化后 patch 的 10 帧无红边合图。

## 尚未写入结果的正式配置

1. `spoo_monitor`、`spoo_table`、`spoo_partition`
2. `aparate_monitor`、`aparate_table`
3. `3d2fool_monitor_visible_surface`
4. `passive_vo_partition`

下面的内容由 `scripts/record_baseline_experiment.py` 在每个实验完成后自动追加。

<!-- BASELINE-RUN:baseline_smoke_aparate_monitor_s0_u2:START -->
## baseline_smoke_aparate_monitor_s0_u2

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`aparate_monitor` / `baseline_aparate_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；APARATE centre placement with linear object scale 0.2.
- 论文一致性边界：Paper loss/centre/scale; frozen RGB-D carrier mask replaces YOLO object detection.
- 外接物理尺寸：0.144 × 0.116 m；外接面积 0.0167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`Adam`，学习率 0.01，总更新数 2
- EOT：开；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.00220
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_aparate_monitor_s0_u2.log`
- APARATE 正则：patch 外权重 1.0，TV 0.1，NPS 1.5

### 统一四任务结果

| ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0079 | 0.4359 | 0.0592 | 0.0233 | 4.51 | 5.70 | 0.701 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_aparate_monitor_s0_u2/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_aparate_monitor_s0_u2/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_aparate_monitor_s0_u2/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_aparate_monitor_s0_u2:END -->

<!-- BASELINE-RUN:baseline_smoke_spoo_monitor_s0_u2:START -->
## baseline_smoke_spoo_monitor_s0_u2

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`spoo_monitor` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 64, 64]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 2
- EOT：开；采样数 1
- EOT 参数：亮度 0.15，对比度 0.15，噪声 0.01，平移 0.02，尺度 0.03，旋转 10.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_spoo_monitor_s0_u2.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.0, 1.0, 0.019661646336317062, 1.0]`
- SPOO 区域是否已到 1/9 并冻结：`False`；末步面积比例 0.98034

### 统一四任务结果

| ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0084 | 0.4592 | 0.0587 | 0.0249 | 4.26 | 5.43 | 0.717 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_spoo_monitor_s0_u2/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_spoo_monitor_s0_u2/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_spoo_monitor_s0_u2/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_spoo_monitor_s0_u2:END -->

<!-- BASELINE-RUN:baseline_smoke_3d2fool_monitor_visible_surface_s0_u2:START -->
## baseline_smoke_3d2fool_monitor_visible_surface_s0_u2

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`3d2fool_monitor_visible_surface` / `baseline_3d2fool_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；Texture the complete visible monitor carrier.
- 论文一致性边界：Visible-surface planar adaptation; TUM has no complete 3-D vehicle mesh.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 128, 128]`
- 优化：`Adam`，学习率 0.01，总更新数 2
- EOT：开；采样数 1
- EOT 参数：亮度 0.2，对比度 0.1，噪声 0.01，平移 0.0，尺度 0.0，旋转 0.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_3d2fool_monitor_visible_surface_s0_u2.log`
- 3D²Fool 适配：128 种子重复 6 次并裁成 512；TV 0.1，NPS 5.0，天气强度 0.15

### 统一四任务结果

| ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0078 | 0.4459 | 0.0581 | 0.0263 | 4.41 | 5.72 | 0.690 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_3d2fool_monitor_visible_surface_s0_u2/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_3d2fool_monitor_visible_surface_s0_u2/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_3d2fool_monitor_visible_surface_s0_u2/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_3d2fool_monitor_visible_surface_s0_u2:END -->

<!-- BASELINE-RUN:baseline_smoke_passive_vo_partition_s0_u2:START -->
## baseline_smoke_passive_vo_partition_s0_u2

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`passive_vo_partition` / `baseline_vo_mprms`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：partition；Fixed planar screen/partition selected before optimization.
- 论文一致性边界：Paper MPRMS and fixed-screen placement; linear scale is frozen before training.
- 外接物理尺寸：1.328 × 0.977 m；外接面积 1.2974 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`PGD-sign`，学习率 0.00392156862745098，总更新数 2
- EOT：关；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05273
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_passive_vo_partition_s0_u2.log`
- Passive VO 更新：PGD-sign；固定屏幕位置；不启用 EOT。

### 统一四任务结果

| ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0090 | 0.4254 | 0.0616 | 0.0147 | 4.17 | 5.32 | 0.682 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_passive_vo_partition_s0_u2/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_passive_vo_partition_s0_u2/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_passive_vo_partition_s0_u2/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_passive_vo_partition_s0_u2:END -->

<!-- BASELINE-RUN:baseline_smoke_aparate_monitor_s0_u1:START -->
## baseline_smoke_aparate_monitor_s0_u1

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`aparate_monitor` / `baseline_aparate_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；APARATE centre placement with linear object scale 0.2.
- 论文一致性边界：Paper loss/centre/scale; frozen RGB-D carrier mask replaces YOLO object detection.
- 外接物理尺寸：0.144 × 0.116 m；外接面积 0.0167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`Adam`，学习率 0.01，总更新数 1
- EOT：开；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.00220
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_aparate_monitor_s0_u1.log`
- APARATE 正则：patch 外权重 1.0，TV 0.1，NPS 1.5

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0078 | 0.4432 | 0.0592 | 0.0236 | 4.51 | 5.70 | 0.707 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_aparate_monitor_s0_u1/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_aparate_monitor_s0_u1/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_aparate_monitor_s0_u1/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_aparate_monitor_s0_u1:END -->

<!-- BASELINE-RUN:baseline_formal_passive_vo_partition_s0_u1000:START -->
## baseline_formal_passive_vo_partition_s0_u1000

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
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_passive_vo_partition_s0_u1000.log`
- Passive VO 更新：PGD-sign；固定屏幕位置；不启用 EOT。

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0107 | 0.4609 | 0.6309 | 0.1029 | 4.76 | 7.51 | 0.432 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_passive_vo_partition_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_passive_vo_partition_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_passive_vo_partition_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_passive_vo_partition_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_3d2fool_monitor_visible_surface_s0_u1000:START -->
## baseline_formal_3d2fool_monitor_visible_surface_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`3d2fool_monitor_visible_surface` / `baseline_3d2fool_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；Texture the complete visible monitor carrier.
- 论文一致性边界：Visible-surface planar adaptation; TUM has no complete 3-D vehicle mesh.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 128, 128]`
- 优化：`Adam`，学习率 0.01，总更新数 1000
- EOT：开；采样数 1
- EOT 参数：亮度 0.2，对比度 0.1，噪声 0.01，平移 0.0，尺度 0.0，旋转 0.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_3d2fool_monitor_visible_surface_s0_u1000.log`
- 3D²Fool 适配：128 种子重复 6 次并裁成 512；TV 0.1，NPS 5.0，天气强度 0.15

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0077 | 0.4443 | 0.0581 | 0.0253 | 4.37 | 5.59 | 0.701 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_3d2fool_monitor_visible_surface_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_3d2fool_monitor_visible_surface_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_3d2fool_monitor_visible_surface_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_3d2fool_monitor_visible_surface_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_aparate_monitor_s0_u1000:START -->
## baseline_formal_aparate_monitor_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`aparate_monitor` / `baseline_aparate_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；APARATE centre placement with linear object scale 0.2.
- 论文一致性边界：Paper loss/centre/scale; frozen RGB-D carrier mask replaces YOLO object detection.
- 外接物理尺寸：0.144 × 0.116 m；外接面积 0.0167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`Adam`，学习率 0.01，总更新数 1000
- EOT：开；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.00220
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_aparate_monitor_s0_u1000.log`
- APARATE 正则：patch 外权重 1.0，TV 0.1，NPS 1.5

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0077 | 0.4353 | 0.0602 | 0.0242 | 4.52 | 5.72 | 0.705 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_aparate_monitor_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_aparate_monitor_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_aparate_monitor_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_aparate_monitor_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_aparate_table_s0_u1000:START -->
## baseline_formal_aparate_table_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`aparate_table` / `baseline_aparate_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：table；APARATE centre placement with linear object scale 0.2.
- 论文一致性边界：Paper loss/centre/scale; frozen RGB-D carrier mask replaces YOLO object detection.
- 外接物理尺寸：0.382 × 0.187 m；外接面积 0.0715 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`Adam`，学习率 0.01，总更新数 1000
- EOT：开；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.00209
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_aparate_table_s0_u1000.log`
- APARATE 正则：patch 外权重 1.0，TV 0.1，NPS 1.5

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0084 | 0.4390 | 0.0588 | 0.0229 | 4.53 | 5.73 | 0.686 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_aparate_table_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_aparate_table_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_aparate_table_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_aparate_table_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_spoo_table_s0_u1000:START -->
## baseline_formal_spoo_table_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`spoo_table` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：table；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：1.911 × 0.935 m；外接面积 1.7866 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 64, 64]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 1000
- EOT：开；采样数 1
- EOT 参数：亮度 0.15，对比度 0.15，噪声 0.01，平移 0.02，尺度 0.03，旋转 10.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.09074
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_spoo_table_s0_u1000.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.02483983524143696, 1.0, 0.11493419855833054, 1.0]`
- SPOO 区域是否已到 1/9 并冻结：`False`；末步面积比例 0.86308

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0090 | 0.4359 | 0.0552 | 0.0235 | 4.66 | 5.86 | 0.630 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_spoo_table_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_spoo_table_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_spoo_table_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_spoo_table_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_spoo_partition_s0_u1000:START -->
## baseline_formal_spoo_partition_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`spoo_partition` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：partition；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：2.015 × 1.482 m；外接面积 2.9877 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 64, 64]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 1000
- EOT：开；采样数 1
- EOT 参数：亮度 0.15，对比度 0.15，噪声 0.01，平移 0.02，尺度 0.03，旋转 10.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.09606
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_spoo_partition_s0_u1000.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.003559405915439129, 0.2249622344970703, 0.0, 1.0]`
- SPOO 区域是否已到 1/9 并冻结：`False`；末步面积比例 0.22140

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0083 | 0.4267 | 0.0605 | 0.0201 | 4.53 | 5.73 | 0.668 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_spoo_partition_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_spoo_partition_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_spoo_partition_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_spoo_partition_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_formal_spoo_monitor_s0_u1000:START -->
## baseline_formal_spoo_monitor_s0_u1000

- 阶段：正式 1000 更新
- 方法配置：`spoo_monitor` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 64, 64]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 1000
- EOT：开；采样数 1
- EOT 参数：亮度 0.15，对比度 0.15，噪声 0.01，平移 0.02，尺度 0.03，旋转 10.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_formal_spoo_monitor_s0_u1000.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.01604676991701126, 0.9855542778968811, 0.0, 1.0]`
- SPOO 区域是否已到 1/9 并冻结：`False`；末步面积比例 0.96951

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0081 | 0.4352 | 0.0590 | 0.0308 | 4.48 | 5.91 | 0.721 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_spoo_monitor_s0_u1000/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_formal_spoo_monitor_s0_u1000/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_formal_spoo_monitor_s0_u1000/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_formal_spoo_monitor_s0_u1000:END -->

<!-- BASELINE-RUN:baseline_smoke_aparate_monitor_s0_u3:START -->
## baseline_smoke_aparate_monitor_s0_u3

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`aparate_monitor` / `baseline_aparate_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；APARATE centre placement with linear object scale 0.2.
- 论文一致性边界：Paper loss/centre/scale; frozen RGB-D carrier mask replaces YOLO object detection.
- 外接物理尺寸：0.144 × 0.116 m；外接面积 0.0167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`Adam`，学习率 0.01，总更新数 3
- EOT：开；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.0，尺度 0.03，旋转 20.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.00220
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_aparate_monitor_s0_u3.log`
- APARATE 正则：patch 外权重 1.0，TV 0.1，NPS 1.5

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0081 | 0.4422 | 0.0593 | 0.0235 | 4.49 | 5.70 | 0.713 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_aparate_monitor_s0_u3/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_aparate_monitor_s0_u3/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_aparate_monitor_s0_u3/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_aparate_monitor_s0_u3:END -->

<!-- BASELINE-RUN:baseline_smoke_3d2fool_monitor_visible_surface_s0_u3:START -->
## baseline_smoke_3d2fool_monitor_visible_surface_s0_u3

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`3d2fool_monitor_visible_surface` / `baseline_3d2fool_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；Texture the complete visible monitor carrier.
- 论文一致性边界：Visible-surface planar adaptation; TUM has no complete 3-D vehicle mesh.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 128, 128]`
- 优化：`Adam`，学习率 0.01，总更新数 3
- EOT：开；采样数 1
- EOT 参数：亮度 0.2，对比度 0.1，噪声 0.01，平移 0.0，尺度 0.0，旋转 0.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_3d2fool_monitor_visible_surface_s0_u3.log`
- 3D²Fool 适配：128 种子重复 6 次并裁成 512；TV 0.1，NPS 5.0，天气强度 0.0

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0083 | 0.4504 | 0.0576 | 0.0259 | 4.42 | 5.77 | 0.698 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_3d2fool_monitor_visible_surface_s0_u3/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_3d2fool_monitor_visible_surface_s0_u3/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_3d2fool_monitor_visible_surface_s0_u3/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_3d2fool_monitor_visible_surface_s0_u3:END -->

<!-- BASELINE-RUN:baseline_smoke_passive_vo_partition_s0_u3:START -->
## baseline_smoke_passive_vo_partition_s0_u3

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`passive_vo_partition` / `baseline_vo_mprms`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：partition；Fixed planar screen/partition selected before optimization.
- 论文一致性边界：Paper MPRMS and fixed-screen placement; linear scale is frozen before training.
- 外接物理尺寸：1.328 × 0.977 m；外接面积 1.2974 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`PGD-sign`，学习率 0.00392156862745098，总更新数 3
- EOT：关；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05273
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_passive_vo_partition_s0_u3.log`
- Passive VO 更新：PGD-sign；固定屏幕位置；不启用 EOT。

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0085 | 0.4320 | 0.0605 | 0.0138 | 4.31 | 5.44 | 0.719 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_passive_vo_partition_s0_u3/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_passive_vo_partition_s0_u3/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_passive_vo_partition_s0_u3/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_passive_vo_partition_s0_u3:END -->

<!-- BASELINE-RUN:baseline_smoke_spoo_monitor_s17_u3:START -->
## baseline_smoke_spoo_monitor_s17_u3

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`spoo_monitor` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 48, 64]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 3
- EOT：关；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_spoo_monitor_s17_u3.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.0, 1.0, 0.9994999766349792, 1.0]`
- SPOO 区域是否已到 1/9 并冻结：`True`；末步面积比例 0.00050

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0082 | 0.4392 | 0.0594 | 0.0227 | 4.50 | 5.67 | 0.693 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_spoo_monitor_s17_u3/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_spoo_monitor_s17_u3/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_spoo_monitor_s17_u3/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_spoo_monitor_s17_u3:END -->

<!-- BASELINE-RUN:baseline_smoke_aparate_monitor_s0_u4:START -->
## baseline_smoke_aparate_monitor_s0_u4

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`aparate_monitor` / `baseline_aparate_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；APARATE centre placement with linear object scale 0.2.
- 论文一致性边界：Paper loss/centre/scale; frozen RGB-D carrier mask replaces YOLO object detection.
- 外接物理尺寸：0.144 × 0.116 m；外接面积 0.0167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 64, 64]`
- 优化：`Adam`，学习率 0.01，总更新数 4
- EOT：开；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.0，尺度 0.03，旋转 20.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.00220
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_aparate_monitor_s0_u4.log`
- APARATE 正则：patch 外权重 1.0，TV 0.1，NPS 1.5

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0080 | 0.4405 | 0.0593 | 0.0234 | 4.50 | 5.70 | 0.709 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_aparate_monitor_s0_u4/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_aparate_monitor_s0_u4/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_aparate_monitor_s0_u4/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_aparate_monitor_s0_u4:END -->

<!-- BASELINE-RUN:baseline_smoke_3d2fool_monitor_visible_surface_s0_u4:START -->
## baseline_smoke_3d2fool_monitor_visible_surface_s0_u4

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`3d2fool_monitor_visible_surface` / `baseline_3d2fool_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；Texture the complete visible monitor carrier.
- 论文一致性边界：Visible-surface planar adaptation; TUM has no complete 3-D vehicle mesh.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 128, 128]`
- 优化：`Adam`，学习率 0.01，总更新数 4
- EOT：开；采样数 1
- EOT 参数：亮度 0.2，对比度 0.1，噪声 0.0，平移 0.0，尺度 0.0，旋转 0.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_3d2fool_monitor_visible_surface_s0_u4.log`
- 3D²Fool 适配：128 种子重复 6 次并裁成 512；TV 0.1，NPS 5.0，天气强度 0.0

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0084 | 0.4401 | 0.0577 | 0.0260 | 4.41 | 5.75 | 0.701 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_3d2fool_monitor_visible_surface_s0_u4/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_3d2fool_monitor_visible_surface_s0_u4/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_3d2fool_monitor_visible_surface_s0_u4/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_3d2fool_monitor_visible_surface_s0_u4:END -->

<!-- BASELINE-RUN:baseline_smoke_3d2fool_monitor_visible_surface_s0_u1:START -->
## baseline_smoke_3d2fool_monitor_visible_surface_s0_u1

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`3d2fool_monitor_visible_surface` / `baseline_3d2fool_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；Texture the complete visible monitor carrier.
- 论文一致性边界：Visible-surface planar adaptation; TUM has no complete 3-D vehicle mesh.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `random`，分辨率 `[1, 3, 128, 128]`
- 优化：`Adam`，学习率 0.01，总更新数 1
- EOT：开；采样数 1
- EOT 参数：亮度 0.2，对比度 0.1，噪声 0.0，平移 0.0，尺度 0.0，旋转 0.0°，透视 0.0
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_3d2fool_monitor_visible_surface_s0_u1.log`
- 3D²Fool 适配：128 种子重复 6 次并裁成 4096；TV 0.1，NPS 5.0，天气强度 0.0

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0079 | 0.4421 | 0.0578 | 0.0260 | 4.40 | 5.60 | 0.685 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_3d2fool_monitor_visible_surface_s0_u1/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_3d2fool_monitor_visible_surface_s0_u1/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_3d2fool_monitor_visible_surface_s0_u1/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_3d2fool_monitor_visible_surface_s0_u1:END -->

<!-- BASELINE-RUN:baseline_smoke_spoo_monitor_s17_u1:START -->
## baseline_smoke_spoo_monitor_s17_u1

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`spoo_monitor` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 450, 600]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 1
- EOT：关；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_spoo_monitor_s17_u1.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.0, 1.0, 0.0, 1.0]`
- SPOO 区域是否已到 1/9 并冻结：`False`；末步面积比例 1.00000

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0085 | 0.4433 | 0.0574 | 0.0247 | 4.36 | 5.53 | 0.725 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_spoo_monitor_s17_u1/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_spoo_monitor_s17_u1/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_spoo_monitor_s17_u1/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_spoo_monitor_s17_u1:END -->

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

<!-- BASELINE-RUN:baseline_smoke_controllerfix2_spoo_monitor_s17_u120:START -->
## baseline_smoke_controllerfix2_spoo_monitor_s17_u120

- 阶段：短冒烟，只验代码与摆放
- 方法配置：`spoo_monitor` / `baseline_spoo_depth`
- 场景：`rgbd_dataset_freiburg3_sitting_halfsphere`
- 载体与摆放：monitor；SPOO learns [l,r,t,b] inside the full frozen carrier and stops at area ratio 1/9.
- 论文一致性边界：Paper region optimization; TUM carrier plane replaces the paper's segmented object.
- 外接物理尺寸：0.720 × 0.578 m；外接面积 0.4167 m²
- 纹理：初始化 `image`，分辨率 `[1, 3, 450, 600]`
- 优化：`L-BFGS texture + Adam borders (largest-gradient edge only)`，学习率 0.3689，总更新数 120
- EOT：关；采样数 1
- EOT 参数：亮度 0.1，对比度 0.2，噪声 0.1，平移 0.02，尺度 0.03，旋转 20.0°，透视 0.01
- 渲染检查：预检通过 `True`；训练输出平均覆盖率 0.05464
- 日志：`/mnt/data/wangqq/vggt/logs/baseline_smoke_controllerfix2_spoo_monitor_s17_u120.log`
- SPOO 目标面积比例：0.1111111111111111；最终边界 `[l,r,t,b]` = `[0.0, 0.8089011311531067, 0.04391011595726013, 0.7445869445800781]`
- SPOO 区域是否已到 1/9 并冻结：`False`；末步面积比例 0.56678

### 统一四任务结果

| 对象 | ATE | RPE旋转 | Depth AbsRel | Point-map Acc | Track EPE中位 | Track EPE均值 | Track visF1 | 渲染复现 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Clean | 0.0082 | 0.4457 | 0.0595 | 0.0225 | 4.49 | 5.69 | 0.690 | True |
| Attack | 0.0078 | 0.4301 | 0.0609 | 0.0291 | 4.67 | 5.84 | 0.725 | True |

### 本实验三张图（服务器绝对路径）

1. Patch 初始图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_controllerfix2_spoo_monitor_s17_u120/geometry_patch/initial_patch.png`
2. Patch 优化后图：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/baseline_smoke_controllerfix2_spoo_monitor_s17_u120/geometry_patch/optimized_patch.png`
3. 优化后 patch 在 10 帧中的无红边合图：`/mnt/data/wangqq/vggt/reports/baseline_experiments/figures/baseline_smoke_controllerfix2_spoo_monitor_s17_u120/rgbd_dataset_freiburg3_sitting_halfsphere/contact_sheet_patch.png`

> 第 3 张图由生产几何、深度遮挡掩码和同一纹理重渲染；命令固定使用 `--no_outline --no_label --no_outline_frames`。
<!-- BASELINE-RUN:baseline_smoke_controllerfix2_spoo_monitor_s17_u120:END -->
