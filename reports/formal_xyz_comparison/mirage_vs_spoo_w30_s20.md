# sitting_xyz：MIRAGE-3D 与 SPOO 的收敛对比

- 训练集：该序列前半段 608 帧；每步随机抽 10 帧。
- 测试集：后半段 608 帧，完全不参与训练和早停。
- 位姿协议：30 帧窗口、步长 20、相邻窗口重叠 10 帧；先拼成一条完整轨迹，再做一次全局 Sim(3) 对齐。
- 载体：两种方法使用同一个 v5 显示器三维四边形和相同物理面积。
- 停止规则：固定训练审计集停滞后降低学习率；到达最低学习率后继续停滞才 early-stop，并恢复最佳审计纹理。
- 旧的 10 帧窗口／步长 5 协议已删除，不参与本表。

## 收敛证据

| 阶段 | 实际更新数 | 最佳更新 | 最佳审计分数 | 最终学习率 | early-stop |
|---|---:|---:|---:|---:|---|
| MIRAGE pose warm-up | 3700 | 1400 | +0.012468 | 1e-05 | 是 |
| MIRAGE-3D | 6300 | 5800 | +0.093993 | 1e-05 | 是 |
| SPOO stage 1 | 900 | 100 | +0.039465 | 0.18445 | 是 |
| SPOO | 900 | 100 | +0.088593 | 0.18445 | 是 |

## 后半段完整四任务结果

表中“恶化”为相对各方法同一路径 Clean 的变化；正数表示攻击方向变差。

| 指标 | MIRAGE Clean | MIRAGE Attack | MIRAGE 恶化 | SPOO Clean | SPOO Attack | SPOO 恶化 |
|---|---:|---:|---:|---:|---:|---:|
| ATE (m) | 0.074348 | 0.098202 | +32.1% | 0.074348 | 0.063536 | -14.5% |
| RPE trans (m) | 0.007120 | 0.008077 | +13.4% | 0.007120 | 0.007107 | -0.2% |
| RPE rot (deg) | 0.332757 | 0.334611 | +0.6% | 0.332757 | 0.331945 | -0.2% |
| Depth AbsRel | 0.073427 | 0.086499 | +17.8% | 0.073427 | 0.071961 | -2.0% |
| Point Acc | 0.068765 | 0.169246 | +146.1% | 0.068765 | 0.064483 | -6.2% |
| Point Comp | 0.083016 | 0.180173 | +117.0% | 0.083016 | 0.078880 | -5.0% |
| Track EPE (px) | 2.683020 | 2.679527 | -0.1% | 2.683020 | 2.652445 | -1.1% |
| Track vis-F1 | 0.940644 | 0.927610 | +1.4% | 0.940644 | 0.941325 | -0.1% |

## 绝对路径

- MIRAGE-3D 初始纹理：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/formal_xyz_v3_mirage_joint_warm_adaptive_s0_umax6000/geometry_patch/initial_patch.png`
- MIRAGE-3D 优化纹理：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/formal_xyz_v3_mirage_joint_warm_adaptive_s0_umax6000/geometry_patch/optimized_patch.png`
- MIRAGE-3D 30/20 结果：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/formal_xyz_v3_mirage_joint_warm_adaptive_s0_umax6000/heldout_w30_s20_v5_controlled`
- SPOO 初始纹理：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/formal_xyz_v3_spoo_stage2_adaptive_s17_umax6000/geometry_patch/initial_patch.png`
- SPOO 优化纹理：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/formal_xyz_v3_spoo_stage2_adaptive_s17_umax6000/geometry_patch/optimized_patch.png`
- SPOO 30/20 结果：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/formal_xyz_v3_spoo_stage2_adaptive_s17_umax6000/heldout_w30_s20_v5_controlled`
