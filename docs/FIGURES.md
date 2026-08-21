# 图表说明（对应 outputs/figures/）

本文件是 `outputs/figures/README.md` 的版本化副本。
图片本身不入库（可由 scripts/ 下对应脚本一条命令重生成）。

三档：`current/` 是当前结论用图，`method/` 是评测方法学用图（结论与分辨率无关），
`archive_128era/` 是已被取代的旧图，只作历史记录，**不要放进汇报**。

所有正式结果均在 TUM fr3 sitting_halfsphere，显示器手工四边形放置，
贴片 0.639 x 0.411 m，10 帧中 6 帧可见。干净基线 ATE 0.0082 m / RPE 旋转 0.45 度，
ATE 上限 0.2519 m（随机轨迹达 90.5%）。

## current/ —— 当前结论

| 文件 | 讲什么 | 关键数字 |
|---|---|---|
| `mon64_1_textures.png` | 上排贴片实际样子，下排减掉初始化后的对抗扰动 | 扰动去相关长度 3.8 / 8.2 / 13.7 mm，对应容差 0.64 / 1.6 / 3.2 mm |
| `mon64_2_trajectory.png` | 相机轨迹，已按评测口径 Sim(3) 对齐 | 干净逐帧误差 0.008 m，攻击后 0.19-0.21 m |
| `mon64_3_displacement.png` | 同一补丁在 0 / 1.6 / 6.4 mm 位移下的渲染与像素差 | 1.6 mm 只改约 3% 像素，ATE 掉到 1/6 |
| `mon64_4_pointcloud.png` | 预测点云俯视，同一 Sim(3) 映射到 GT 坐标系 | 攻击后 128 组 0% 的点落在干净重建范围内 |
| `mon64_5_tolerance.png` | 容差曲线：ATE 随贴装位移的衰减，三种分辨率 | 128 在 0.64 mm 掉过半，64 在 1.6 mm 仍保 83% |
| `mon64_6_loss_comparison.png` | 四损失 × 三种子，两个指标分面 | ATE 最强 pairwise 86.6%，旋转最强 scale_invariant 9.84° |

生成脚本：`scripts/fig_texture_and_traj.py`、`scripts/fig_displace_and_cloud.py`、
`scripts/fig_tolerance_curve.py`、`scripts/fig_loss_comparison.py`
数据表：`scripts/loss64_table.py`（四损失 × 三种子，含种子标准差与可排名判定）

容差曲线另有一版可交互网页（含数据表），数字与 PNG 版一致。

**读图必须带的四条注记：**
1. 轨迹图和点云图都**已做 Sim(3) 对齐**。gauge absorbed 约 0.86，不对齐的话画面上大部分
   "破坏"是评测会直接消掉的整体变换。
2. 位移毫米数是**每轴最大值**（抖动在 ±该值内均匀采样），不是典型值。
3. **两个指标排名不同，必须同时报。** ATE 上 pairwise > aligned > scale_invariant > 旧损失；
   RPE 旋转上 scale_invariant 遥遥领先（9.84°，其余 1.8-3.3°）。前者衡量轨迹被压塌的程度，
   后者衡量相机朝向被拧歪的程度，是两种不同的破坏。四者 ATE 均在随机基线 90.5% 以下，
   都落在可分辨区间。
4. 种子噪声实测标准差约 1.9%，而四个损失均值跨度 28%，所以这个排名是可信的
   （128×128 那一轮没有重复种子，排名不具备这个依据）。

## method/ —— 评测方法学，结论不随分辨率变化

| 文件 | 讲什么 | 注意 |
|---|---|---|
| `method_ate_ceiling.png` | 原始 ATE 不可解释：随机预测已达上限约 90% | 图中"best attack achieved"一组是 128 期数据，需更新 |
| `method_gauge_absorbed.png` | 一次全局 Sim(3) 吸收掉多少破坏 | 现象仍成立（当前跑实测 0.857），散点是旧跑 |
| `method_loss_saturation.png` | scale-invariant 项在约 250 步自我饱和，旧项无界增长 | 损失函数的数学性质，与分辨率、放置无关 |
| `method_alignment_effect.png` | 对齐前后对比：92% 的破坏被评测消掉 | 讲清 gauge 问题最直观的一张 |

## archive_128era/ —— 已被取代

这些跑在 128x128 纹理 + 自动放置上。该配置需要亚毫米对齐（ATE 容差 0.64 mm），
物理不可实现，因此**其上的横向排名不能作为结论**。

| 文件 | 为什么作废 |
|---|---|
| `fig2_loss_ranking.png` | 128x128 上的四损失排名；正被 64x64 三种子重跑替换 |
| `fig4_targeted_progression.png` | Q3/Q4 定向路线的历史记录 |
| `fig7_target_vs_achieved.png` | 同上 |
| `fig8_halfsphere_targeting.png` | 同上 |
| `fig9_calibration_textures.png` | 标定教训仍成立，但纹理是 128 期 |
| `_contact_sheet.png` | 上述五张的缩略图总览 |
