# 自适应收敛诊断：sitting_static / sitting_xyz

日期：2026-09-17

## 要回答的问题

旧实验统一训练 1000 步，并使用余弦学习率；第 1000 步时学习率已经接近 0。因此，“后期指标不再明显变化”只能说明预设调度结束，不能证明纹理已经真正收敛。

本实验从同一张 warning 初始图重新训练，用固定训练审计集决定何时降低学习率、何时停止。训练期间不查看后半段未见测试帧，避免用测试集挑 checkpoint。

## 自适应规则

- 初始学习率：0.002；前 20 次更新线性 warm-up。
- 每 100 次更新，在四个固定训练审计子集上评估一次。
- 审计同时计算五个目标：`pose_trans`、`pose_rot`、`depth`、`point`、`track`。
- 每个目标先计算 `attack / clean`，再取五项对数倍率的最小值作为审计分数；因此只有最弱的一项也改善，整体分数才会上升。
- 审计分数相对上一次显著改善不足 0.005，连续 3 次后学习率减半。
- 学习率最低为 1e-5；到达最低学习率后再连续 5 次无显著改善，提前停止。
- 最大训练上限为 3000 步。
- 始终保存审计分数最高的纹理；训练结束时恢复该纹理作为正式输出。

## 不变的实验设置

- 数据划分：每个序列前半帧训练，后半帧只在训练全部结束后测试一次。
- 每一步：从该序列前半帧池随机抽连续语义下的 10 帧组合。
- Patch：64×64 RGB，warning 图初始化；使用 v5 显示器世界坐标四边形。
- 损失：`mirage_projected_split_pose`，五目标 max-min 联合梯度。
- 正则项：TV 0.001、printability 0.001、natural-reference 0.05。
- EOT：本轮关闭；本实验只判断原 1000 步是否训练充分。
- 随机种子：0。

## 固定训练审计集

每个序列从自己的前半训练池等分构造 4 组，每组 10 帧。审计组固定不变，且首帧至少有 20 个有效 tracking 查询点。

- `sitting_static`：`/mnt/data/wangqq/vggt/data/adaptive_audits/firsthalf_sitting_static_k10/audit_plan.json`
- `sitting_xyz`：`/mnt/data/wangqq/vggt/data/adaptive_audits/firsthalf_sitting_xyz_k10/audit_plan.json`

## 代码验证

- Python 与 shell 语法检查：通过。
- 自适应调度器纯函数测试：6/6 通过。
- 两步端到端冒烟：通过；五项审计、训练、断点、最优纹理恢复和最终渲染均正常。
- 冒烟输出：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/adaptive_smoke_static_u2/`

## 正式运行

### sitting_static

- GPU：wangqq GPU 1
- 运行名：`perseq_v5_k10_firsthalf_sitting_static_joint_adaptive_s0_umax3000`
- 日志：`/mnt/data/wangqq/vggt/logs/perseq_v5_k10_firsthalf_sitting_static_joint_adaptive_s0_umax3000.log`
- 输出：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/perseq_v5_k10_firsthalf_sitting_static_joint_adaptive_s0_umax3000/`
- 初始固定审计最弱对数倍率：-0.125792。
- 状态：运行中。

### sitting_xyz

- GPU：wangqq GPU 3
- 运行名：`perseq_v5_k10_firsthalf_sitting_xyz_joint_adaptive_s0_umax3000`
- 日志：`/mnt/data/wangqq/vggt/logs/perseq_v5_k10_firsthalf_sitting_xyz_joint_adaptive_s0_umax3000.log`
- 输出：`/mnt/data/wangqq/vggt/outputs_attack_geometry_aware_tum10/perseq_v5_k10_firsthalf_sitting_xyz_joint_adaptive_s0_umax3000/`
- 初始固定审计最弱对数倍率：-0.073924。
- 状态：运行中。

## 结果判定

训练结束后才在各自未见后半帧上做一次正式四任务评估，并与旧 1000 步结果比较：

1. 训练审计和未见帧都继续改善：旧 1000 步训练不足。
2. 训练审计改善、未见帧不改善：优化已继续推进，但主要是训练分布过拟合。
3. 两边都不改善：旧结果已接近该配置能够达到的上限。

正式结论必须同时报告：实际停止步数、学习率变化、最佳 checkpoint 步数、训练审计曲线，以及未见后半帧四任务结果。

## 后续输入帧数消融：10 / 20 / 30 帧

2026-09-17 决定先固定一种数据划分，只研究单次送入 VGGT 的帧数：

- 三张 GPU 分别负责 `context=10`、`context=20`、`context=30`。
- 三组均使用每个序列前半段训练、后半段测评；暂不混入 even/odd 划分。
- 同一序列的三组实验使用相同 warning 初始化、v5 patch 载体、损失、正则项、随机种子和自适应停止规则。
- 训练与测评的窗口长度保持一致：10 帧训练对应 10 帧测评，20 对应 20，30 对应 30。
- Camera pose 将后半段各窗口拼成完整未见轨迹，再只做一次全局 Sim(3) 对齐；其余三项在相同窗口上汇总。

三个上下文长度、八个序列的固定训练审计计划均已生成：

`/mnt/data/wangqq/vggt/data/adaptive_audits/firsthalf_<scene>_k<10|20|30>/audit_plan.json`

启动脚本：

- 单个实验：`/mnt/data/wangqq/vggt/scripts/runs/launch_adaptive_firsthalf_context.sh`
- 一个上下文长度的八序列队列：`/mnt/data/wangqq/vggt/scripts/runs/run_adaptive_firsthalf_context_queue.sh`

当前阻塞：两台 AutoDL 均处于关机或无卡状态，SSH 端口拒绝连接；wangqq GPU 0/2 按约定不占用，GPU 1/3 正在运行上面的 10 帧收敛诊断。三卡实例开卡后即可直接启动 10/20/30 三条队列。

### L20 显存检查

随后确认 wangqq GPU 0/2 已获准使用且没有 CUDA 进程，于是分别尝试在 46 GB L20 上启动 `sitting_xyz` 的 20 帧和 30 帧版本。两者均在第 0 步固定五任务审计中 OOM，尚未进入训练：

- 20 帧：即使 `activation_checkpoint=1`，仍在 VGGT aggregator 前向中 OOM。
- 30 帧：即使 `activation_checkpoint=1`，仍在 VGGT aggregator 前向中 OOM。

这不是并发占卡或反向图保存造成的；报错发生在无梯度审计的 aggregator 前向，说明 46 GB 无法容纳当前分辨率下 20/30 帧的单次 VGGT 输入。不能通过分块输入规避，否则就不再是要比较的 20/30 帧联合输入实验。后续安排应为：L20 运行 10 帧任务，20/30 帧使用 80 GB A800。

### 四张 L20 的当前安排

确认 GPU 0/2 无 CUDA 进程并获准使用后，补齐另外两个 sitting 序列。四组均为 `firsthalf + context=10 + adaptive`：

| GPU | 序列 | 初始审计最弱对数倍率 | 状态 |
|---:|---|---:|---|
| 0 | sitting_rpy | -0.073826 | 运行中 |
| 1 | sitting_static | -0.125792 | 运行中 |
| 2 | sitting_halfsphere | -0.054333 | 运行中 |
| 3 | sitting_xyz | -0.073924 | 运行中 |

新增日志：

- `/mnt/data/wangqq/vggt/logs/perseq_v5_k10_firsthalf_sitting_rpy_joint_adaptive_s0_umax3000.log`
- `/mnt/data/wangqq/vggt/logs/perseq_v5_k10_firsthalf_sitting_halfsphere_joint_adaptive_s0_umax3000.log`
