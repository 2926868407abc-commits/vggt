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
  `pose_scale_invariant_mse`（尺度不变版）/ `pose_pairwise_relative_mse`（校准后在
  halfsphere 上最强，**此前"无效"的结论已推翻**，见已知问题 8）/
  `feature_l1` / `pose_clean_untargeted` / `pose_{reverse,drift,scale,yaw}_targeted`

### 三序列 loss 对照（**已校准**：正则逐序列校准 + `PHYSICAL_EOT=0`，1000 步）

这是唯一可信的一版。校准前的旧表见下一节，**结论不同，不要引用旧表**。

| 序列 | loss | ATE | %ceiling | devRel | gaugeAbs |
|---|---|---|---|---|---|
| sitting_xyz | 旧 `pose_gt_untargeted` | 0.01622 | 8.1% | 0.98 | 0.916 |
| （上界 0.2005 | `scale_invariant` | 0.17445 | 87.0% | 3.02 | 0.712 |
| clean 3.9% | `pairwise` | 0.02969 | 14.8% | 3.29 | 0.950 |
| 随机 93.8%） | **`aligned_residual`** | **0.19739** | **98.5%** | **10.68** | 0.908 |
| sitting_halfsphere | 旧 `pose_gt_untargeted` | 0.18363 | 72.9% | 52.73 | 0.986 |
| （上界 0.2519 | `scale_invariant` | 0.20379 | 80.9% | 10.71 | 0.925 |
| clean 3.2% | **`pairwise`** | **0.22240** | **88.3%** | 6.81 | 0.871 |
| 随机 90.5%） | `aligned_residual` | 0.11057 | 43.9% | 0.61 | **0.303** |
| sitting_static | 旧 `pose_gt_untargeted` | 0.02191 | 70.6% | 20.84 | 0.967 |
| （上界 0.0310 | `scale_invariant` | 0.01751 | 56.4% | 8.46 | 0.936 |
| clean 19.5% | `pairwise` | 0.01862 | 60.0% | 10.72 | 0.947 |
| 随机 90.3%） | **`aligned_residual`** | **0.03069** | **98.9%** ⚠ | 3.65 | 0.732 |

⚠ 该格未收敛：末 20 步 loss 的 std 是 1.254（均值 3.366），其余各格 std ≤ 0.03。
数字本身没错，但可信度低于其他格。

**仍然没有一个 loss 通吃**：`aligned_residual` 在 xyz / static 上接近上界，
在 halfsphere 上只有 43.9%；`pairwise` 反过来在 halfsphere 上最强。

**`devRel` 与 ATE 会给出相反排序**，必须一起报。halfsphere 上旧 loss 的 devRel 是
全表最高的 52.7，ATE 却只有 72.9%——因为它 98.6% 的破坏是全局 Sim(3)，被对齐吃掉。

### 校准前的旧对照（**已作废，仅存档**）

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
   验证了帧 0 扰动下它显著低于帧 0 归一化版）。

   **⚠ 这里曾经两次下错结论。** 未校准时它在 sitting_xyz 上只有 4.2%，我判定
   "目标函数推不动攻击、不要再花时间"。**那是错的**——正则权重按旧 loss 的量级
   调好，而 pairwise 的梯度小两个数量级，一直被同一副镣铐按死。用
   `scripts/calibrate_attack_reg_balance.py` 逐序列校准后：xyz 14.8%，
   **halfsphere 88.3%（四个 loss 里最强）**。

   教训：**任何 loss 对比在正则校准之前都不算数**，见已知问题 8。当时我排除了
   梯度损坏、步长、loss 敏感度三项，也试过按比例缩正则（只缩了权重、没做完整
   校准，4.2%→5.9%），据此就下了结论——排查不完整时不该给出"不要再花时间"
   这种终局判断。

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

   **但 `ate_frac_of_ceiling` 本身也会饱和**：最强的 run 已经到 98–99%，再往上
   分辨不出强弱。所以管线同时产出 `dev_from_clean_rel`——相对该模型**自己的 clean
   预测**的 RMS 轨迹位移，除以 clean 轨迹半径。不做对齐，因而不丢规范、无上界。
   这个框架取自 NeurIPS'24 *Beware of Road Markings*：他们同样面对"相对深度尺度
   跨模型差异巨大"的问题，解法是定义 MRSR = `sum(f(x̂)-f(x))/sum(f(x))`，
   **跟自己的 clean 预测比而不是跟 GT 比**。

   两个指标会给出**相反的排序**，必须一起报。halfsphere 上旧 loss 的
   `dev_from_clean_rel` 是全表最高的 52.7，ATE 却只有 72.9%——它把轨迹推得极远
   但 98.6% 是全局 Sim(3)。

   *另一个可借鉴的报法（USENIX Adversary is on the Road）*：报绝对米数 + clean/
   attacked 成对 + 涨幅百分比，并用道路宽度之类的物理量做安全性锚定
   （"RMSE 超过 13 米，而道路宽度是…"），比纯比值更有说服力。

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
   不是目标函数。工具：`scripts/calibrate_attack_reg_balance.py`（复刻一次训练迭代，
   分别对 attack 和 reg 求 `d/d texture`，给出把各 loss 拉到同一比值所需的权重）。
   **比值逐序列不同，每个序列都要单独校准。**

   校准的实际效果：pairwise 在 xyz 上 4.2%→14.8%，在 halfsphere 上升到 88.3%
   （四个 loss 里最强）。**校准前后的排名完全不同**，这是本轮最重要的方法论结论。

   *更好的方案（来自 3DGAA，未实现）*：不预先校准，而是训练中**动态归一化**权重——
   `λ_adv = w_adv/(w_adv+w_shape+ε)`，`λ_shape = max(1-λ_adv, λ_min)`，两项损失各自
   归一化，`λ_adv` 与攻击损失成反比，攻击变强时自动转向物理真实性，`λ_min` 取 0.4
   防止塌向单一目标。这解决了本脚本的固有缺陷：它只在初始点匹配，而 aligned 的
   loss 训练中涨了 40 倍，比值早就漂走了。

9. **单元测试覆盖不到训练回路**
   `tests/test_pose_*.py` 在 float64、合成轨迹、脱离训练回路下验证 loss 数学，
   10/10 全绿却完全没能预测 pairwise 训不动。**新 loss 除数学测试外，还需要一次
   20 步冒烟并检查 `|g_attack|/|g_reg|` 与训练/测试一致性**，否则"测试全绿但训练
   不动"会重复发生。

10. **单次运行的差异小于 ~7% 不算结论**
    同一配置跑两遍，20 步内 loss 最大相对差 **7.045%**（GPU 非确定性内核逐步放大；
    第 1 步三者完全一致，之后发散）。**任何小于这个量级的 loss 差异都读不出来，
    必须先跑重复。** 已有的主结论都远在噪声之上（xyz 上 aligned 98.5% vs 旧 8.1%
    是 12 倍），不受影响。

    相关：判断"训练/测试差距"时不能只看百分比。sitting_static + aligned 曾报出
    18.2% 的差距，但该 run 末 20 步 loss 的 std 是 1.254（均值 3.366），这个差距
    只有 0.5 sigma，**是未收敛的震荡不是转移失败**。
    `check_train_test_consistency.py` 现在同时报 `gap_sigma`，只在
    「百分比 > 20% 且 > 2 sigma」时才告警。

11. **EOT 只有光度是不够的**（问题 3 的补充，来自文献对照）
    对比四篇物理攻击论文：ECCV'22 Optimal Adversarial Patches 做尺寸/旋转/亮度/
    饱和度随机化，**粘贴位置按透视模型算**（远处更小、更靠近消失点），消融显示
    EOT 让攻击性能 **+40.63%**；CVPR'24 3D²Fool 明说"EoT 不足以抵抗恶劣天气"，
    额外加曝光/阴影（多项式模型 + 高斯模糊边界）和**基于深度图的雨雾噪声**；
    π-Jack 用 EoT + 风格迁移；3DGAA 有独立的物理增强模块含深度软阴影。

    **我们的 EOT 只有 brightness/contrast/gamma/noise，对单应 `H_t` 零扰动，
    且全序列共享一个标量。** 注意反差：他们的 EOT 让攻击更强，我们的实测让
    aligned 在 halfsphere 上损失 64% 的效果——因为纯光度且共享标量时，攻击可以
    直接学去利用那个噪声。**补几何扰动（对 `H_t` 加扰动 + per-frame 独立采样）
    是正解。**

## 分段 gauge 攻击（Q1–Q4，已完成）

导师提的四个可行性问题，结论如下。

**Q1/Q2：分段 $g_i$ 能否产生单一 Sim(3) 吸收不掉的误差 —— 能，且有闭式解。**

ATE 只依赖相机位置，所以目标就是位移场 $\boldsymbol\delta\in\mathbb{R}^{3N}$。评测扣掉一个
Sim(3)，其在单位元处的作用恰好张成 **7 个方向**：3 平移 $\mathbf{e}_k$、3 旋转
$\boldsymbol\omega_k\times\mathbf{p}_i$、1 尺度 $\mathbf{p}_i$。落在这 7 维里的分量无论
多大都被抹掉（实测位移达场景半径 **109 倍**仍精确归零），正交的分量一分不少留下。

所以不用试 family——把低频余弦基的 Sim(3) 投影减掉即可（`orthogonal_mode`，正交性验证到
1e-10）。**order 2 在三序列达 89.2–93.1%**，而手工最好的 `trans_ramp` 只有 76.2% 且会饱和
（提高幅度不涨，因为幅度只改模长不改夹角）。工具：`scripts/diag_optimal_gauge.py`、
`scripts/diag_piecewise_gauge.py`。

**Q3：补丁能否推到这样的目标 —— 能，达成率 99.9–105.7%。**
超过 100% 是因为补丁沿目标方向出发后找到了更远的落点（形状残差 0.43–0.57，只有 xyz 是
0.034）。**指定方向有用，指定终点不必。**

**Q4：三个头能否同时推向一致目标 —— 能，但取决于序列。**

| 序列 | 未校准 1/1/1 | 校准后最佳 | 能否强+隐蔽 |
|---|---|---|---|
| sitting_static | 40.0% | **94.7%，0/4 触发** | ✅ |
| sitting_xyz | 3.7% | 91.0%，但 1/4 触发 | ❌ |
| sitting_halfsphere | 5.3% | 31.9% 隐蔽 / 70.6% 露馅 | ❌ 清晰权衡 |

准确的表述不是「三头联合更强」，而是**「三头联合让强度和自洽性可以被显式权衡，而不是碰运气」**。

## 已知问题 10：多项 loss 必须逐序列校准梯度（本项目已两次栽在这里）

第一次是 `pose_pairwise_relative_mse`，被同一套正则权重压死，误判为「目标函数无效」；
第二次是 Q4 三头联合损失用 1/1/1，**位姿项只拿到 2.9–8.9% 的梯度**，而深度项
（正交模态下目标是「一动别动」）独占 42–80%，等于刹车踩得比油门狠，三序列 ATE 只有
3.7% / 5.3% / 40.0%。校准后变成 91.0% / 70.6% / 94.7%——**同样的 loss 和目标，只改权重，差 25 倍。**

比值逐序列差 4 倍以上，**不能共用一套权重**。测量工具：
`scripts/diag_joint_terms.py`（三头）、`scripts/calibrate_attack_reg_balance.py`（攻击 vs 正则）。

## 工作约定

- **改动要小而可验证。** 一次只改一个问题，不要顺手重构无关代码。
- **不改评测语义。** 评测脚本只读不写；诊断需要新指标时新建脚本，不要改现有 eval。
- **多项 loss 先量梯度比值再跑，逐序列量。** 见已知问题 10——这条已经犯过两次，
  两次都得出了错误的「目标函数无效」结论。写完新的多项 loss，第一件事是跑
  `scripts/diag_joint_terms.py`，不是直接放长跑。
- **新 loss 必须配单元测试。** 尤其涉及 SE(3)/Sim(3) 的，测试要能验证不变性。
  数学测试不够，还要跑 20 步冒烟并检查 `|g_attack|/|g_reg|`，见已知问题 9。
- **标准评测协议**（做任何 loss 对比都按这个来，四条缺一不可）：
  1. 至少三个序列：sitting_xyz、sitting_halfsphere、sitting_static（后者是退化对照）。
     排除 walking_halfsphere（可见性 0.370）和 walking_xyz（有帧掉到 0.052）——
     补丁遮挡太重，测的是遮挡不是攻击。
  2. `PHYSICAL_EOT=0`，消除训练/测试差距这个混淆（几何 EOT 补上之前）。
  3. **逐序列跑 `calibrate_attack_reg_balance.py`**，按输出设正则权重。
  4. 同时报 `ate_frac_of_ceiling` 和 `dev_from_clean_rel`，两者排序可能相反。
- **差异小于 ~7% 一律先跑重复再下结论**，见已知问题 10。
- **旧 loss 全部保留**，作为论文 ablation 的对照行，不要删除或原地替换。
- 长实验前先用 `ITERATIONS=20` 冒烟，确认能跑通再放长。
