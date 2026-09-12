# 论文 baseline 迁移到 VGGT：正式方案

这组实验比较老师点名的四类物理 patch 方法。主表尊重各论文自己的摆放、尺寸、初始化、损失、优化器和增强方式，不再把所有方法强行放到同一块显示器、同一面积或同一初始化图上。所有方法训练后再进入同一套 VGGT 四任务评测，因此可比较的是“各论文方法迁移到 VGGT 后的最终攻击效果”。

## 统一不变的部分

- 数据：同一组 TUM RGB-D 10 帧及其 GT 位姿、GT 深度、合成 point-map GT 和 reprojection-track GT。
- 受害模型：VGGT 权重全程冻结，只更新实体 patch 参数。
- 渲染：同一个三维平面投影器、深度遮挡和表面支持检查。
- 正式预算：每个配置 1000 次参数更新。论文原始训练轮数会记录，但不用于主表预算。
- 评测：Camera pose 的 ATE/RPE、Depth AbsRel、Point-map Acc、Track EPE/visF1。
- 每个实验完成后立即写入 `reports/baseline_experiments/baseline_experiment_record.md`，同时列出 Clean/Attack，并强制检查三张图存在：初始 patch、优化后 patch、优化后 patch 的 10 帧无红边合图。

## 四类方法如何对应到 VGGT

| 配置 | 原文关键做法 | 在 VGGT/TUM 中的对应 | 不能声称的部分 |
|---|---|---|---|
| `spoo_monitor/table/partition` | warning 图同时作为 content、style 和待优化图；源图按宽 600 等比缩放为 600×450；从车辆区域的大矩形开始，联合优化纹理和 `[l,r,t,b]` 四条边；面积到物体 mask 的 1/9 后冻结；纹理用 L-BFGS，边界用 Adam 且每步只更新绝对梯度最大的边 | 保留官方 600×450 warning 纹理分辨率；显示器、桌面、隔板分别作为冻结载体；完整载体代替车辆 focus 区；逆深度消失损失 | 原代码初始 mask 是车辆区域而非任意平面全域；TUM 没有对应车辆分割，因此载体平面是明确的 RGB-D 几何替代 |
| `aparate_monitor/table` | 目标检测得到完整 focus mask；patch 固定在目标框中心，线性尺度 0.2；`Uniform(0,1)` 彩色随机纹理；Adam 0.01；±20°、亮度±0.1、对比度 0.8–1.2、有界噪声±0.1；patch 内平方、外部 L1，再加 TV/NPS | 冻结载体平面替代检测框；patch 放载体中心并取宽高各 20%；focus 是完整载体；随机缩放范围因论文未给数值，适配器明确记录为 ±3% | APARATE 没有公开官方代码，也没有给数值 target depth；VGGT 版的零逆深度目标是显式适配。原 APARATE 也没有自由形状 mask |
| `3d2fool_monitor_visible_surface` | 按 NHWC 顺序生成 128×128 随机种子；最近邻放大 8 倍成为 1024×1024，6×6 独立翻转、可选再转 90°，从 6144×6144 中随机裁 4096×4096；覆盖完整三维车辆表面；消失式逆深度 + 0.1 TV + 5 NPS | 把转换后的纹理铺满显示器可见表面；保留官方种子张量顺序、逐像素等价的纹理转换和损失 | TUM 没有完整车辆网格，也没有 CARLA map/weather 标签；因此标签驱动的雨雾天气分支关闭，不能称为完整 3-D/天气复现 |
| `passive_vo_partition` | 预先固定平面屏幕；论文算法 1/2 明确用 `Uniform(0,1)` 初始化；9×9 预投影 box blur；MPRMS 位姿目标；L∞ PGD-sign；每步评估并保留历史最强扰动 | 训练前把隔板尺寸拟合到最接近 8.8% 的平均投影面积并冻结；种子 0 的彩色随机图；攻击 VGGT camera head | 原文用黑/白材质渲染图 `I0/I1` 把系数图转成相机 RGB；TUM 没有这对离线渲染图，所以 VGGT 适配把系数 `P` 直接当可打印 RGB |

APARATE 的可学习形状版本不是论文 baseline。若后续需要，可作为“APARATE + learnable mask（ours extension）”单独实验，不能混入原方法主表。

## 初始化图片的含义

每个 run 同时保存“原方法真正优化的底层变量”和“经过方法内变换后送入三维渲染器的预览”，避免把两者混为一张图：

- `geometry_patch/initial_patch.png`：初始底层变量。SPOO 是官方 600×450 warning 图；APARATE 是 64×64 彩色随机图；3D²Fool 是按官方 NHWC 随机顺序生成的 128×128 种子再最近邻放大 8 倍；Passive VO 按论文算法是种子 0 的 64×64 彩色随机系数图。
- `geometry_patch/initial_deployed_patch.png`：初始变量经过重复、裁剪或 box blur 后的确定性部署预览。
- `geometry_patch/optimized_patch.png`：最终底层变量；SPOO 会裁成最终学到的矩形区域。
- `geometry_patch/optimized_deployed_patch.png`：最终变量经过同一部署变换后的预览。

正式汇报的三张图仍使用 `initial_patch.png`、`optimized_patch.png` 和十帧合成图；部署预览作为审计附件，不替代原始初始化。

## 数据流

1. `build_paper_baseline_placements.py` 从已经提取的 TUM 三维平面中，按论文规则生成 patch 四角和 focus 四角。这个阶段不读取任何攻击分数。
2. `preflight_paper_baseline_placements.py` 用生产渲染器检查十帧正面性、覆盖、遮挡和深度表面支持；不合格就停止训练。
3. `paper_baselines.sh` 读取一个 profile，固定该论文的初始化、优化器、EOT、尺寸和摆放，调用主攻击程序训练纹理或形状。
4. 训练后用保存的同一块 patch 重新渲染，执行统一四任务评测；重渲染位姿必须与训练存档一致。
5. 生成三张图并由 `record_baseline_experiment.py` 把完整设置、数字和服务器绝对路径原子式写入实验记录。任何指标或图片缺失都会报错，不能留下“结果待补”。

## 执行方式

单配置短冒烟：

```bash
PROFILE=aparate_monitor STAGE=smoke STEPS=2 \
  bash scripts/runs/paper_baselines.sh
```

单配置正式实验：

```bash
PROFILE=aparate_monitor STAGE=formal STEPS=1000 \
  bash scripts/runs/paper_baselines.sh
```

正式配置一共七个：SPOO 三个载体、APARATE 两个载体、3D²Fool 一个可见表面适配、Passive VO 一个固定隔板。

另外可以再做一张严格同条件表，把位置、面积、初始化、EOT 和更新数全部锁死，只比较 loss；它回答的是“目标函数本身谁更强”，不能替代上面的论文原生主表。
