# quickTDI 软件文档

## 1. 文档信息

- 项目名称：`quickTDI`
- 当前版本：`0.1.0`（见 `pyproject.toml`）
- 文档修订日期：2026-07-12
- 运行环境：Python `>=3.9`
- 适用领域：空间引力波探测（Taiji/LISA）中的时延干涉（TDI）仿真与分析
- 核心依赖：`numpy`、`scipy`
- 可选依赖：`dynesty`（贝叶斯推断）
- 测试规范：[`quickTDI_软件测试大纲与细则.md`](quickTDI_软件测试大纲与细则.md)

## 2. 项目概述

`quickTDI` 是一个面向 Taiji/LISA 的端到端 TDI 数值仿真框架，覆盖以下链路：

`Orbit -> Gravitational Wave -> Optical Bench -> Laser Locking -> Signal Synthesis -> TDI`

高层统一入口为 [`TJ_Triangle.Triangle`](../src/TJ_Triangle.py)。使用者也可以直接组合底层模块，用于算法开发、数值检查和定制研究。

当前版本主要用于研究和开发验证。正式科学结果应同时给出参数、单位、随机种子、依赖版本、边界处理方法和独立物理校验，不应只以“程序成功运行”作为正确性依据。

## 3. 目标、用户与边界

本文档面向：

- 进行空间引力波仿真、TDI 和灵敏度分析的研究人员；
- 扩展轨道、噪声、锁定或 TDI 算法的开发人员；
- 将 quickTDI 集成到数据分析或推断流程的工程人员。

项目当前目标：

- 提供解析或文件驱动的三航天器轨道与时延算子；
- 生成单源或多源单色 GW 链路响应；
- 模拟六个光学平台及主要噪声分量；
- 提供受支持的激光锁定、信号合成和预定义 TDI 通道；
- 提供基于 PSD 加权频域似然的 dynesty 推断入口。

当前不包括：

- 完整任务级仪器模型和数据处理流水线；
- 分布式计算或生产服务；
- 对所有参数区域的科学精度认证；
- dynesty 收敛性或后验可信度的自动认证；
- `TJ_freplan.py` 的频率规划功能；该文件当前仅为占位模块。

### 3.1 桌面图形界面

项目提供基于 Tkinter/ttk 的桌面控制面板。它支持配置仿真、管理多个 GW 源、选择噪声与锁定模式、在后台线程执行完整流水线、查看 ASD/时域结果以及导出 PNG/PDF/NPZ。

开发环境中可直接运行：

```bash
PYTHONPATH=src python scripts/run_dashboard.py
```

完成可编辑安装后也可使用：

```bash
quicktdi-dashboard
```

## 4. 仓库结构

```text
quickTDI/
  src/
    TJ_Triangle.py     # 高层流水线封装
    TJ_orbit.py        # 轨道、臂长与时延模型
    TJ_gw.py           # 引力波链路响应
    TJ_ob.py           # 光学平台、噪声与测量通道
    TJ_lock.py         # 激光锁定
    TJ_synthesis.py    # sci/tes/ref/xi/eta 合成
    TJ_tdi.py          # TDI 通道定义与求值
    TJ_noise.py        # 白噪声、幂律噪声、acc/oms
    TJ_constant.py     # 常数与臂映射
    TJ_freplan.py      # 占位模块，当前无实现
    TJ_dashboard.py    # Tkinter 桌面仿真控制面板
  utils/
    TJ_infer.py        # PSD、似然与 dynesty 推断工具
  scripts/
    run_dashboard.py   # 桌面控制面板启动入口
    run_dynesty.py     # 推断命令行入口
  docs/
    SOFTWARE_DOCUMENTATION.md
    quickTDI_软件测试大纲与细则.md
  test.ipynb           # 手工演示与验证 Notebook
  pyproject.toml       # 打包与依赖声明
```

## 5. 系统架构与对象生命周期

### 5.1 流水线顺序

1. `TJ_orbit.orbit`：构造或加载轨道，提供位置、速度、臂长和时延算子。
2. `TJ_gw.gw`：根据轨道生成六条有向链路的 GW 响应。
3. `TJ_ob.ob`：创建六个光学平台和噪声/测量数组。
4. `TJ_lock.lock`：执行激光锁定并传播激光状态。
5. `TJ_synthesis.synthesis`：合成 `sci/tes/ref/xi/eta`。
6. `TJ_tdi.tdi`：对 `eta` 施加延迟并计算 TDI 通道。

通常必须按上述顺序调用。`Triangle` 会在缺失关键前置对象时抛出 `RuntimeError`，但部分底层类默认调用者已经完成前置准备。

### 5.2 高层编排器

`Triangle` 的主要能力：

- GW 源管理：`add_gw_source`、`set_gw_sources`、`add_gw_frequency_series`；
- 组件构建：`create_gw_object`、`setup_optical_benches`；
- 仿真执行：`apply_laser_lock`、`synthesize_signals`、`run_tdi`；
- 一键流程：`run_full_pipeline`；
- 结果管理：`tdi_results`、`get_tdi_result`、`get_laser_data`、`summary`。

### 5.3 缓存和重新计算

`Triangle` 会保存已经创建的 GW、bench、lock、synthesis 和 TDI 对象。`run_full_pipeline` 遇到已有组件时可能跳过重新构建。

因此：

- 首次运行前应完成源和噪声配置；
- 运行后修改 GW 源、采样参数或 bench 配置时，不应假设所有下游对象会自动失效；
- 对要求严格可复现的参数扫描，建议为每组结构性配置创建新的 `Triangle`，或显式按依赖顺序重建 GW、synthesis 和 TDI；
- 重复使用相同 `store_name` 时，`tdi_results` 中原结果会被覆盖。

## 6. 数据与物理约定

### 6.1 单位体系

- 时间：秒（s）；
- 距离：光秒（light-seconds）；
- 速度：无量纲 `v/c`；
- 频率：Hz；
- 天球角和偏振角：弧度；
- GW `strain`：无量纲。

调用者必须保证单位一致，当前接口不会为所有参数自动做单位转换。

### 6.2 航天器与链路编号

- 航天器编号：`1, 2, 3`；
- 有向臂编号：`1, 2, 3, -1, -2, -3`；
- 标准数组顺序：`ARM_ARRAY = [3, -2, -3, 1, 2, -1]`；
- 映射位于 [`TJ_constant.py`](../src/TJ_constant.py)：
  - `ARM_REC`：arm → receiver；
  - `ARM_SEN`：arm → sender；
  - `RS_ARM`：receiver-sender → arm。

新增链路算法时必须复用这些映射，避免在模块中建立不一致的本地编号规则。

### 6.3 时间数组与采样

模块通常使用半开区间：

```python
np.arange(t_start, t_end, 1 / fsample)
```

因此样本数通常为 `(t_end-t_start)*fsample`，但当该乘积不是整数时应以实际数组长度为准。输入必须满足：

- `t_end > t_start`；
- 采样率和轨道步长为正；
- 需要互相组合的数组具有一致的时间范围和采样约定。

### 6.4 时延模式

- `delay_level='1'`：调用 `orbit.delay`，递归使用随时间变化的有向臂长；
- `delay_level='0'`：调用 `orbit.delay_0`，每个臂使用固定 `tri_arm`；
- 空臂列表应保持时间数组不变；
- 复合时延按照实现定义从臂列表右侧向左侧依次施加。

`delay_level` 应只传字符串 `'0'` 或 `'1'`。当前部分底层路径对非法值的错误消息仍不统一，外部代码应提前校验。

### 6.5 插值与边界

- 轨道距离和 GW 响应使用三次插值；
- 光学平台信号使用 31 阶插值；
- 光学平台和 GW 插值器在时间域外通常返回 `1e-99`，作为数值上的近零填充值；
- 轨道文件位置插值在域外采用外推，而距离插值在域外回填标称臂长。

重要约束：

- 文件轨道三次插值至少需要 4 个有效时间点；
- 光学平台 31 阶插值至少需要 32 个样本；
- 小型演示或测试必须同时满足 OB 和 GW 插值器的最小样本数；
- 分析延迟结果时应排除由最大复合时延和插值阶数决定的边界保护区。

`1e-99` 是边界处理约定，不代表物理模型给出的严格零值。

### 6.6 随机数与可复现性

当前噪声函数使用 NumPy 全局随机状态。若需要复现：

```python
np.random.seed(1234)
# 创建所有包含随机噪声的对象
```

要复现同一数组，必须在每次生成前重新设置相同种子；固定一次种子后连续两次调用会得到不同随机数组。正式结果应记录随机种子、NumPy 版本和噪声参数。

## 7. 核心模块说明

### 7.1 `TJ_constant`

提供物理常量、时间单位、标称臂长、天文单位和有向臂映射。兼容类 `constant` 保留旧式属性访问，新代码应优先使用模块级大写常量。

### 7.2 `TJ_orbit`

支持：

- `data_source='analytical'`：解析日心或地心轨道；
- `data_source='file'`：从 CSV 加载位置，可选加载速度和距离；
- 可选的距离高斯扰动 `distance_noise_sigma`。

关键接口：

- `position(sc_i, t)`；
- `velocity(sc_i, t)`；
- `dij(arm_num, t)`；
- `delay(t_arr, arm_array)`；
- `delay_0(t_arr, arm_array)`。

文件格式：

- positions：`time, sc1_x, sc1_y, sc1_z, ..., sc3_z`，共 10 列；
- velocities：`time, sc1_vx, sc1_vy, sc1_vz, ..., sc3_vz`，共 10 列；
- distances：`time, d_arm3, d_arm-2, d_arm-3, d_arm1, d_arm2, d_arm-1`，共 7 列。

positions 是文件模式必需项。未提供 velocities 时通过位置有限差分计算；未提供 distances 时由位置欧氏距离计算。当前实现对少列、重复时间、非单调时间和多文件时间轴不一致的主动校验有限，数据进入程序前应先验证。

### 7.3 `TJ_gw`

- 支持单个单色源参数或 `source_list` 多源叠加；
- 输出六臂数组 `_gw_arr` 并提供 `gw[arm_num](t)` 和 `ygw(arm_num,t)`；
- `hasGW=False` 时有效时间域内响应为零；
- 多源响应按线性叠加计算。

每个源字典应包含：

```python
{
    "fgw": 0.007,
    "strain": 1e-23,
    "beta": 0.0,
    "lamda": 0.0,
    "psi": 0.0,
}
```

使用极端天区方向时应检查分母 `1-k·n` 附近的数值稳定性。

### 7.4 `TJ_noise`

提供白噪声、正/负幂律 PSD 噪声、截断谱噪声、`acc` 和 `oms`。噪声正确性应通过长度、有限性、统计量和指定频带内的 Welch PSD 共同验证。

`white` 的样本标准差按 `sigma*sqrt(fsample/2)` 缩放，以维持采样率变化下的 PSD 约定。幂律噪声包含 IIR/FIR 处理，分析谱形时应考虑初始瞬态和低频截止。

### 7.5 `TJ_ob`

光学平台对象支持：

- `hasLaser`、`hasAcc`、`hasOms`；
- 原始数组 `_laser/_lock/_acc/_oms/_clock`；
- 测量数组 `_sci/_tes/_ref/_xi/_eta`；
- 对应的无下划线插值器；
- `set_noise` 和各类 `update_*` 方法。

直接修改下划线数组后，必须调用相应 `update_*` 刷新插值器，否则后续延迟访问仍可能读取旧数据。

### 7.6 `TJ_lock`

高层 API 支持三种锁定模式：

- `single`：单臂锁定，可选 `method='backward'` 或 `'trapezoidal'`；
- `dual`：双臂锁定；
- `common`：公共臂锁定。

`run()` 返回锁定后的激光数组，并按实现的锁相规则更新光学平台。所有 bench 均未启用激光噪声时，锁定计算会被跳过。

锁定正确性不能仅通过“输出有限”判断，应比较锁定前后的 RMS 和指定频段 PSD 抑制比。正式验收阈值见测试大纲和项目回归基线。

### 7.7 `TJ_synthesis`

合成各 bench 的 `sci/tes/ref`，计算中间量 `xi`，再生成 TDI 输入 `eta`。支持可变臂长和常臂长两条时延路径，并预计算部分时延以减少重复工作。

当 GW、laser、acc、oms 均关闭时，有效时间域内输出应接近对应解析零结果；边界处仍受插值填充值影响。

### 7.8 `TJ_tdi`

预置通道包括：

- Michelson：`X1/Y1/Z1`、`X2/Y2/Z2`；
- Sagnac/monitor 等辅助通道：`a1/a15/a2`、`z1/z15/z2` 等；
- 其他公开定义见 [`TJ_tdi.py`](../src/TJ_tdi.py)。

通道元素格式：

```python
[eta_label, arm_delays, sign]
```

其中：

- `eta_label` 为 `'1'/'2'/'3'/'1p'/'2p'/'3p'`；
- `arm_delays` 为有向臂列表；
- `sign` 为 `'+'` 或 `'-'`。

辅助函数：

- `cycle_tdi`：循环置换 spacecraft/arm 标签；
- `delay_tdi`：为通道项追加延迟并组合符号。

TDI 的核心验收指标是仅激光噪声条件下的归一化残差或 PSD 抑制，而不只是数组长度和无 NaN。

### 7.9 `TJ_Triangle`

推荐的外部入口。主要构造参数：

- `t_start=0.0`；
- `t_end=40000.0`；
- `tri_arm=10.0`；
- `orbit_type='heliocentric'`；
- `fsample_ob=5.0`；
- `fsample_gw=0.1`。

默认配置会产生较大数组。开发冒烟和单元测试应显式使用较短时长，同时满足插值最小样本数。

### 7.10 `utils/TJ_infer`

主要对象和函数：

- `Parameter`：uniform/log_uniform/normal/fixed 先验；
- `InferenceConfig`：通道、仿真、PSD 和噪声配置；
- `TriangleSimulator`：重复 likelihood 计算的仿真包装；
- `prepare_data`、`compute_psd`、`noise_psd_model`；
- `log_likelihood`、`run_dynesty`。

当前实现要求模型输出长度与每个观测通道严格相同。时间轴应非空、严格递增且近似均匀；外部 PSD 应有限、非负、频率有序。部分输入校验仍计划通过测试驱动补齐，调用者应在进入推断前先检查数据。

## 8. 安装与环境准备

基础安装：

```bash
python3 -m pip install numpy scipy
python3 -m pip install -e .
```

包含推断：

```bash
python3 -m pip install -e .[inference]
```

开发和测试：

```bash
python3 -m pip install pytest pytest-cov
PYTHONPATH=src python3 -c "from TJ_Triangle import Triangle; Triangle(t_end=40)"
```

项目声明支持 Python 3.9 及以上。正式支持矩阵和依赖版本应以测试报告为准；“能够导入”不等同于所有物理结果已经完成兼容性验证。

## 9. 标准使用流程

### 9.1 端到端仿真

```python
import numpy as np
import TJ_tdi
from TJ_Triangle import Triangle

np.random.seed(1234)

tri = Triangle(
    t_start=0,
    t_end=40000,
    tri_arm=10.0,
    orbit_type="heliocentric",
    fsample_ob=5.0,
    fsample_gw=0.1,
)
tri.add_gw_source(
    fgw=0.007,
    strain=1e-23,
    beta=0.0,
    lamda=0.0,
    psi=0.0,
)
tri.create_gw_object(hasGW=True)
tri.setup_optical_benches(hasLaser=True, hasAcc=False, hasOms=False)
tri.apply_laser_lock(
    lock_type="dual",
    arm1_delay=20.1,
    arm2_delay=19.9,
)
tri.synthesize_signals(delay_level="1")
x1 = tri.run_tdi(
    TJ_tdi.X1,
    delay_level="1",
    store_name="X1",
)
```

### 9.2 一键全流程

```python
from TJ_Triangle import Triangle

tri = Triangle()
tri.add_gw_source(fgw=0.003, strain=1e-23)
tri.run_full_pipeline(delay_level="1", lock_type="dual")
print(tri.tdi_results.keys())
```

若尚未添加 GW 源，`create_gw_object(hasGW=True)` 会加入默认源。默认全流程会计算 `X1/X2/Y1/Z1`。

### 9.3 文件轨道

```python
from TJ_orbit import orbit

orb = orbit(
    tri_arm=10.0,
    data_source="file",
    data_files={
        "positions": "positions.csv",
        "velocities": "velocities.csv",
        "distances": "distances.csv",
    },
)
```

加载前应检查列数、时间递增性、有限值、各文件时间轴和单位。

### 9.4 贝叶斯推断脚本

```bash
python3 scripts/run_dynesty.py \
  --data path/to/data.npz \
  --channels X1 \
  --fsample 5.0 \
  --nlive 200 \
  --dlogz 0.1 \
  --out dynesty_results.npz
```

输入 NPZ：

- 每个请求通道必须有同名一维数组，如 `X1`；
- 可选 `t`；
- 可选 `psd_freqs_X1` 和 `psd_vals_X1`；
- 多通道长度必须一致。

当前 CLI 将仿真范围固定为 `t_start=0`、`t_end=40000`，并设置 `fsample_ob=--fsample`。因此数据长度必须与 `40000*fsample` 个模型样本一致；默认 5 Hz 时为 200000 点。CLI 尚未提供 `--t-start/--t-end`，不适合直接用于任意长度的小样本文件。需要小型推断测试时，应直接使用 `utils.TJ_infer` 配置较短的 `InferenceConfig`，或先扩展 CLI 的时间参数。

## 10. 对外接口与兼容性建议

推荐外部依赖：

- `TJ_Triangle.Triangle`；
- `TJ_tdi` 中预定义的公开通道；
- `utils.TJ_infer` 的公开配置和采样入口；
- `TJ_constant` 的模块级大写常量。

以下内容视为内部实现：

- 下划线前缀数组和状态，如 `_laser`、`_gw_arr`、`_gw_sources`；
- 各模块临时缓存和插值器布局；
- 未在 `Triangle` 或本节列出的实验性路径。

当前版本尚未声明严格的语义化 API 稳定性。外部项目升级 quickTDI 时应运行自己的回归测试。

## 11. 数值验证与结果解释

### 11.1 推荐的正确性检查

- 映射：六个 arm 与 sender/receiver 双向一致；
- 时延：空链恒等、常臂公式、复合臂逐步参考；
- GW：零应变、幅度线性、多源叠加、标量参考；
- 噪声：均值、方差、Welch PSD 和频谱斜率；
- 锁定：锁定前后 RMS 和有效频带 PSD；
- TDI：仅激光噪声时的残余和仅 GW 时的信号保留；
- 推断：注入真值的 likelihood 高于明显偏离参数。

### 11.2 抑制指标

测试大纲统一采用：

- `R_rms = RMS(output) / max(RMS(reference_input), ε)`；
- `R_psd = median(PSD_output / PSD_reference)`；
- `D_dB = -10*log10(R_psd)`。

计算时必须同时报告频带、窗函数、分段长度、边界保护区和参考输入。不同设置下的 dB 数值不能直接比较。

### 11.3 回归结果

参考数组必须附带：

- 源、轨道、噪声和锁定参数；
- 随机种子；
- 单位和时间轴；
- quickTDI、Python、NumPy、SciPy 版本；
- 容差、生成脚本和校验值。

历史结果只能用于检测变化，不能替代解析极限、独立公式或标量参考实现。

## 12. 性能特征

主要耗时和内存受以下因素影响：

- `(t_end-t_start)`；
- `fsample_ob` 和 `fsample_gw`；
- GW 源数量；
- 锁定模式；
- TDI 通道项数和最长复合时延；
- likelihood 调用次数。

默认 `Triangle` 配置包含 200000 个 OB 样本，适合完整仿真但不适合快速单元测试。开发测试应使用满足最小插值点数的短时 fixture。

推断中 `TriangleSimulator` 会复用部分对象，但每次参数变化仍会重建 GW 和 synthesis。开始大规模采样前应先测量单次 likelihood 耗时和峰值内存。

## 13. 验证与测试状态

测试规格见 [`quickTDI_软件测试大纲与细则.md`](quickTDI_软件测试大纲与细则.md)，包括：

- H/M/L 测试优先级；
- S1 至 S4 缺陷严重度；
- 物理正确性、数值容差和边界保护；
- 模块、集成、推断、性能和回归用例；
- 测试报告和需求追溯要求。

当前仓库状态：

- `test.ipynb` 是已有的手工演示和验证入口；
- 尚未提交完整的 `tests/` 自动化测试目录；
- 尚未形成经评审的 TDI/锁定抑制阈值和版本化数值基线；
- 因此 v0.1.0 不能仅依据现有 Notebook 声明完成正式软件验收或全面科学认证。

后续测试实现应以测试大纲中的 H 级物理用例和异常输入用例为第一优先级。

## 14. 故障排查

### 14.1 前置对象错误

- `RuntimeError: GW object not created`
  - 先调用 `create_gw_object()`。
- `RuntimeError: Optical benches not set up`
  - 先调用 `setup_optical_benches()`。

### 14.2 插值创建失败

- 检查 OB 样本是否至少 32 点；
- 检查 GW 和文件轨道样本是否至少 4 点；
- 检查时间数组是否严格递增且没有重复值；
- 检查数据是否包含 NaN/Inf。

### 14.3 TDI 结果异常

- 确认已经完成 synthesis；
- 检查 eta 标签、臂编号、符号和 delay level；
- 排除时序两端的延迟/插值保护区；
- 分别在仅 laser、仅 GW 和全部关闭条件下定位问题；
- 不要把域外填充值 `1e-99` 当作物理响应。

### 14.4 推断长度不一致

- 确认所有数据通道等长；
- 确认时间步长与 `--fsample` 一致；
- 当前 CLI 要求数据长度匹配 40000 秒模型；
- 小样本请直接使用自定义 `InferenceConfig`。

### 14.5 导入失败

```bash
python3 -m pip install -e .
```

或：

```bash
PYTHONPATH=src python3 your_script.py
```

## 15. 扩展规范

- 新模块放入 `src/`，遵循 `TJ_*` 命名。
- 类使用 `PascalCase`，函数使用 `snake_case`，常量使用 `UPPER_SNAKE_CASE`。
- Docstring 使用 NumPy 风格的 `Parameters`、`Returns`、`Raises`、`Examples`。
- 公共接口应给出类型、单位、合法范围、异常和边界行为。
- 新增 TDI 通道必须保持 `[eta_label, arm_delays, sign]` 结构，并增加：
  - 通道结构测试；
  - 三次循环或对应对称性测试；
  - 仅激光噪声消除测试；
  - 仅 GW 信号保留测试。
- 新增随机函数应优先接受 `numpy.random.Generator`，减少对全局随机状态的依赖。
- 修改数值算法时必须提供独立 Oracle、极限情况和回归基线更新说明。
- 新功能必须同步更新本文件、测试大纲和自动化测试。

## 16. 当前限制与改进路线

当前限制：

- 尚无完整自动化回归测试和 CI；
- 部分公共输入缺少主动校验，错误可能表现为底层 NumPy/SciPy 异常；
- 部分模块共享可变对象和内部数组，状态失效规则尚未完全封装；
- 高阶插值对最小样本数和边界区较敏感；
- 噪声使用全局 NumPy RNG；
- TDI 和锁定尚缺经评审的定量抑制基线；
- 推断 CLI 固定为 40000 秒仿真，且小样本使用不便；
- `TJ_freplan.py` 尚未实现。

优先改进：

1. 按测试大纲建立 `pytest + CI`，首先覆盖 H 级物理和异常用例。
2. 为构造参数、CSV、TDI 通道、prior、时间轴和 PSD 增加统一校验。
3. 建立 TDI/锁定的理论或独立参考基线，并版本化保存参数与容差。
4. 明确 `Triangle` 组件依赖和自动失效/重建规则。
5. 将随机接口迁移为显式 `numpy.random.Generator`。
6. 为推断 CLI 增加 `--t-start`、`--t-end` 和更完整的数据一致性检查。
7. 建立固定硬件上的耗时、内存和 likelihood 性能基线。
8. 增补 README、API 参考和发布兼容性说明。
