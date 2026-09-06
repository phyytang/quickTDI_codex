# quickTDI 软件文档

## 1. 项目概述

quickTDI 是一个面向空间引力波探测（Taiji/LISA）的**时延干涉测量（TDI）**仿真工具包。该项目实现了从航天器轨道建模到 TDI 组合输出的完整数值仿真流水线，主要功能包括：

- 航天器星座轨道建模（日心/地心解析轨道与外部数据文件支持）
- 引力波（GW）信号响应生成（单/多色源，含天线方向图）
- 光学平台（Optical Bench）噪声仿真（激光频率噪声、加速度噪声、OMS噪声）
- 激光臂锁定（Arm-locking）频率稳定算法
- 信号合成（相位计输出与 η 信号计算）
- TDI 组合计算（第一/二代 Michelson、Sagnac 等通道）
- 基于 Tkinter 的图形化桌面控制台（Dashboard）

**依赖**：Python ≥ 3.9、NumPy、SciPy、Matplotlib  
**安装**：`pip install -e .` 或 `pip install -e .[inference]`

---

## 2. 模块说明

### 2.1 `TJ_constant.py` — 物理常数与星座结构常量

**功能**：定义仿真所需的物理常数、单位换算和星座结构映射字典。

#### 主要常量

| 常量 | 含义 | 值/来源 |
|---|---|---|
| `SPEED_OF_LIGHT` | 真空光速 | 2.9979×10⁸ m/s |
| `ARM` | 臂长（光秒） | 3×10⁶ km / c ≈ 10.008 s |
| `AU` | 日地距离（光秒） | 约 499.0 s |
| `FREQ_ORBIT` | 轨道频率 | 1 / 年 ≈ 3.17×10⁻⁸ Hz |
| `PI` | 圆周率 | math.pi |
| `YEAR`, `MONTH`, `WEEK`, `DAY`, `HOUR` | 时间单位 | 秒 |

#### 星座结构映射

| 字典 | 功能 | 典型映射 |
|---|---|---|
| `ARM_REC` | 臂号 → 接收航天器 | `{1: 2, 2: 3, 3: 1, -1: 3, -2: 1, -3: 2}` |
| `ARM_SEN` | 臂号 → 发送航天器 | `{1: 3, 2: 1, 3: 2, -1: 2, -2: 3, -3: 1}` |
| `RS_ARM` | 收发对 → 臂号 | `{12: 3, 21: -3, 13: -2, 31: 2, 23: 1, 32: -1}` |
| `ARM_ARRAY` | 标准臂数组 | `[3, -2, -3, 1, 2, -1]`（对应 RS 对 [12, 13, 21, 23, 31, 32]） |

**向后兼容**：保留 `constant` 类作为类级接口（已弃用，推荐直接使用模块级常量）。

---

### 2.2 `TJ_orbit.py` — 航天器轨道模块

**功能**：提供航天器 1/2/3 的位置、速度和臂长距离计算，支持解析轨道和外部文件导入两种模式。

#### 核心函数

- **`sc_pos_analytic(sc_i, t, armlen, orbit_type)`**：解析开普勒轨道位置（二阶偏心率展开）。
  - `orbit_type='heliocentric'`：日心轨道，偏心率为 `ecc = armlen/(2√3·AU)`。
  - `orbit_type='geocentric'`：地心轨道，使用地球轨道参数 e=0.0167、航天器偏心率 e1=0.05。
- **`sc_vel_analytic(sc_i, t, armlen, orbit_type)`**：通过中心有限差分计算速度（步长 1/20 s）。
- **`sc_dis_analytic(sc_i, sc_j, t, ...)`**：两航天器间的欧几里得距离。
- **`sc_dis_arm_analytic(arm_num, t, ...)`**：沿指定臂的距离。
- **`nij(sc_i, sc_j, t, armlen, level)`**：臂单位方向向量（从发送方指向接收方）。
  - `level=0`：同时间计算。
  - `level=1`：发送方取延迟时间 `t - armlen`。
- **`vij(sc_i, sc_j, t, armlen, level)`**：两航天器速度差向量。

#### 类 `orbit`

主要的轨道对象，管理时间网格上的臂长距离插值器。

**构造参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `t_start` | 0.0 | 起始时间（s） |
| `t_end` | 10000.0 | 结束时间（s） |
| `t_step` | 1.0 | 时间步长（s） |
| `tri_arm` | 10.0 | 三角形臂长（光秒） |
| `orbit_type` | `'heliocentric'` | 轨道类型 |
| `data_source` | `'analytical'` | 数据来源（`'analytical'` / `'file'`） |
| `data_files` | None | CSV 文件路径字典（文件模式必填） |
| `distance_noise_sigma` | 0.0 | 距离高斯噪声标准差（光秒） |

**关键方法**：

- **`position(sc_i, t)`**：返回航天器 `sc_i` 在时刻 `t` 的位置向量 `[x, y, z]`（光秒）。
- **`velocity(sc_i, t)`**：返回速度向量 `[vx, vy, vz]`（无量纲 v/c）。
- **`dij(arm_num, t)`**：返回沿臂 `arm_num` 的发送-接收距离（光秒）。使用三次样条插值。
- **`delay(t_arr, arm_array)`**：TDI 核心延迟算子。对时间数组递归应用光行时延迟。
  - 例如 `delay(t, [2, -2])` 计算沿臂 2 往返的延迟时间。
- **`delay_0(t_arr, arm_array)`**：常数臂长近似的简化延迟。

**文件模式 CSV 格式**：

- `positions.csv`：`time, sc1_x, sc1_y, sc1_z, sc2_x, sc2_y, sc2_z, sc3_x, sc3_y, sc3_z`
- `velocities.csv`：`time, sc1_vx, sc1_vy, sc1_vz, sc2_vx, sc2_vy, sc2_vz, sc3_vx, sc3_vy, sc3_vz`
- `distances.csv`：`time, d_arm3, d_arm-2, d_arm-3, d_arm1, d_arm2, d_arm-1`

**访问结构**：

- `d_int`：6 个臂距离的列表插值器（顺序 `[3, -2, -3, 1, 2, -1]`）
- `dint`：3×3 矩阵 `[receiver-1][sender-1]` 索引的距离插值器
- `di`：按臂号索引 `[None, 1, 2, 3, -3, -2, -1]` 的距离插值器

---

### 2.3 `TJ_gw.py` — 引力波信号生成模块

**功能**：计算引力波在各航天器臂上的应变响应，支持单色/多色源叠加。

#### 核心函数

- **`uvec(beta, lamda)`**：GW 偏振基向量 **u**。
- **`vvec(beta, lamda)`**：GW 偏振基向量 **v**。
- **`kvec(beta, lamda)`**：波传播方向向量 **k**（从源指向探测器）。
- **`pvec(beta, lamda, psi)`**：偏振向量 **p** = cosψ·**u** + sinψ·**v**。
- **`qvec(beta, lamda, psi)`**：偏振向量 **q** = -sinψ·**u** + cosψ·**v**（**p** ⊥ **q**）。
- **`eplus(beta, lamda)`**：3×3 加偏振张量 **e₊** = **u**⊗**u** - **v**⊗**v**。
- **`ecros(beta, lamda)`**：3×3 叉偏振张量 **e×** = **u**⊗**v** + **v**⊗**u**。
- **`xipls(beta, lamda, nij)`**：加偏振天线方向图系数 = (n·u)² - (n·v)²。
- **`xicrs(beta, lamda, nij)`**：叉偏振天线方向图系数 = 2(n·u)(n·v)。
- **`ygwsr(arm_num, t, orbit_data, fgw, strain, ...)`**：单臂单时刻的 GW 应变响应。考虑光行时间效应和天线方向图。

#### 类 `gw`

**构造参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `orbits` | — | 轨道对象 |
| `t_start` | 0.0 | 起始时间（s） |
| `t_end` | 10000.0 | 结束时间（s） |
| `fsample` | 2 | 采样频率（Hz） |
| `fgw` | 0.001 | 单源 GW 频率（Hz） |
| `strain` | 1e-20 | 单源应变振幅 |
| `hasGW` | False | 是否启用 GW 生成 |
| `source_list` | None | 多源字典列表（覆盖单源参数） |

**source_list 格式**：
```python
[
    {'fgw': 0.001, 'strain': 1e-20, 'beta': 0, 'lamda': 0, 'psi': 0},
    {'fgw': 0.002, 'strain': 5e-21, 'beta': 1.0, 'lamda': 2.0, 'psi': 0.5}
]
```

**访问结构**：

- `gw_int`：6 个臂的 GW 应变插值器列表
- `gwint`：3×3 矩阵索引 `[receiver-1][sender-1]`
- `gw`：按臂号索引 `[None, 1, 2, 3, 3p, 2p, 1p]`（对应臂 1, 2, 3, -3, -2, -1）
- `ygw(arm_num, t)`：获取指定臂时间的插值应变

---

### 2.4 `TJ_noise.py` — 噪声生成模块

**功能**：生成具有不同功率谱密度（PSD）特性的噪声时间序列。

#### 噪声类型总览

| 函数 | PSD 特性 | 实现方法 |
|---|---|---|
| `white()` | 平坦（白噪声） | `x[n] ~ N(μ, σ²·f_s/2)` |
| `power_2()` | ∝ f² | 一阶差分 FIR 滤波 |
| `power_m2()` | ∝ 1/f² | 一阶 IIR 递归滤波（α=0.9999） |
| `power_4()` | ∝ f⁴ | 二阶差分 FIR 滤波 |
| `power_m4()` | ∝ 1/f⁴（不稳定） | 直接二阶 IIR 递归 |
| `power_m4_2()` | ∝ 1/f⁴（稳定版） | 两级级联 power_m2() |
| `m2_trun()` | ∝ 1/f²（截断） | Plaszczynski (2007) 算法，可设截止频率 |
| `m4_trun()` | ∝ 1/f⁴（截断） | 两级级联 m2_trun() |
| `oms()` | OMS 组合噪声 | f² + 1/f² 联合模型 |
| `acc()` | 加速度噪声 | 白噪声 + f² + 1/f² + 1/f⁴ 四分量模型 |

#### 关键函数

- **`white(length, mean, sigma, fsample)`**：白高斯噪声，振幅 `sigma·√(fsample/2)` 以保持 PSD 归一化。
- **`power_2(length, ...)`**：离散差分 `y[n]=x[n]-x[n-1]`，归一化因子 `fsample/(2π)`。
- **`power_m2(length, ...)`**：递归积分 `y[n]=α·y[n-1]+x[n]`，全局参数 `alpha_para=0.9999`。
- **`acc(length, fsample, lowfz, strain)`**：LISA 检验质量加速度噪声，两个拐角频率 f₁=0.4 mHz, f₂=8.0 mHz。
- **`oms(length, fsample, lowfz, highfz, strain)`**：光学计量系统噪声，拐角频率 f₃=2.0 mHz。

**PSD 归一化**：所有有色噪声函数将应变（无单位）转换为相位单位（除以光速 3×10⁸ m/s），并归一化使得在 1 Hz 处 PSD 值正确。

---

### 2.5 `TJ_ob.py` — 光学平台模块

**功能**：模拟每颗航天器上的光学平台仪器，包括噪声生成和相位计输出。

#### 类 `ob`

**构造参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `t_start` | 0.0 | 起始时间（s） |
| `t_end` | 10000.0 | 结束时间（s） |
| `fsample` | 2 | 采样频率（Hz） |
| `hasLaser` | True | 启用激光频率噪声 |
| `LNamp` | 1e-13 | 激光噪声振幅 |
| `hasAcc` | False | 启加速度噪声 |
| `hasOms` | False | 启用 OMS 噪声 |
| `lowfz` | 1e-5 | 低频截止（Hz） |

#### 内部时间序列（`_` 前缀为原始数据，无前缀为插值器）

| 数据 | 插值器 | 说明 |
|---|---|---|
| `_laser_noise` | `laser` | 自由运转激光频率噪声 |
| `_lock_laser` | `lock` | 锁定后的激光噪声 |
| `_acc` | `acc` | 加速度噪声 |
| `_oms` | `oms` | 光学计量系统噪声 |
| `_clock` | `clock` | 时钟噪声（当前为 0） |
| `_sci` | `sci` | 科学相位计输出 |
| `_tes` | `tes` | 测试相位计输出 |
| `_ref` | `ref` | 参考相位计输出 |
| `_xi` | `xi` | ξ 信号（中间 TDI 量） |
| `_eta` | `eta` | η 信号（TDI 组合输入） |

#### 关键方法

- **`set_noise(parameters, noise_type)`**：更新指定噪声类型并刷新插值器。
- **`update_itfmeter()`**：刷新 `sci`、`tes`、`ref` 三个相位计插值器。
- **`update_eta()`**：刷新 η 信号插值器。
- **`update_laser()`**：刷新激光噪声插值器。

**注意**：所有插值器均采用高阶样条（`kind=31`），边界外返回 `1e-99`（近似为零）。

---

### 2.6 `TJ_lock.py` — 激光锁定模块

**功能**：实现臂锁定（Arm-locking）算法，利用臂间延迟反馈抑制激光频率噪声。

#### 类 `lock`

**构造参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `orbits` | — | 轨道对象 |
| `obs` | — | 光学平台列表 |
| `gws` | — | GW 对象 |
| `lock_type` | `'single'` | 锁定类型：`'single'` / `'dual'` / `'common'` |
| `method` | `'trapezoidal'` | 单臂积分方法：`'backward'` / `'trapezoidal'` |
| `arm_delay` | 20.0 | 单臂延迟（s） |
| `arm1_delay` | 20.1 | 双臂锁定较长延迟（s） |
| `arm2_delay` | 19.9 | 双臂锁定较短延迟（s） |
| `gfactor` | 10000.0 | 环路增益 |
| `afactor` | 100.0 | 比例增益因子 |
| `w0` | 2π×0.4 | 双臂锁定拐角频率（rad/s） |
| `lowpass_cutoff` | 1.0 | 低通滤波截止（Hz） |
| `lowpass_order` | 6 | Butterworth 低通滤波器阶数 |

#### 锁定算法

1. **单臂锁定（Single-arm）**：
   - 后向欧拉法（`backward`）：使用后向差分近似积分
   - 梯形法（`trapezoidal`）：使用梯形近似积分

2. **双臂锁定（Dual-arm）**：
   - 双线性（Tustin）变换离散化控制器
   - 利用两臂不同延迟的共同模抑制
   - 高阶 IIR 滤波器系数（4 阶状态）

3. **公共臂锁定（Common-arm）**：
   - 使用后向前规则离散化
   - 两臂反馈组合

#### 关键方法

- **`run()`**：执行锁定算法并自动更新所有光学平台对象。
  - 检测是否有激光噪声，无则跳过。
  - 将锁定后的激光传播到 `obs[-1]`（主平台）和 `_phase_lock()` 分配至其他平台。
- **`_phase_lock()`**：N1 相位锁定方案——利用延迟的激光信号和 GW 信号模拟跨臂激光传播。

---

### 2.7 `TJ_synthesis.py` — 信号合成模块

**功能**：综合激光噪声、GW 信号、加速度噪声和 OMS 噪声，生成相位计输出和 η 信号。

#### 类 `synthesis`

**构造参数**：

| 参数 | 说明 |
|---|---|
| `orbits` | 轨道对象（提供延迟函数） |
| `obs` | 光学平台列表 `[None, ob1, ob2, ob3, ob4, ob5, ob6]` |
| `gws` | GW 对象 |
| `delay_level` | `'1'`（柔性臂长）或 `'0'`（常数臂长） |

#### 合成流程

**第一阶段：相位计输出合成**

对每颗光学平台（共 6 个，索引 1–6 和 -1, -2, -3），按下列公式计算：

```
sci = [传入激光 (延迟)] - [本地激光] + [GW] + [OMS]
tes = [相邻本地激光] - [本地激光] - 2×[加速度噪声]
ref = [相邻本地激光] - [本地激光]
```

所有传入激光都经过轨道延迟（`delay_level='1'` 使用时变距离，`'0'` 用常数距离）。

**第二阶段：η 信号合成**

对每个光学台，组合相位计输出得到 TDI 输入：

```
η = sci + (tes - ref)/2
    + [相邻台tes(延迟) - 相邻台ref(延迟)]/2
    ± [本地ref - 相邻ref]/2
```

**性能优化**：
- 预计算 6 个延迟时间数组以避免重复计算（约 3 倍加速）
- 读取信号配置开关（`hasLaser`、`hasAcc`、`hasOms`、`hasGW`），跳过不必要的计算

---

### 2.8 `TJ_tdi.py` — TDI 组合模块

**功能**：实现时延干涉测量（TDI）组合，通过组合多臂延迟测量信号来消除激光频率噪声。

#### TDI 通道格式

每个 TDI 通道定义为 `[eta_label, arm_delays, sign]` 三元组的列表：

- `eta_label`：光学平台标识（`'1'`, `'2'`, `'3'`, `'1p'`, `'2p'`, `'3p'`）
- `arm_delays`：需遍历的臂列表（如 `[2, -2]` 表示先臂 2 再臂 -2）
- `sign`：`'+'` 加法、`'-'` 减法

#### 预定义 TDI 通道

| 通道 | 类型 | 说明 |
|---|---|---|
| `X1`, `Y1`, `Z1` | 第一代 Michelson | 基础激光噪声消除组合（8 项） |
| `X2`, `Y2`, `Z2` | 第二代 Michelson | 更高精度消除（16 项） |
| `a1`, `b1`, `g1` | 第一代 Sagnac | 零通道/监视通道（6 项） |
| `a15`, `b15`, `g15` | 增强 Sagnac | （12 项） |
| `a2`, `b2`, `g2` | 第二代 Sagnac | （16 项） |
| `z1`, `z15`, `z2` | 对称 Sagnac |
| `U1`, `V1`, `W1` | 备用 Michelson |
| `E1`, `F1`, `G1` | 备用组合 |
| `P1`, `Q1`, `R1` | 备用组合 |
| `PL4L1`–`PL4L3` | Pipeline 组合 | `E1 + P1` |

**通道生成规则**：
- `Y1 = cycle_tdi(X1)`、`Z1 = cycle_tdi(Y1)`——通过循环轮换生成
- `X2` 由 `X1` 的四个子项通过高阶延迟构造而成

#### 核心函数

- **`delay_tdi(delay_op, tdi_channel)`**：向现有 TDI 通道追加延迟操作。
- **`cycle_tdi(tdi_channel)`**：循环轮换星座索引生成等效通道。

#### 类 `tdi`

**构造参数**：

| 参数 | 说明 |
|---|---|
| `orbits` | 轨道对象（提供延迟函数） |
| `obs` | 光学平台列表 |

**关键方法**：

- **`run(TDI_channel=X1, delay_level='1')`**：计算 TDI 组合输出。
  - 对通道中的每项，延迟 η 信号并加/减贡献。
  - 使用高阶插值（k=31）保证延迟后精度。

**数学原理**：

```
TDI = Σᵢ (±1) × ηᵢ(t - Lⱼ(t) - Lₖ(t - Lⱼ) - ...)
```

通过构造使激光噪声项抵消，同时保留 GW 信号。

---

### 2.9 `TJ_Triangle.py` — 三角形星座流水线封装

**功能**：提供统一的高层接口，封装从轨道到 TDI 输出的完整仿真流水线。

#### 类 `Triangle`

**构造参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `t_start` | 0.0 | 起始时间（s） |
| `t_end` | 40000.0 | 结束时间（s） |
| `tri_arm` | 10.0 | 臂长（光秒） |
| `orbit_type` | `'heliocentric'` | 轨道类型 |
| `fsample_ob` | 5.0 | 光学平台采样率（Hz） |
| `fsample_gw` | 0.1 | GW 采样率（Hz） |

#### 关键方法

**GW 源管理**：

- **`add_gw_source(fgw, strain, beta, lamda, psi)`**：添加单个单色 GW 源。
- **`add_gw_frequency_series(frequencies, strain, ...)`**：添加频率序列源（频率扫描）。
- **`set_gw_sources(source_list)`**：从字典列表批量设置源。
- **`create_gw_object(hasGW=True)`**：创建 GW 对象。

**光学平台**：

- **`setup_optical_benches(hasLaser, hasAcc, hasOms, LNamp)`**：创建 6 个光学平台 `[None, ob1..ob6]`。

**激光锁定与合成**：

- **`apply_laser_lock(lock_type, ...)`**：应用臂锁定。
- **`synthesize_signals(delay_level)`**：合成信号。

**TDI 计算**：

- **`initialize_tdi()`**：初始化 TDI 对象。
- **`run_tdi(TDI_channel, delay_level, store_name)`**：运行 TDI 组合并存储结果。
- **`get_tdi_result(name)`**：按名称检索 TDI 结果。

**全流水线**：

- **`run_full_pipeline(delay_level, lock_type, tdi_channels)`**：一键执行完整流水线（GW 创建 → 光学平台 → 锁定 → 合成 → TDI）。

**其他**：

- **`get_laser_data()`**：返回自由和锁定激光噪声数据 `(free, locked, tarray, fsample)`。
- **`summary()`**：打印当前配置和状态的摘要。

#### 使用示例

```python
from TJ_Triangle import Triangle
import TJ_tdi

# 基础用法
tri = Triangle(t_start=0, t_end=40000, tri_arm=10.0)
tri.add_gw_source(fgw=0.007, strain=1e-23, beta=0, lamda=0, psi=0)
tri.setup_optical_benches()
tri.apply_laser_lock(lock_type='dual')
tri.synthesize_signals(delay_level='1')
X1 = tri.run_tdi(TJ_tdi.X1, delay_level='1')

# 频率扫描
freqs = np.logspace(-4, -1, 30)
tri.add_gw_frequency_series(freqs, strain=1e-23)
tri.run_full_pipeline()  # 一键执行全部

# 查看状态
tri.summary()
```

---

### 2.10 `TJ_freplan.py` — 频率规划模块（待实现）

**当前状态**：仅包含占位注释 `# The module should implement frequency planning`，功能尚未实现。

**预期功能**：实现观测频带的频率规划策略。

---

### 2.11 `TJ_dashboard.py` — 图形化桌面控制台

**功能**：基于 Tkinter + Matplotlib 的图形用户界面，用于配置、运行和检查仿真结果。

#### 主要特性

- **仿真配置面板**（左侧可滚动）：时间范围、臂长、轨道类型、采样率、随机种子
- **GW 源管理**：通过 Treeview 表格增/删/改 GW 源
- **仪器噪声开关**：激光噪声、加速度噪声、OMS 噪声独立启停
- **激光锁定选择**：单臂/双臂/公共臂锁定模式
- **TDI 通道选择**：X1/X2/Y1/Z1 复选框
- **流水线进度条**：6 阶段可视化（Orbit → GW → Benches → Locking → Synthesis → TDI）
- **四个结果标签页**：
  1. **Spectrum**：ASD 频谱图（对数坐标），支持手动调整范围
  2. **Time series**：时域波形窗口，支持步进/播放
  3. **Orbit geometry**：3D 星座几何图（带航天器到质心的偏移）
  4. **Run log**：带时间戳的运行日志
- **数据导出**：PNG/PDF 图表导出，NPZ 数值数据导出
- **后台线程运行**：仿真在独立线程中运行，避免 GUI 阻塞

#### 关键类

**`RunConfig`**：数据类，存储从 UI 收集并验证的仿真参数。

**`DashboardApp(Tk)`**：主应用窗口类，管理：
- UI 构造与样式配置
- 仿真线程的启动/取消
- 事件队列（`events`）的轮询与结果渲染
- 可视化更新和动画播放

**`QueueWriter`**：IO 重定向类，将仿真代码的 `print` 输出捕获到日志面板。

#### 启动方式

```bash
# 命令行启动
PYTHONPATH=src python -m TJ_dashboard

# 或通过 pyproject.toml 定义的入口点
quicktdi-dashboard
```

---

## 3. 代码约定

- **模块命名**：所有模块以 `TJ_` 前缀命名
- **类命名**：`PascalCase`（如 `Triangle`、`orbit`）
- **函数命名**：`snake_case`（如 `sc_pos_analytic`）
- **常量命名**：`UPPER_SNAKE_CASE`（如 `SPEED_OF_LIGHT`）
- **缩进**：4 空格
- **类型注解**：使用 `numpy.typing.NDArray`、`typing` 模块注解
- **文档字符串**：NumPy 风格（`Parameters`、`Returns`、`Examples`、`Notes` 段）
- **向后兼容**：保留旧的小写别名（如 `armRec = ARM_REC`）以兼容已有代码
- **性能优化**：多处核心循环已向量化（利用 NumPy 批量操作替代逐元素 Python 循环）

---

## 4. 光学平台编号约定

航天器编号为 1、2、3，构成等边三角形星座。光学平台（OB）按以下方式索引：

| 列表索引 | 含义 | 所在航天器 | 对应臂 |
|---|---|---|---|
| `obs[1]` | OB1 | SC1 | 臂 1（接收） |
| `obs[2]` | OB2 | SC2 | 臂 2（接收） |
| `obs[3]` | OB3 | SC3 | 臂 3（接收） |
| `obs[-3]` = `obs[4]` | OB3' | SC2 | 臂 -1（反向发送） |
| `obs[-2]` = `obs[5]` | OB2' | SC3 | 臂 -2（反向发送） |
| `obs[-1]` = `obs[6]` | OB1' | SC1 | 臂 -3（反向发送，主激光器） |

**臂编号规则**：
- 正向臂 (1, 2, 3)：顺时针方向传输
- 反向臂 (-1, -2, -3)：逆时针方向传输
- 臂 1：SC3 → SC2，臂 2：SC1 → SC3，臂 3：SC2 → SC1

---

## 5. 测试与验证

- 主要验证工具：`test.ipynb`（Jupyter Notebook 交互式工作流）
- 快速冒烟测试：`PYTHONPATH=src python -c "from TJ_Triangle import Triangle; Triangle()"`
- 暂无自动化测试套件；若需添加测试，建议使用 `pytest`，文件放入 `tests/` 目录。
