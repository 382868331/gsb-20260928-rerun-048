# 几何光线追迹核心

这是一个教学用的最小 Python 几何光学模型。光线按三维位置和单位方向传播，每一步选择前进方向上最近的有效曲面交点；圆形孔径外的交点不参与追迹。

## 已包含

- 平面与球面曲面，球面半径采用沿光轴的符号约定：球心在顶点右侧为正。
- 圆形孔径、折射面、镜面和像面/探测器。
- 材料表支持标量折射率，也支持 `Material(..., spectral_indices={450: 1.5, 650: 1.48})` 这种逐波长验证响应。光谱材料只接受明确给出的精确波长；缺失响应不会使用邻近值插值或外推，而会保留为 `unknown_response` 分支。
- 使用 `Ray.spectral(..., wavelengths=[450, 650], fractions=[...])` 可把同一输入拆成多个互相独立的波长分量。每个分量保留自己的波长、方向、能量、分支路径与终止原因，分支根使用 `root[wavelength=450]` 这类标识。
- Snell 折射、镜面反射、全反射、非偏振 Fresnel 反射率、平面分光片和孔径光阑。
- reflectivity 在所有表面类型上含义统一：始终表示反射能量比例（与镜面一致）。分光面可用 transmissivity 显式设置前向透射比例；不设置时默认透射等于 1 - reflectivity，两者之和小于 1 时差额记为表面损耗。旧工程曾把分光面的 reflectivity 字段存成前向透射比例；加载这类无 convention 标记（或标记为 legacy-beamsplitter-reflectivity-is-transmission）的工程时会自动迁移为 reflectivity 与 transmissivity 两个物理量、发出警告并在 Project.migration_notes 中记录，原实验的前向/后向能量解释保持不变。
- 折射面与分光面把每束光分裂为透射和反射两条子光束，两条子光束都继续追迹；全反射只保留反射支。
- 每条终止子光束保留来源标签、分支编号、父分支和深度，事件带全局编号避免跨分支重复记账。
- `TraceBatch.detector_hits`、`hits_by_source`、`detector_summary` 按探测面、来源光线和波长汇总能量与落点；`power_by_wavelength` 与 `power_by_source_wavelength` 可从总量拆回各波长，汇总中的 branches 保留汇入同一探测区域的每条分支（来源、波长、分支编号、路径）。
- `TraceBatch.branch_energy_flow()` 按分支列出每条终止子光束的完整能量流向：来源、分支编号、父分支、逐次反射/透射/损耗事件（表面、功率、反射率/透射率）与终止状态，可从任意探测落点回溯整条光路。
- `TraceBatch.energy_ledger` 分别统计到达、表面吸收、孔径擦边（`grazing`）、未到达（`no_image`）以及达到事件/分支上限而停止（`max_events`、`max_branches`）的剩余能量；停止能量不计为吸收，并给出 `unaccounted` 闭合误差。
- `compare_detector_power()`、`compare_energy_ledger()` 与 `compare_wavelength_response()` 可比较修改材料波长响应前后的探测能量、材料损耗（`material_loss`）和孔径截断（`aperture_grazing`），所有差值同时保留按波长拆分。旧批次应先通过 `TraceBatch.save()` 存档，再用新的多波长输入追加批次；旧 JSON 和旧分支能量解释不会被重写。
- `TraceSettings.max_events`、`max_branches`、`min_branch_weight` 共同约束反复反射和指数分裂。
- `Project` 与 `TraceBatch` 可保存/读取 JSON，便于工程和追迹批次重开。
 保存工程后用同样的输入光线重新追迹，会得到相同的分支路径、终止原因、能量和落点。

## 偏振通道与能量账

- 使用 `Ray.polarized(..., angles=[0, 90], fractions=[...])` 可把同一输入拆成多个互不相干的线偏振分量（角度在垂直于传播方向的参考平面内度量）。各分量按份额分配输入能量，总和恒等于输入而绝不会复制成两份完整输入；每个分量保留自己的偏振角、三维电场矢量、能量、分支路径与终止原因，分支根使用 `root[pol=90]` 标识，并可与光谱分量组合成 `root[wavelength=450][pol=90]`。偏振分量在折射面按 s/p 菲涅尔振幅系数分别更新电场方向与能量，镜面与全反射按几何反射更新电场。
- 分光面可用 `polarization_reflectivity={"s": ..., "p": ...}` 与 `polarization_transmissivity` 声明逐偏振轴响应；声明了偏振表却缺少某轴的已验证响应时，分支以 `unknown_response` 终止并记录 `unknown_polarization_response` 事件，不会拿无偏振标量冒充已知值。未声明偏振表时标量 reflectivity/transmissivity 对两个偏振轴一致生效。
- 探测面可用 `analyzer_angle_deg` 设置分析方向：透过能量按电场在分析轴上的投影计算，未透过部分记为 `analyzer_rejected`，与表面吸收（`absorbed`/`material_loss`）、体损耗（`bulk_loss`）、孔径擦边（`grazing`）分列，不混入无法追溯的差额。材料可用 `bulk_attenuation` 或逐波长的 `spectral_bulk_attenuation` 声明体损耗系数，缺失的波长响应同样保留为 `unknown_response`。
- `compare_polarization_response()` 比较旋转输入偏振或分析方向前后的探测能量（含 `power_by_polarization` 拆分）与完整能量账差值；`detector_summary` 的 `power_by_polarization` 只是新增一种汇总视角，不改变既有批次的数值结果。旧的无偏振工程与批次 JSON 缺少偏振字段，重开后仍按原来的无偏振解释（s/p 各半）计算，能量账保持不变。

## 简化假设

线偏振分量传播与材料体损耗已按上一节实现；下列其余简化仍然适用。

当前模型只描述理想化几何光线：表面无限薄且局部光滑，材料使用各向同性标量折射率，透镜厚度由两个表面之间的传播隐式给出。模型不含真实制造所需的像差拟合、镀膜、公差、散射、衍射、非球面或精确孔径装配；偏振只覆盖线偏振分量的相干分解，不含相位延迟与旋光。

因此它适合检查方向、交点、全反射和能量账，不应宣称结果可直接用于制造镜头。成像质量分析可在后续批次中基于落点、焦斑尺寸、RMS 弥散和光路数据扩展。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## 最小示例

```python
from raytrace import Material, MaterialLibrary, Project, Ray, CircularAperture, PlaneSurface, SphericalSurface

materials = MaterialLibrary()
materials.add(Material("glass", 1.5))

project = Project(
    "biconvex",
    [
        SphericalSurface("front", (0, 0, 0), (1, 0, 0), radius=10, kind="refract",
                         aperture=CircularAperture(2), back_material="glass"),
        SphericalSurface("back", (1, 0, 0), (1, 0, 0), radius=-10, kind="refract",
                         aperture=CircularAperture(2), front_material="glass"),
        PlaneSurface("image", (5, 0, 0), (1, 0, 0), kind="detector",
                     aperture=CircularAperture(3)),
    ],
    materials,
)

batch = project.trace_rays([Ray((-1, 0.5, 0), (1, 0, 0))])
print(batch.status_summary(), batch.results[0].endpoint)
```
