# 几何光线追迹核心

这是一个教学用的最小 Python 几何光学模型。光线按三维位置和单位方向传播，每一步选择前进方向上最近的有效曲面交点；圆形孔径外的交点不参与追迹。

## 已包含

- 平面与球面曲面，球面半径采用沿光轴的符号约定：球心在顶点右侧为正。
- 圆形孔径、折射面、镜面和像面/探测器。
- 材料表与标量折射率；`Ray.wavelength_nm` 已预留，后续可替换为色散模型。
- Snell 折射、镜面反射、全反射和非偏振 Fresnel 反射率。
- 每次相交保存位置、法线、方向、折射率、透射/反射/吸收能量和后续方向。
- `grazing`、`reached_image`、`no_image`、`max_events` 等光线状态均随批次保留。
- `Project` 与 `TraceBatch` 可保存/读取 JSON，便于工程和追迹批次重开。

## 简化假设

当前模型只描述理想化几何光线：表面无限薄且局部光滑，材料使用各向同性标量折射率，透镜厚度由两个表面之间的传播隐式给出。模型不含真实制造所需的像差拟合、色散、镀膜、公差、散射、偏振传播、衍射、体积吸收、非球面或精确孔径装配。

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
