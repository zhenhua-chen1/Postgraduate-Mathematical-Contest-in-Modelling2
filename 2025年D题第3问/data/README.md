# 数据说明

`main.py` 的所有输入均来自本目录，不读取项目外路径。

仓库自带的紧凑输入结构为：

```text
data/
├── manifest.json
├── model_c_validation_1km_50m.npz
└── nwp/
    ├── grid.npz
    └── nwp_regular_YYYYMMDDHHMM.npz
```

- `model_c_validation_1km_50m.npz`：02:00--05:00 的模型 c 融合场、覆盖度与不确定度；
- `nwp/grid.npz`：1 km网格经纬度、地形、有效区域掩膜和0--2000 m高度层；
- `nwp/nwp_regular_*.npz`：12个时次的位温、东西风、南北风和垂直风；
- `manifest.json`：紧凑数据清单，不包含原始资料的本机路径。

紧凑数据合计约139 MB，单个文件均小于100 MB，可直接用于 GitHub。它们保留了第三问建模所需的信息，但不重复提交约9.5 GB的天气雷达原始 CSV。

程序中仍保留原始资料解析函数，便于审阅模型 c 的构造细节；GitHub 版的正常复现实验只需要上述紧凑文件。
