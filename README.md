# Nonlinear Crystal SHG Calculator (非线性晶体二次谐波计算器)

## 简介 (Introduction)
这是一个基于 Python 和 Streamlit 开发的光学仿真工具，用于计算常见非线性晶体（如 BBO, LBO, CLBO, KDP 等）在二次谐波产生（SHG）过程中的关键参数。

## 主要功能 (Features)
- **多晶体支持**：内置 BBO, LBO, CLBO, KDP, DKDP 等常用晶体数据库。
- **相位匹配计算**：支持 Type I 和 Type II 相位匹配角的精确计算。
- **有效非线性系数 ($d_{eff}$)**：基于 IEEE 标准和 Kleinman 对称性计算不同切向下的 $d_{eff}$。
- **可视化交互**：提供交互式 Web 界面，实时查看参数变化。
- **可视化图表**：支持 3D 晶体切向示意图和折射率椭球绘制。

## 安装与运行 (Installation & Usage)

1. **克隆仓库**
   ```bash
   git clone [https://github.com/Hongxin-Chen/Nonlinear_Crystal_Simulation_of_SHG.git](https://github.com/Hongxin-Chen/Nonlinear_Crystal_Simulation_of_SHG.git)
   cd Nonlinear_Crystal_Simulation_of_SHG
