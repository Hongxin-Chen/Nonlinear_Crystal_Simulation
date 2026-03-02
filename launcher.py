"""
非线性晶体二次谐波(SHG)模拟器
功能：相位匹配计算、3D可视化、接受带宽分析
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from simulation import Solver
from configuration import SimulationConfig  

# ============================================================================
# 页面配置与样式
# ============================================================================
st.set_page_config(
    page_title="非线性晶体模拟 V2.0",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
        h1 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("非线性晶体模拟器")

with st.expander("🎉 我们很高兴地发布 **v2.0 版本**！", expanded=False):
    st.markdown("""
    * ✨ **支持和频 (SFG) 模拟**
        > 现在可以像模拟倍频 (SHG) 一样，精确计算 SFG 过程中的 $d_{eff}$、走离角等关键参数。

    * 🔥 **全新温度匹配模块**
        > 引入温度相位匹配 (TPM) 算法。支持通过调节温度寻找最佳匹配条件，完美适配非临界相位匹配 (NCPM) 等实验场景。

    * 📚 **多来源 Sellmeier 方程**
        > 支持为同一晶体切换不同文献来源的折射率方程（如 Kato, Eimerl 等），让仿真更贴近真实实验数据。
        > * *Roadmap: 我们将持续维护并扩充每个晶体的 Sellmeier 方程库，致力于提供最全面准确的仿真数据。*        
                 
    <br>
                
    **为了实现从“理论计算”到“工程应用”的美好愿景，以下功能正在开发中**：

    <br>          

    * ⚡ **引入实际转换效率算法**
        > 将支持输入具体的激光参数（如**输入功率**、**发散角**、**$M^2$ 因子**、**光斑大小**等），通过耦合波方程数值解，直接预测实际工程中的功率输出与转换效率。

    * 🌊 **光束演化与热效应模拟**
        > 上海研发团队正在开发全物理场仿真程序。
        > * 将综合考量**热透镜效应**、**光斑恶化**及**衍射效应**等真实物理条件。
        > * 结合本程序提供的核心晶体参数，实现对激光器内部光束演化行为的深度洞察与全链路仿真。
    """, unsafe_allow_html=True)
# ============================================================================
# 晶体类型定义与参数输入
# ============================================================================

# 晶体类型字典：单轴/双轴分类
CRYSTAL_TYPES = {
    'LBO': 'biaxial',    # 双轴
    'KTP': 'biaxial',    # 双轴
    'BBO': 'uniaxial',   # 单轴
    'CLBO': 'uniaxial',  # 单轴
    'KDP': 'uniaxial',   # 单轴
    'DKDP': 'uniaxial'   # 单轴
}

# Sellmeier方程信息字典 - 根据晶体和来源组织
SELLMEIER_INFO = {
    'LBO': {
        '福晶': """LBO (双轴晶体) - 福晶科技

n_x² = 2.454140 + 0.011249/(λ² - 0.011350) - 0.014591λ² - 6.60×10⁻⁵λ⁴

n_y² = 2.539070 + 0.012711/(λ² - 0.012523) - 0.018540λ² + 2.00×10⁻⁴λ⁴

n_z² = 2.586179 + 0.013099/(λ² - 0.011893) - 0.017968λ² - 2.26×10⁻⁴λ⁴

温度系数：
dn_x/dT = -9.3×10⁻⁶/°C
dn_y/dT = -13.6×10⁻⁶/°C
dn_z/dT = -(6.3 + 2.1λ)×10⁻⁶/°C

(λ单位：μm)""",
        'Thorlabs': """LBO (双轴晶体) - Thorlabs

n_x² = 2.4542 + 0.01125/(λ² - 0.01135) - 0.01388λ²
温度项：Δn_x = (ΔT + 0.02913ΔT²) × (-3.76λ + 2.30) × 10⁻⁶

n_y² = 2.5390 + 0.01277/(λ² - 0.01189) - 0.01849λ² + 4.3025×10⁻⁵λ⁴ - 2.9131×10⁻⁵λ⁶
温度项：Δn_y = (ΔT - 0.0003289ΔT²) × (6.01λ - 19.40) × 10⁻⁶

n_z² = 2.5865 + 0.01310/(λ² - 0.01223) - 0.01862λ² + 4.5778×10⁻⁵λ⁴ - 3.2526×10⁻⁵λ⁶
温度项：Δn_z = (ΔT - 0.0007449ΔT²) × (1.50λ - 9.70) × 10⁻⁶

(λ单位：μm, ΔT = T - 20°C)"""
    },
    
    'BBO': {
        '默认': """BBO (单轴晶体, n_o = n_x = n_y, n_e = n_z)

n_o² = 1 + 0.90291λ²/(λ² - 0.003926) + 0.83155λ²/(λ² - 0.018786) + 0.76536λ²/(λ² - 60.01)

n_e² = 1 + 1.151075λ²/(λ² - 0.007142) + 0.21803λ²/(λ² - 0.02259) + 0.656λ²/(λ² - 263)

温度系数：dn_o/dT = -16.6×10⁻⁶/°C, dn_e/dT = -9.3×10⁻⁶/°C

(λ单位：μm)"""
    },
    
    'CLBO': {
        '福晶': """CLBO (单轴晶体, n_o = n_x = n_y, n_e = n_z) - 福晶科技

n_o² = 2.2104 + 0.01018/(λ² - 0.01424) - 0.01258λ²

n_e² = 2.0588 + 0.00838/(λ² - 0.01363) - 0.00607λ²

温度系数：
dn_o/dT = (-12.48 - 0.328/λ) × 10⁻⁶ (°C⁻¹)
dn_e/dT = (-8.36 + 0.047/λ - 0.039/λ² + 0.014/λ³) × 10⁻⁶ (°C⁻¹)

(λ单位：μm, 适用范围：0.2128-1.3382 μm，参考温度：20°C)

参考文献：
Umemura, N., et al. "New data on the phase-matching properties of CsLiB6O10." Advanced Solid State Lasers. Optica Publishing Group, 1999.""",
        'OXIDE': """CLBO (单轴晶体, n_o = n_x = n_y, n_e = n_z) - OXIDE

n_o² = 2.2145 + 0.00890/(λ² - 0.02051) - 0.01413λ²

n_e² = 2.0588 + 0.00866/(λ² - 0.01202) - 0.00607λ²

温度系数：
dn_o/dT = (-1.04λ² + 0.35λ - 12.91) × 10⁻⁶ (°C⁻¹)
dn_e/dT = (3.31λ² - 2.43λ - 8.40) × 10⁻⁶ (°C⁻¹)

(λ单位：μm)

参考文献：
Nobuhiro Umemura and Kiyoshi Kato, "Ultraviolet generation tunable to 0.185 µm in CsLiB6O10," Appl. Opt. 36, 6794-6796 (1997)"""
    },
    
    'KTP': {
        '默认': """KTP (双轴晶体, 福晶科技数据)

n_x² = 3.0065 + 0.03901/(λ² - 0.04251) - 0.01327λ²

n_y² = 3.0333 + 0.04154/(λ² - 0.04547) - 0.01408λ²

n_z² = 3.3134 + 0.05694/(λ² - 0.05658) - 0.01682λ²

温度系数：dn_x/dT = 1.1×10⁻⁵/°C, dn_y/dT = 1.3×10⁻⁵/°C, dn_z/dT = 1.6×10⁻⁵/°C

(λ单位：μm)"""
    },
    
    'KDP': {
        '默认': """KDP (单轴晶体, n_o = n_x = n_y, n_e = n_z)

n_o² = 2.259276 + 0.01008956/(λ² - 0.012942625) + 13.00522λ²/(λ² - 400)

n_e² = 2.132668 + 0.008637494/(λ² - 0.012281043) + 3.2279924λ²/(λ² - 400)

(λ单位：μm, 不含温度项)"""
    },
    
    'DKDP': {
        '默认': """DKDP (单轴晶体, n_o = n_x = n_y, n_e = n_z)

n_o² = 1.9575544 + 0.2901391λ²/(λ² - 0.0281399) - 0.02824391λ² + 0.004977826λ⁴

n_e² = 1.5057799 + 0.6276034λ²/(λ² - 0.0131558) - 0.01054063λ² + 0.002243821λ⁴

(λ单位：μm, 不含温度项)"""
    }
}

# 侧边栏：参数输入
with st.sidebar:
    st.header("仿真参数设置")
    
    # 基础参数
    crystal_name = st.selectbox(
        "晶体类型", 
        ["LBO", "BBO", "CLBO","KTP","KDP","DKDP"], 
        key="crystal_selectbox",
        help="晶体的Sellmeier方程不同来源有所差异"
    )
    
    # 根据晶体类型确定可选的Sellmeier方程来源
    if crystal_name == "CLBO":
        source_options = ["福晶", "OXIDE"]
        default_index = 0
    elif crystal_name == "LBO":
        source_options = ["福晶", "Thorlabs"]
        default_index = 0
    else:
        source_options = ["默认"]
        default_index = 0
    
    # Sellmeier方程来源选择
    sellmeier_source = st.selectbox(
        "Sellmeier方程来源",
        source_options,
        index=default_index,
        key="sellmeier_source",
        help="不同文献或厂商提供的Sellmeier方程系数可能有所不同"
    )
    
    # 动态显示当前选择晶体的Sellmeier方程
    # 从字典中获取对应来源的方程信息
    sellmeier_text = SELLMEIER_INFO.get(crystal_name, {}).get(sellmeier_source, "暂无该来源的方程信息")
    with st.expander(f"📖 {crystal_name} 的 Sellmeier 方程 ({sellmeier_source})", expanded=False):
        st.text(sellmeier_text)
    
    st.divider()
    
    # 非线性过程选择
    process_type = st.selectbox(
        "非线性过程",
        ["SHG (倍频)", "SFG (和频)"],
        key="process_type",
        help="SHG: ω+ω→2ω | SFG: ω₁+ω₂→ω₃"
    )
    
    # 根据过程类型显示不同的波长输入
    if "SHG" in process_type:
        wavelength_nm = st.number_input("输入基频λ (nm)", value=1064.0, step=0.1, help="输入波长")
        wavelength2_nm = None
        output_wavelength = wavelength_nm / 2
        st.info(f"✨ 输出波长: **{output_wavelength:.2f} nm**")
        process_type_code = 'SHG'
    else:  # SFG
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            wavelength_nm = st.number_input("输入λ₁ (nm)", value=1064.0, step=0.1, help="第一个输入波长")
        with col_w2:
            wavelength2_nm = st.number_input("输入λ₂ (nm)", value=532.0, step=0.1, help="第二个输入波长")
        
        output_wavelength = 1 / (1/wavelength_nm + 1/wavelength2_nm)
        st.success(f"✨ 输出波长: **{output_wavelength:.2f} nm**")
        process_type_code = 'SFG'
    
    # 匹配方式选择
    matching_method = st.selectbox(
        "匹配方式",
        ["角度匹配", "温度匹配"],
        key="matching_method",
        help="角度匹配: 通过调节光传播方向实现相位匹配 | 温度匹配: 在固定传播轴方向下通过调节温度实现相位匹配"
    )
    
    st.divider()
    
    # 根据匹配方式显示不同的参数设置
    if matching_method == "角度匹配":
        # 角度匹配参数
        temperature = st.number_input("温度 (°C)", value=20.0, step=0.1, help="晶体工作温度，通常室温20°C")
        
        # 根据晶体类型配置平面和角度
        crystal_type = CRYSTAL_TYPES[crystal_name]
        
        if crystal_type == 'uniaxial':
            # 单轴晶体：平面锁定XZ，φ角可调
            plane = "XZ"
            st.selectbox("k矢量所在平面", ["XZ"], index=0, disabled=True, 
                        help="单轴晶体对平面没有限制，这里默认XZ平面，不影响计算")
            phi = st.number_input("φ角 (度)", value=45.0, step=0.1, 
                                 help="单轴晶体的φ角，常用45°或90°")
        else:
            # 双轴晶体：平面可选，角度根据平面自动锁定，默认XY平面
            plane = st.selectbox("k矢量所在平面", ["XY", "YZ", "XZ"], index=0, 
                                help="双轴晶体可选择不同平面")
            
            if plane == "XY":
                phi = 90.0
                st.number_input("θ角 (度)", value=90.0, step=0.1, disabled=True, 
                               help="XY平面时θ角锁定为90°")
            else:  # YZ或XZ
                phi = 0.0
                st.number_input("φ角 (度)", value=0.0, step=0.1, disabled=True, 
                               help="YZ/XZ平面时φ角锁定为0°")
    else:
        # 温度匹配参数
        temperature = 20.0  # 温度匹配时，初始温度不重要，会扫描温度范围
        plane = "XZ"  # 默认平面
        phi = 0.0  # 默认角度
        
        # 温度匹配使用XYZ模式，不需要平面和角度设置
        st.info("💡 温度匹配模式：光沿晶体主轴传播，通过调节温度实现相位匹配")
        
        # 只让用户选择传播轴
        fixed_axis_sidebar = st.selectbox(
            "👉 请选择传播轴:",
            ["X", "Y", "Z"],
            key='fixed_axis_temp_match',
            help="选择光传播的晶体主轴方向"
        )
        
        # 温度范围设置
        st.markdown("**温度范围**", help="选择求解的温度范围")
        col_temp1, col_temp2 = st.columns(2)
        with col_temp1:
            temp_min_sidebar = st.number_input("最低温度 (°C)", value=20.0, step=1.0, key="temp_min_sidebar")
        with col_temp2:
            temp_max_sidebar = st.number_input("最高温度 (°C)", value=200.0, step=1.0, key="temp_max_sidebar")
        
        temp_step_sidebar = st.number_input("温度步长 (°C)", value=0.1, min_value=0.01, max_value=1.0, step=0.01, key="temp_step_sidebar")

    st.divider()

# ============================================================================
# 初始化计算核心
# ============================================================================

try:
    user_config = SimulationConfig(
        crystal_name=crystal_name, 
        wavelength=wavelength_nm, 
        temperature=temperature, 
        plane=plane,
        process_type=process_type_code,
        wavelength2=wavelength2_nm,
        sellmeier_source=sellmeier_source
    ) 
    simulation = Solver(user_config)

except Exception as e:
    st.error(f"初始化失败: {e}")
    st.stop()

# ============================================================================
# 运行计算
# ============================================================================

# 使用 Session State 管理状态，防止点击 Tab 时数据丢失
if 'has_run' not in st.session_state:
    st.session_state.has_run = False

# 运行按钮
if st.button("运行", type="primary", use_container_width=True):

    # 每次运行前清除旧结果
    keys_to_clear = ['res_angle_fig', 'res_wave_fig', 'res_temp_fig', 'temp_match_result', 
                     'ncpm_res_temp_fig', 'ncpm_res_wl_fig', 'ncpm_res_ang_results', 'ncpm_res_ang_planes',
                     'all_bandwidths_calculated', 'ncpm_all_calculated']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    with st.spinner("正在求解相位匹配方程..."):
        try:
            if matching_method == "角度匹配":
                # 角度匹配计算
                # 1. 计算临界角
                st.session_state.theta_dict = simulation.criticalangle()
                # 2. 计算走离角
                st.session_state.walkoff_dict = simulation.walkoff_angle(st.session_state.theta_dict, phi)
                # 3. 计算有效非线性系数
                st.session_state.d_eff_dict = simulation.d_eff(st.session_state.theta_dict, phi)
                # 标记运行完成
                st.session_state.has_run = True
                st.session_state.matching_method_run = "角度匹配"
            else:
                # 温度匹配计算 - 对所选传播轴的所有模式进行计算
                # 生成该传播轴的所有可能模式
                all_modes_for_axis = []
                
                if process_type_code == 'SHG':
                    λω = f"{wavelength_nm:.0f}nm"
                    λ2ω = f"{wavelength_nm/2:.0f}nm"
                    
                    if fixed_axis_sidebar == 'X':
                        # X轴传播：可用偏振Y, Z
                        all_modes_for_axis = [
                            f"𝐘 ({λω}) + 𝐘 ({λω}) → 𝐙 ({λ2ω}) (Type I)",
                            f"𝐙 ({λω}) + 𝐙 ({λω}) → 𝐘 ({λ2ω}) (Type I)",
                            f"𝐘 ({λω}) + 𝐙 ({λω}) → 𝐙 ({λ2ω}) (Type II)",
                            f"𝐘 ({λω}) + 𝐙 ({λω}) → 𝐘 ({λ2ω}) (Type II)"
                        ]
                    elif fixed_axis_sidebar == 'Y':
                        # Y轴传播：可用偏振X, Z
                        all_modes_for_axis = [
                            f"𝐗 ({λω}) + 𝐗 ({λω}) → 𝐙 ({λ2ω}) (Type I)",
                            f"𝐙 ({λω}) + 𝐙 ({λω}) → 𝐗 ({λ2ω}) (Type I)",
                            f"𝐗 ({λω}) + 𝐙 ({λω}) → 𝐙 ({λ2ω}) (Type II)",
                            f"𝐗 ({λω}) + 𝐙 ({λω}) → 𝐗 ({λ2ω}) (Type II)"
                        ]
                    else:  # Z
                        # Z轴传播：可用偏振X, Y
                        all_modes_for_axis = [
                            f"𝐗 ({λω}) + 𝐗 ({λω}) → 𝐘 ({λ2ω}) (Type I)",
                            f"𝐘 ({λω}) + 𝐘 ({λω}) → 𝐗 ({λ2ω}) (Type I)",
                            f"𝐗 ({λω}) + 𝐘 ({λω}) → 𝐘 ({λ2ω}) (Type II)",
                            f"𝐗 ({λω}) + 𝐘 ({λω}) → 𝐗 ({λ2ω}) (Type II)"
                        ]
                else:  # SFG
                    λ1 = f"{wavelength_nm:.0f}nm"
                    λ2 = f"{wavelength2_nm:.0f}nm"
                    λout = f"{1/(1/wavelength_nm + 1/wavelength2_nm):.0f}nm"
                    
                    if fixed_axis_sidebar == 'X':
                        # X轴传播：可用偏振Y, Z
                        all_modes_for_axis = [
                            f"𝐘 ({λ1}) + 𝐘 ({λ2}) → 𝐙 ({λout}) (Type I)",
                            f"𝐙 ({λ1}) + 𝐙 ({λ2}) → 𝐘 ({λout}) (Type I)",
                            f"𝐘 ({λ1}) + 𝐙 ({λ2}) → 𝐙 ({λout}) (Type II)",
                            f"𝐙 ({λ1}) + 𝐘 ({λ2}) → 𝐙 ({λout}) (Type II)",
                            f"𝐘 ({λ1}) + 𝐙 ({λ2}) → 𝐘 ({λout}) (Type II)",
                            f"𝐙 ({λ1}) + 𝐘 ({λ2}) → 𝐘 ({λout}) (Type II)"
                        ]
                    elif fixed_axis_sidebar == 'Y':
                        # Y轴传播：可用偏振X, Z
                        all_modes_for_axis = [
                            f"𝐗 ({λ1}) + 𝐗 ({λ2}) → 𝐙 ({λout}) (Type I)",
                            f"𝐙 ({λ1}) + 𝐙 ({λ2}) → 𝐗 ({λout}) (Type I)",
                            f"𝐗 ({λ1}) + 𝐙 ({λ2}) → 𝐙 ({λout}) (Type II)",
                            f"𝐙 ({λ1}) + 𝐗 ({λ2}) → 𝐙 ({λout}) (Type II)",
                            f"𝐗 ({λ1}) + 𝐙 ({λ2}) → 𝐗 ({λout}) (Type II)",
                            f"𝐙 ({λ1}) + 𝐗 ({λ2}) → 𝐗 ({λout}) (Type II)"
                        ]
                    else:  # Z
                        # Z轴传播：可用偏振X, Y
                        all_modes_for_axis = [
                            f"𝐗 ({λ1}) + 𝐗 ({λ2}) → 𝐘 ({λout}) (Type I)",
                            f"𝐘 ({λ1}) + 𝐘 ({λ2}) → 𝐗 ({λout}) (Type I)",
                            f"𝐗 ({λ1}) + 𝐘 ({λ2}) → 𝐘 ({λout}) (Type II)",
                            f"𝐘 ({λ1}) + 𝐗 ({λ2}) → 𝐘 ({λout}) (Type II)",
                            f"𝐗 ({λ1}) + 𝐘 ({λ2}) → 𝐗 ({λout}) (Type II)",
                            f"𝐘 ({λ1}) + 𝐗 ({λ2}) → 𝐗 ({λout}) (Type II)"
                        ]
                
                # 对每个模式进行温度匹配计算
                temp_match_results = {}
                for mode in all_modes_for_axis:
                    try:
                        result = simulation.temperature_phase_matching(
                            mode, 
                            temperature_range=(temp_min_sidebar, temp_max_sidebar), 
                            temp_step=temp_step_sidebar, 
                            fixed_axis=fixed_axis_sidebar
                        )
                        temp_match_results[mode] = result
                    except Exception as e:
                        temp_match_results[mode] = {'error': str(e)}
                
                st.session_state['temp_match_results'] = temp_match_results
                st.session_state['fixed_axis_sidebar'] = fixed_axis_sidebar
                st.session_state['fixed_axis_sidebar'] = fixed_axis_sidebar
                st.session_state.has_run = True
                st.session_state.matching_method_run = "温度匹配"
            
            # 清除旧的3D图
            if '3d_fig' in st.session_state:
                del st.session_state['3d_fig']
        except Exception as e:
            st.error(f"计算过程出错: {e}")

# ============================================================================
# 结果展示区：三大模块
# 模块1: 相位匹配参数计算（临界角、走离角、有效非线性系数）
# 模块2: 3D折射率椭球示意图生成
# 模块3: 接受带宽分析（角度、波长、温度带宽）
# ============================================================================

if st.session_state.has_run:
    
    st.divider()
    
    # 根据匹配方式显示不同的结果
    if st.session_state.get('matching_method_run') == "角度匹配":
        # ============================================================================
        # 角度匹配结果展示
        # ============================================================================
        
        # ============================================================================
        # 模块1: 相位匹配参数计算
        # ============================================================================
        st.subheader("📊 1. 相位匹配参数计算")
    
        theta_dict = st.session_state.theta_dict
        walkoff_dict = st.session_state.walkoff_dict
        d_eff_dict = st.session_state.d_eff_dict
        
        # 准备表格数据
        table_data = []
        valid_modes = [] # 记录有效的模式，后面画图用
        
        for mode in theta_dict:
            angle = theta_dict[mode]
            # 判断是否有效 (不是 NaN)
            if not np.isnan(angle):
                valid_modes.append(mode)
                pm_angle_str = f"{angle:.4f}°"
                walkoff_str = walkoff_dict.get(mode, "N/A")
                d_eff_str = f"{d_eff_dict.get(mode, 'N/A'):.4f}" if mode in d_eff_dict else "N/A"
            else:
                pm_angle_str = "❌ 无解"
                walkoff_str = "-"
                d_eff_str = "-"
                
            table_data.append({
                "匹配模式": mode,
                "临界角": pm_angle_str,
                "走离角 [负值代表远离Z轴(XZ,YZ)或X轴(XY)]": walkoff_str,
                "有效非线性系数(pm/V)": d_eff_str
            })
        
        # 展示表格
        df = pd.DataFrame(table_data)
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "匹配模式": st.column_config.TextColumn(width="medium"),
                "临界角": st.column_config.TextColumn(width="small"),
                "走离角 [负值代表远离Z轴(XZ,YZ)或X轴(XY)]": st.column_config.TextColumn(width="large"),
                "有效非线性系数(pm/V)": st.column_config.TextColumn(width="medium"),

            }
        )
        
        # region  3d 图

        # ============================================================================
        # 模块2: 3D折射率椭球示意图
        # ============================================================================
        st.subheader("🎨 2. 折射率椭球与相位匹配示意图 (3D)")

        if not valid_modes:
            st.warning("当前没有有效的相位匹配模式，无法进行3D可视化。")
        else:
            # 用户选择显示选项
            col_sel, _ = st.columns([1, 2])
            with col_sel:
                target_mode_3d = st.selectbox("👉 请选择要可视化的模式:", valid_modes, key='mode_3d')

            # 生成3D图按钮
            if st.button("生成3D图", type="secondary", key="btn_3d"):
                # --- 创建3D折射率椭球示意图 (夸大视觉效果) ---
                fig = go.Figure()

            # region 1. 数据获取和缩放系数设置
            # 获取输入光1的折射率
            indices_w1 = user_config.get_indices(user_config.wavelength1_nm)
            n_x_w1 = indices_w1['n_x']
            n_y_w1 = indices_w1['n_y']
            n_z_w1 = indices_w1['n_z']
            
            # 获取输入光2的折射率（SFG需要）
            if user_config.process_type == 'SFG':
                indices_w2 = user_config.get_indices(user_config.wavelength2_nm)
                n_x_w2 = indices_w2['n_x']
                n_y_w2 = indices_w2['n_y']
                n_z_w2 = indices_w2['n_z']

            # 获取输出光的折射率
            indices_out = user_config.get_indices(user_config.wavelength_out_nm)
            n_x_out = indices_out['n_x']
            n_y_out = indices_out['n_y']
            n_z_out = indices_out['n_z']

            # === 使用真实折射率值，不进行缩放 ===
            scale_w1_x = n_x_w1
            scale_w1_y = n_y_w1
            scale_w1_z = n_z_w1
            
            if user_config.process_type == 'SFG':
                scale_w2_x = n_x_w2
                scale_w2_y = n_y_w2
                scale_w2_z = n_z_w2
            
            scale_out_x = n_x_out
            scale_out_y = n_y_out
            scale_out_z = n_z_out
            
            # 确定标签文本和颜色
            if user_config.process_type == 'SHG':
                input1_label = f'基频光 (ω) {user_config.wavelength1_nm:.1f}nm'
                output_label = f'倍频光 (2ω) {user_config.wavelength_out_nm:.1f}nm'
            else:  # SFG
                input1_label = f'输入光1 (ω₁) {user_config.wavelength1_nm:.1f}nm'
                input2_label = f'输入光2 (ω₂) {user_config.wavelength2_nm:.1f}nm'
                output_label = f'和频光 (ω₃) {user_config.wavelength_out_nm:.1f}nm'
                
                # 确定哪束光波长更短，用于颜色分配
                if user_config.wavelength2_nm < user_config.wavelength1_nm:
                    # λ2更短
                    short_wave_label = input2_label
                    long_wave_label = input1_label
                    short_wave_color_start = 'rgb(255, 215, 0)'  # 金黄色
                    short_wave_color_end = 'rgb(255, 235, 100)'
                    long_wave_color_start = 'rgb(255, 80, 80)'
                    long_wave_color_end = 'rgb(255, 150, 150)'
                    short_indices = (n_x_w2, n_y_w2, n_z_w2)
                    long_indices = (n_x_w1, n_y_w1, n_z_w1)
                else:
                    # λ1更短
                    short_wave_label = input1_label
                    long_wave_label = input2_label
                    short_wave_color_start = 'rgb(255, 215, 0)'  # 金黄色
                    short_wave_color_end = 'rgb(255, 235, 100)'
                    long_wave_color_start = 'rgb(255, 80, 80)'
                    long_wave_color_end = 'rgb(255, 150, 150)'
                    short_indices = (n_x_w1, n_y_w1, n_z_w1)
                    long_indices = (n_x_w2, n_y_w2, n_z_w2)
            # endregion

            # region 2. 生成折射率椭球
            # 创建球坐标系的网格 (theta: 0到π, phi: 0到2π)
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            
            # 生成输入光1折射率椭球的坐标
            x_w1 = scale_w1_x * np.outer(np.cos(u), np.sin(v))
            y_w1 = scale_w1_y * np.outer(np.sin(u), np.sin(v))
            z_w1 = scale_w1_z * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # SFG: 生成输入光2折射率椭球的坐标
            if user_config.process_type == 'SFG':
                x_w2 = scale_w2_x * np.outer(np.cos(u), np.sin(v))
                y_w2 = scale_w2_y * np.outer(np.sin(u), np.sin(v))
                z_w2 = scale_w2_z * np.outer(np.ones(np.size(u)), np.cos(v))

            # 生成输出光折射率椭球的坐标
            x_out = scale_out_x * np.outer(np.cos(u), np.sin(v))
            y_out = scale_out_y * np.outer(np.sin(u), np.sin(v))
            z_out = scale_out_z * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # region 3. 创建3D图
            fig = go.Figure()
            
            # SHG模式：红色输入光
            if user_config.process_type == 'SHG':
                fig.add_trace(go.Surface(
                    x=x_w1, y=y_w1, z=z_w1,
                    colorscale=[[0, 'rgb(255, 80, 80)'], [1, 'rgb(255, 150, 150)']],
                    showscale=False,
                    opacity=0.25,
                    name=input1_label,
                    hovertemplate=f'{input1_label}<br>n_x={n_x_w1:.4f}<br>n_y={n_y_w1:.4f}<br>n_z={n_z_w1:.4f}<extra></extra>',
                    contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                    hidesurface=False
                ))
            
            # SFG模式：添加两束输入光，短波长用黄色
            if user_config.process_type == 'SFG':
                # 短波长光（黄色）
                if user_config.wavelength1_nm < user_config.wavelength2_nm:
                    short_x, short_y, short_z = x_w1, y_w1, z_w1
                    long_x, long_y, long_z = x_w2, y_w2, z_w2
                else:
                    short_x, short_y, short_z = x_w2, y_w2, z_w2
                    long_x, long_y, long_z = x_w1, y_w1, z_w1
                
                # 添加短波长光椭球（黄色）
                fig.add_trace(go.Surface(
                    x=short_x, y=short_y, z=short_z,
                    colorscale=[[0, short_wave_color_start], [1, short_wave_color_end]],
                    showscale=False,
                    opacity=0.25,
                    name=short_wave_label,
                    hovertemplate=f'{short_wave_label}<br>n_x={short_indices[0]:.4f}<br>n_y={short_indices[1]:.4f}<br>n_z={short_indices[2]:.4f}<extra></extra>',
                    contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                    hidesurface=False
                ))
                
                # 添加长波长光椭球（红色）
                fig.add_trace(go.Surface(
                    x=long_x, y=long_y, z=long_z,
                    colorscale=[[0, long_wave_color_start], [1, long_wave_color_end]],
                    showscale=False,
                    opacity=0.25,
                    name=long_wave_label,
                    hovertemplate=f'{long_wave_label}<br>n_x={long_indices[0]:.4f}<br>n_y={long_indices[1]:.4f}<br>n_z={long_indices[2]:.4f}<extra></extra>',
                    contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                    hidesurface=False
                ))

            # 添加输出光椭球（蓝色）
            fig.add_trace(go.Surface(
                x=x_out, y=y_out, z=z_out,
                colorscale=[[0, 'rgb(50, 100, 255)'], [1, 'rgb(100, 150, 255)']],
                showscale=False,
                opacity=0.25,
                name=output_label,
                hovertemplate=f'{output_label}<br>n_x={n_x_out:.4f}<br>n_y={n_y_out:.4f}<br>n_z={n_z_out:.4f}<extra></extra>',
                contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                hidesurface=False
            ))
            # endregion

            # region 3. 添加坐标轴
            # 添加坐标轴参考线
            axis_length = 3.5  # 固定长度用于示意图
            
            # X轴 (红色)
            fig.add_trace(go.Scatter3d(
                x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
                mode='lines',
                line=dict(color='red', width=4),
                name='X轴',
                showlegend=True
            ))
            
            # X轴标注
            fig.add_trace(go.Scatter3d(
                x=[axis_length * 1.15], y=[0], z=[0],
                mode='text',
                text=['X'],
                textfont=dict(size=18, color='red', family='Arial Black'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Y轴 (绿色)
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
                mode='lines',
                line=dict(color='green', width=4),
                name='Y轴',
                showlegend=True
            ))
            
            # Y轴标注
            fig.add_trace(go.Scatter3d(
                x=[0], y=[axis_length * 1.15], z=[0],
                mode='text',
                text=['Y'],
                textfont=dict(size=18, color='green', family='Arial Black'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Z轴/光轴 (蓝色)
            fig.add_trace(go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
                mode='lines',
                line=dict(color='blue', width=4),
                name='Z轴',
                showlegend=True
            ))
            
            # Z轴标注
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[axis_length * 1.15],
                mode='text',
                text=['Z'],
                textfont=dict(size=18, color='blue', family='Arial Black'),
                showlegend=False,
                hoverinfo='skip'
            ))
            # endregion

            # endregion

            # region 3. 添加k矢量和S矢量
            # === 添加临界角下的 k 矢量和 S 矢量 (示意图) ===
            theta_critical = theta_dict[target_mode_3d]
            if not np.isnan(theta_critical):
                vector_length = 2.8  # 矢量长度
                
                # 根据所选平面确定实际的 theta 和 phi
                # 球坐标: theta是与Z轴夹角, phi是在XY平面投影与X轴夹角
                if user_config.plane == "XY":
                    # XY平面: 计算得到的临界角是phi, 用户输入的是theta
                    theta_rad = np.deg2rad(phi)  # 用户输入的theta
                    phi_rad = np.deg2rad(theta_critical)  # 计算得到的phi
                    display_theta = phi
                    display_phi = theta_critical
                else:  # XZ 或 YZ 平面
                    # XZ/YZ平面: 计算得到的临界角是theta, 用户输入的是phi
                    theta_rad = np.deg2rad(theta_critical)  # 计算得到的theta
                    phi_rad = np.deg2rad(phi)  # 用户输入的phi
                    display_theta = theta_critical
                    display_phi = phi
                
                # === 输入光1的k和S矢量 ===
                # 使用标准球坐标转笛卡尔坐标公式
                k1_x = vector_length * np.sin(theta_rad) * np.cos(phi_rad)
                k1_y = vector_length * np.sin(theta_rad) * np.sin(phi_rad)
                k1_z = vector_length * np.cos(theta_rad)
                
                # 获取走离角字符串并提取所有E光的信息
                walkoff_str = walkoff_dict[target_mode_3d]
                # walkoff_str格式示例: "𝐎 (0°) | 𝐄 (-0.5272° / -9.2020 mrad) | 𝐄 (-0.5268° / -9.1949 mrad)"
                # target_mode_3d格式示例: "𝐎 (1064) + 𝐄 (1064) → 𝐄 (532) (Type-I)"
                # 顺序：输入光1 | 输入光2 | 输出光
                import re
                
                # 打印调试信息
                print(f"DEBUG: target_mode_3d = {target_mode_3d}")
                print(f"DEBUG: walkoff_str = {walkoff_str}")
                
                # === 从target_mode_3d中提取所有波长 ===
                # 分割输入和输出部分
                mode_parts = target_mode_3d.split('→')
                input_part = mode_parts[0].strip()  # "𝐎 (1064) + 𝐄 (1064)" 或 "𝐎 (1064)"
                output_part = mode_parts[1].strip() if len(mode_parts) > 1 else ""  # "𝐄 (532) (Type-I)"
                
                # 提取所有波长（按顺序：输入光1, 输入光2, 输出光）
                wavelengths_list = []
                
                # 处理输入光
                if '+' in input_part:  # SFG模式：两个输入光
                    input_beams = input_part.split('+')
                    for beam in input_beams:
                        wl_match = re.search(r'\((\d+)nm\)', beam)
                        if wl_match:
                            wavelengths_list.append(int(wl_match.group(1)))
                else:  # SHG模式：一个输入光（两束相同）
                    wl_match = re.search(r'\((\d+)nm\)', input_part)
                    if wl_match:
                        wavelength = int(wl_match.group(1))
                        wavelengths_list.append(wavelength)
                        wavelengths_list.append(wavelength)
                
                # 处理输出光（去除Type-I/Type-II后缀）
                output_clean = output_part.split('(Type')[0].strip() if '(Type' in output_part else output_part
                wl_match = re.search(r'\((\d+)nm\)', output_clean)
                if wl_match:
                    wavelengths_list.append(int(wl_match.group(1)))
                
                print(f"DEBUG: 从mode提取的波长列表 = {wavelengths_list}")
                
                # === 从walkoff_str提取偏振和走离角 ===
                beams = walkoff_str.split('|')
                
                # 构建E光数据列表，匹配波长和走离角
                e_wave_data = []  # [(wavelength_nm, walkoff_deg), ...]
                
                for idx, beam_str in enumerate(beams):
                    beam_str = beam_str.strip()
                    # 检查是否为E光
                    if '𝐄' in beam_str:
                        # 提取走离角
                        match = re.search(r'([+-]?\d+\.\d+)°', beam_str)
                        if match and idx < len(wavelengths_list):
                            walkoff_deg = float(match.group(1))
                            wavelength = wavelengths_list[idx]
                            e_wave_data.append((wavelength, walkoff_deg))
                            print(f"DEBUG: 第{idx+1}个光是E光, 波长={wavelength}nm, 走离角={walkoff_deg}°")
                
                print(f"DEBUG: e_wave_data = {e_wave_data}")
                
                # 绘制 k 矢量 (波矢量) - 金黄色箭头
                k1_label = 'k矢量'
                fig.add_trace(go.Scatter3d(
                    x=[0, k1_x], y=[0, k1_y], z=[0, k1_z],
                    mode='lines',
                    line=dict(color='gold', width=5),
                    name=f'{k1_label} (θ={display_theta:.2f}°, φ={display_phi:.1f}°)',
                    showlegend=True,
                    hovertemplate=f'{k1_label}<br>θ=%.2f°<br>φ=%.1f°<extra></extra>' % (display_theta, display_phi)
                ))
                
                # 使用 Cone 绘制 k 矢量箭头
                fig.add_trace(go.Cone(
                    x=[k1_x], y=[k1_y], z=[k1_z],
                    u=[k1_x*0.1], v=[k1_y*0.1], w=[k1_z*0.1],
                    colorscale=[[0, 'gold'], [1, 'gold']],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.12,
                    name=f'{k1_label}箭头',
                    showlegend=False
                ))
                
                # 在k矢量旁边添加标注（放在更外侧）
                fig.add_trace(go.Scatter3d(
                    x=[k1_x*1], y=[k1_y*1], z=[k1_z*2],
                    mode='text',
                    text=['k'],
                    textfont=dict(size=16, color='gold', family='Arial Black'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # === 绘制所有E光的S矢量 ===
                # 为不同的S矢量使用不同的颜色
                s_colors = ['darkorange', 'purple', 'green']
                
                print(f"DEBUG: 准备绘制 {len(e_wave_data)} 个E光的S矢量")
                
                for idx, (wavelength_nm, walkoff_deg) in enumerate(e_wave_data):
                    print(f"DEBUG: 绘制第{idx+1}个S矢量: 波长={wavelength_nm}nm, 走离角={walkoff_deg}°")
                    
                    # 计算S矢量方向（走离角夸大3倍以便观察）
                    exaggerated_walkoff_rad = np.deg2rad(walkoff_deg * 3)
                    
                    # 根据平面确定走离方向
                    if user_config.plane in ["XZ", "YZ"]:
                        s_theta_rad = theta_rad - exaggerated_walkoff_rad
                        s_x = vector_length * np.sin(s_theta_rad) * np.cos(phi_rad)
                        s_y = vector_length * np.sin(s_theta_rad) * np.sin(phi_rad)
                        s_z = vector_length * np.cos(s_theta_rad)
                    else:  # XY平面
                        s_phi_rad = phi_rad - exaggerated_walkoff_rad
                        s_x = vector_length * np.sin(theta_rad) * np.cos(s_phi_rad)
                        s_y = vector_length * np.sin(theta_rad) * np.sin(s_phi_rad)
                        s_z = vector_length * np.cos(theta_rad)
                    
                    # 选择颜色
                    color = s_colors[idx % len(s_colors)]
                    
                    # 绘制S矢量线条
                    s_label = f'S ({wavelength_nm:.0f})'
                    fig.add_trace(go.Scatter3d(
                        x=[0, s_x], y=[0, s_y], z=[0, s_z],
                        mode='lines',
                        line=dict(color=color, width=5),
                        name=s_label,
                        showlegend=True,
                        hovertemplate=f'{s_label}<br>实际走离角={walkoff_deg:.4f}°<extra></extra>'
                    ))
                    
                    # 绘制S矢量箭头
                    fig.add_trace(go.Cone(
                        x=[s_x], y=[s_y], z=[s_z],
                        u=[s_x*0.1], v=[s_y*0.1], w=[s_z*0.1],
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        sizemode="absolute",
                        sizeref=0.12,
                        name=f'{s_label}箭头',
                        showlegend=False
                    ))
                    
                    # 在S矢量旁边添加标注（都放在内侧但错开）
                    s_text = f'S\n({wavelength_nm:.0f})'
                    label_distance = 1.05 + idx * 0.05  # 每个S矢量的标注距离稍微递增
                    fig.add_trace(go.Scatter3d(
                        x=[s_x*label_distance], y=[s_y*label_distance], z=[s_z*label_distance],
                        mode='text',
                        text=[s_text],
                        textfont=dict(size=12, color=color, family='Arial Black'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            # endregion
                
            # region 4. 添加角度标注 (走离角、theta角、phi角)
                # === 用弧线标注所有E光的走离角（k矢量和S矢量之间的角度）===
                print(f"DEBUG: 准备绘制 {len(e_wave_data)} 个走离角弧线")
                
                for idx, (wavelength_nm, walkoff_deg) in enumerate(e_wave_data):
                    print(f"DEBUG: 绘制第{idx+1}个走离角弧线: 波长={wavelength_nm}nm, 走离角={walkoff_deg}°")
                    
                    # 重新计算该E光的S矢量位置
                    exaggerated_walkoff_rad = np.deg2rad(walkoff_deg * 3)
                    
                    if user_config.plane in ["XZ", "YZ"]:
                        s_theta_rad = theta_rad - exaggerated_walkoff_rad
                        s_x = vector_length * np.sin(s_theta_rad) * np.cos(phi_rad)
                        s_y = vector_length * np.sin(s_theta_rad) * np.sin(phi_rad)
                        s_z = vector_length * np.cos(s_theta_rad)
                    else:
                        s_phi_rad = phi_rad - exaggerated_walkoff_rad
                        s_x = vector_length * np.sin(theta_rad) * np.cos(s_phi_rad)
                        s_y = vector_length * np.sin(theta_rad) * np.sin(s_phi_rad)
                        s_z = vector_length * np.cos(theta_rad)
                    
                    # 归一化k和S方向
                    k_norm = np.array([k1_x, k1_y, k1_z]) / np.linalg.norm([k1_x, k1_y, k1_z])
                    s_norm = np.array([s_x, s_y, s_z]) / np.linalg.norm([s_x, s_y, s_z])
                    
                    # 计算从k到S的弧线（使用球面线性插值）
                    # 为不同的E光使用不同的弧线半径和颜色
                    arc_radius_base = 1.5
                    arc_radius_walkoff = arc_radius_base - idx * 0.2  # 每个E光的弧线半径递减
                    n_points_walkoff = 25
                    color = s_colors[idx % len(s_colors)]
                    
                    # 使用球面线性插值生成k到S之间的弧线点
                    walkoff_arc_x = []
                    walkoff_arc_y = []
                    walkoff_arc_z = []
                    
                    for i in range(n_points_walkoff):
                        t = i / (n_points_walkoff - 1)
                        # 球面线性插值 (slerp)
                        theta_interp = np.arccos(np.clip(np.dot(k_norm, s_norm), -1, 1))
                        if theta_interp > 1e-6:  # 避免除零
                            sin_theta = np.sin(theta_interp)
                            a = np.sin((1 - t) * theta_interp) / sin_theta
                            b = np.sin(t * theta_interp) / sin_theta
                            interp_direction = a * k_norm + b * s_norm
                        else:
                            interp_direction = k_norm
                        
                        # 归一化并缩放到弧线半径
                        interp_direction = interp_direction / np.linalg.norm(interp_direction)
                        walkoff_arc_x.append(arc_radius_walkoff * interp_direction[0])
                        walkoff_arc_y.append(arc_radius_walkoff * interp_direction[1])
                        walkoff_arc_z.append(arc_radius_walkoff * interp_direction[2])
                    
                    # 绘制弧线
                    fig.add_trace(go.Scatter3d(
                        x=walkoff_arc_x, y=walkoff_arc_y, z=walkoff_arc_z,
                        mode='lines',
                        line=dict(color=color, width=3),
                        name=f'走离角弧线({wavelength_nm:.0f}nm)',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # 走离角标注文字位置（弧线中点）
                    mid_direction = (k_norm + s_norm) / 2
                    mid_direction = mid_direction / np.linalg.norm(mid_direction)
                    text_x = mid_direction[0] * (arc_radius_walkoff + 0.3)
                    text_y = mid_direction[1] * (arc_radius_walkoff + 0.3)
                    text_z = mid_direction[2] * (arc_radius_walkoff + 0.3)
                    
                    fig.add_trace(go.Scatter3d(
                        x=[text_x], y=[text_y], z=[text_z],
                        mode='text',
                        text=[f'ρ={walkoff_deg:.4f}°'],
                        textfont=dict(size=11, color=color, family='Arial Black'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # === 用弧线标注theta角（Z轴与k矢量的夹角）===
                arc_radius_theta = 0.8  # 弧线半径
                n_points = 30  # 弧线点数
                theta_arc = np.linspace(0, theta_rad, n_points)
                
                # 弧线在从Z轴到k矢量的平面上
                arc_theta_x = arc_radius_theta * np.sin(theta_arc) * np.cos(phi_rad)
                arc_theta_y = arc_radius_theta * np.sin(theta_arc) * np.sin(phi_rad)
                arc_theta_z = arc_radius_theta * np.cos(theta_arc)
                
                fig.add_trace(go.Scatter3d(
                    x=arc_theta_x, y=arc_theta_y, z=arc_theta_z,
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name='θ角弧线',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # theta角度标注文字
                theta_label_r = 1.0
                theta_label_theta = theta_rad / 2
                theta_label_x = theta_label_r * np.sin(theta_label_theta) * np.cos(phi_rad)
                theta_label_y = theta_label_r * np.sin(theta_label_theta) * np.sin(phi_rad)
                theta_label_z = theta_label_r * np.cos(theta_label_theta)
                
                fig.add_trace(go.Scatter3d(
                    x=[theta_label_x], y=[theta_label_y], z=[theta_label_z],
                    mode='text',
                    text=[f'θ={display_theta:.2f}°'],
                    textfont=dict(size=12, color='blue', family='Arial'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # === 绘制k矢量在XY平面上的投影 ===
                k_proj_x = k1_x
                k_proj_y = k1_y
                k_proj_z = 0
                
                # 从k矢量到其投影的虚线
                fig.add_trace(go.Scatter3d(
                    x=[k1_x, k_proj_x], y=[k1_y, k_proj_y], z=[k1_z, k_proj_z],
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dot'),
                    name='k投影线',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # k矢量在XY平面上的投影线（从原点到投影点）
                fig.add_trace(go.Scatter3d(
                    x=[0, k_proj_x], y=[0, k_proj_y], z=[0, 0],
                    mode='lines',
                    line=dict(color='purple', width=3, dash='dash'),
                    name='k在XY平面投影',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # === 用弧线标注phi角（X轴与投影的夹角，在XY平面上）===
                arc_radius_phi = 0.6  # 弧线半径
                phi_arc = np.linspace(0, phi_rad, n_points)
                
                # 弧线在XY平面上
                arc_phi_x = arc_radius_phi * np.cos(phi_arc)
                arc_phi_y = arc_radius_phi * np.sin(phi_arc)
                arc_phi_z = np.zeros(n_points)  # 完全在XY平面内（z=0）
                
                fig.add_trace(go.Scatter3d(
                    x=arc_phi_x, y=arc_phi_y, z=arc_phi_z,
                    mode='lines',
                    line=dict(color='green', width=3),
                    name='φ角弧线',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # phi角度标注文字
                phi_label_r = 0.75
                phi_label_phi = phi_rad / 2
                phi_label_x = phi_label_r * np.cos(phi_label_phi)
                phi_label_y = phi_label_r * np.sin(phi_label_phi)
                phi_label_z = 0
                
                fig.add_trace(go.Scatter3d(
                    x=[phi_label_x], y=[phi_label_y], z=[phi_label_z],
                    mode='text',
                    text=[f'φ={display_phi:.2f}°'],
                    textfont=dict(size=12, color='green', family='Arial'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                # endregion
                
                # region 5. 绘制晶体长方体
                # === 绘制晶体长方体（端面垂直于k矢量）===
                # k矢量方向的单位向量
                k_unit = np.array([k1_x, k1_y, k1_z]) / np.linalg.norm([k1_x, k1_y, k1_z])
                
                # 晶体参数
                crystal_length = 2.5  # 晶体长度（沿k方向）
                crystal_width = 0.8   # 晶体宽度
                crystal_height = 0.8  # 晶体高度
                
                # 晶体中心位置（后端面在原点，所以中心在 crystal_length/2 位置）
                crystal_center_distance = crystal_length / 2  # 晶体中心距原点的距离
                crystal_center = k_unit * crystal_center_distance
                
                # 构建与k垂直的两个正交向量（作为晶体的宽度和高度方向）
                # 选择一个不与k平行的向量
                if abs(k_unit[2]) < 0.9:
                    v1 = np.array([0, 0, 1])
                else:
                    v1 = np.array([1, 0, 0])
                
                # 通过叉乘得到两个正交向量
                v2 = np.cross(k_unit, v1)
                v2 = v2 / np.linalg.norm(v2)  # 归一化
                v3 = np.cross(k_unit, v2)
                v3 = v3 / np.linalg.norm(v3)  # 归一化
                
                # 定义长方体的8个顶点（相对于中心）
                # 顶点定义：沿k方向 ±crystal_length/2，沿v2方向 ±crystal_width/2，沿v3方向 ±crystal_height/2
                vertices = []
                for i in [-1, 1]:
                    for j in [-1, 1]:
                        for k in [-1, 1]:
                            vertex = (crystal_center + 
                                    i * (crystal_length / 2) * k_unit + 
                                    j * (crystal_width / 2) * v2 + 
                                    k * (crystal_height / 2) * v3)
                            vertices.append(vertex)
                
                vertices = np.array(vertices)
                
                # 定义长方体的12条边（连接顶点）
                edges = [
                    [0, 1], [2, 3], [4, 5], [6, 7],  # 平行于k的边
                    [0, 2], [1, 3], [4, 6], [5, 7],  # 平行于v2的边
                    [0, 4], [1, 5], [2, 6], [3, 7]   # 平行于v3的边
                ]
                
                # 绘制长方体的边框
                for edge in edges:
                    v_start = vertices[edge[0]]
                    v_end = vertices[edge[1]]
                    fig.add_trace(go.Scatter3d(
                        x=[v_start[0], v_end[0]],
                        y=[v_start[1], v_end[1]],
                        z=[v_start[2], v_end[2]],
                        mode='lines',
                        line=dict(color='cyan', width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # 绘制晶体的两个端面（用半透明平面）
                # 前端面（靠近k矢量方向）
                front_center = crystal_center + (crystal_length / 2) * k_unit
                # 后端面（远离k矢量方向）
                back_center = crystal_center - (crystal_length / 2) * k_unit
                
                # 创建端面的网格点
                face_u = np.linspace(-crystal_width/2, crystal_width/2, 5)
                face_v = np.linspace(-crystal_height/2, crystal_height/2, 5)
                face_u, face_v = np.meshgrid(face_u, face_v)
                
                # 前端面
                front_face_x = front_center[0] + face_u * v2[0] + face_v * v3[0]
                front_face_y = front_center[1] + face_u * v2[1] + face_v * v3[1]
                front_face_z = front_center[2] + face_u * v2[2] + face_v * v3[2]
                
                fig.add_trace(go.Surface(
                    x=front_face_x, y=front_face_y, z=front_face_z,
                    colorscale=[[0, 'rgba(0, 255, 255, 0.3)'], [1, 'rgba(0, 255, 255, 0.3)']],
                    showscale=False,
                    opacity=0.3,
                    name='晶体前端面',
                    hoverinfo='skip',
                    contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}}
                ))
                
                # 后端面
                back_face_x = back_center[0] + face_u * v2[0] + face_v * v3[0]
                back_face_y = back_center[1] + face_u * v2[1] + face_v * v3[1]
                back_face_z = back_center[2] + face_u * v2[2] + face_v * v3[2]
                
                fig.add_trace(go.Surface(
                    x=back_face_x, y=back_face_y, z=back_face_z,
                    colorscale=[[0, 'rgba(0, 255, 255, 0.3)'], [1, 'rgba(0, 255, 255, 0.3)']],
                    showscale=False,
                    opacity=0.3,
                    name='晶体后端面',
                    hoverinfo='skip',
                    contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}}
                ))
                # endregion
                
                # region 6. 绘制截面椭圆
                # === 绘制垂直于k矢量的截面与折射率椭球的交线（椭圆）===
                # 截面位置在原点（晶体后端面）
                cross_section_center = np.array([0.0, 0.0, 0.0])
                
                # 在截面上绘制折射率椭球的交线（椭圆）
                n_ellipse_points = 150
                angles = np.linspace(0, 2*np.pi, n_ellipse_points)
                
                # 绘制输入光和输出光的椭圆（两者对比）
                ellipses_to_draw = []
                
                # SHG模式：输入光1使用红色
                if user_config.process_type == 'SHG':
                    input1_color = 'rgba(255, 80, 80, 0.4)'
                    ellipses_to_draw.append((f'{user_config.wavelength1_nm:.0f}', scale_w1_x, scale_w1_y, scale_w1_z, input1_color, 6))
                
                # SFG模式：根据波长判断颜色
                elif user_config.process_type == 'SFG':
                    # 输入光1的颜色
                    if user_config.wavelength1_nm < user_config.wavelength2_nm:
                        input1_color = 'rgba(255, 215, 0, 0.4)'  # 短波长 - 黄色
                        input2_color = 'rgba(255, 80, 80, 0.4)'   # 长波长 - 红色
                    else:
                        input1_color = 'rgba(255, 80, 80, 0.4)'   # 长波长 - 红色
                        input2_color = 'rgba(255, 215, 0, 0.4)'  # 短波长 - 黄色
                    
                    ellipses_to_draw.append((f'{user_config.wavelength1_nm:.0f}', scale_w1_x, scale_w1_y, scale_w1_z, input1_color, 6))
                    ellipses_to_draw.append((f'{user_config.wavelength2_nm:.0f}', scale_w2_x, scale_w2_y, scale_w2_z, input2_color, 6))
                
                # 输出光使用蓝色
                ellipses_to_draw.append((f'{user_config.wavelength_out_nm:.0f}', scale_out_x, scale_out_y, scale_out_z, 'rgba(50, 100, 255, 0.4)', 6))
                
                for label, scale_x, scale_y, scale_z, color, width in ellipses_to_draw:
                    # 计算椭圆上的点
                    # 使用缩放后的椭球尺寸: (x/scale_x)^2 + (y/scale_y)^2 + (z/scale_z)^2 = 1
                    # 垂直于k的平面通过原点，法向量为k_unit
                    
                    ellipse_points = []
                    radii = []  # 存储每个方向的半径值
                    for angle in angles:
                        # 在垂直于k的平面上选择一个方向
                        direction_in_plane = np.cos(angle) * v2 + np.sin(angle) * v3
                        
                        # 沿着这个方向找到椭球表面的点
                        # 参数方程: P = t * direction_in_plane
                        # 代入椭球方程求t: (t*dx/scale_x)^2 + (t*dy/scale_y)^2 + (t*dz/scale_z)^2 = 1
                        dx, dy, dz = direction_in_plane
                        inv_n_squared = (dx/scale_x)**2 + (dy/scale_y)**2 + (dz/scale_z)**2
                        
                        if inv_n_squared > 1e-10:  # 避免除零
                            t = 1.0 / np.sqrt(inv_n_squared)
                            point = cross_section_center + t * direction_in_plane
                            ellipse_points.append(point)
                            radii.append(t)
                    
                    if len(ellipse_points) > 0:
                        ellipse_points = np.array(ellipse_points)
                        radii = np.array(radii)
                        
                        # 绘制椭圆交线
                        fig.add_trace(go.Scatter3d(
                            x=ellipse_points[:, 0],
                            y=ellipse_points[:, 1],
                            z=ellipse_points[:, 2],
                            mode='lines',
                            line=dict(color=color, width=width),
                            name=f'{label}截面椭圆',
                            showlegend=True
                        ))
                        
                        # === 找到长轴和短轴 ===
                        max_radius_idx = np.argmax(radii)
                        min_radius_idx = np.argmin(radii)
                        
                        major_radius = radii[max_radius_idx]
                        minor_radius = radii[min_radius_idx]
                        
                        major_angle = angles[max_radius_idx]
                        minor_angle = angles[min_radius_idx]
                        
                        # 长轴方向
                        major_direction = np.cos(major_angle) * v2 + np.sin(major_angle) * v3
                        major_point = cross_section_center + major_radius * major_direction
                        major_point_neg = cross_section_center - major_radius * major_direction
                        
                        # 短轴方向
                        minor_direction = np.cos(minor_angle) * v2 + np.sin(minor_angle) * v3
                        minor_point = cross_section_center + minor_radius * minor_direction
                        minor_point_neg = cross_section_center - minor_radius * minor_direction
                        
                        # 绘制长轴虚线
                        # 根据波长选择颜色（label现在是波长）
                        wavelength_val = float(label.replace('nm', ''))
                        if user_config.process_type == 'SHG':
                            # SHG: 基频光用红色，倍频光用蓝色
                            axis_color = 'rgb(255, 80, 80)' if wavelength_val == user_config.wavelength1_nm else 'rgb(50, 100, 255)'
                        else:  # SFG
                            # 输入光1的颜色
                            if wavelength_val == user_config.wavelength1_nm:
                                axis_color = 'rgb(255, 215, 0)' if user_config.wavelength1_nm < user_config.wavelength2_nm else 'rgb(255, 80, 80)'
                            elif wavelength_val == user_config.wavelength2_nm:
                                axis_color = 'rgb(255, 215, 0)' if user_config.wavelength2_nm < user_config.wavelength1_nm else 'rgb(255, 80, 80)'
                            else:  # 和频光用蓝色
                                axis_color = 'rgb(50, 100, 255)'
                        
                        fig.add_trace(go.Scatter3d(
                            x=[major_point_neg[0], major_point[0]],
                            y=[major_point_neg[1], major_point[1]],
                            z=[major_point_neg[2], major_point[2]],
                            mode='lines',
                            line=dict(color=axis_color, width=3, dash='dash'),
                            name=f'{label}长轴',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # 绘制短轴虚线
                        fig.add_trace(go.Scatter3d(
                            x=[minor_point_neg[0], minor_point[0]],
                            y=[minor_point_neg[1], minor_point[1]],
                            z=[minor_point_neg[2], minor_point[2]],
                            mode='lines',
                            line=dict(color=axis_color, width=3, dash='dash'),
                            name=f'{label}短轴',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # 标注长轴值 - 根据不同光源分散标注位置
                        # 波长值已经在上面解析
                        if wavelength_val == user_config.wavelength1_nm:
                            offset_a = v2 * 0.3
                        elif wavelength_val == user_config.wavelength2_nm:
                            offset_a = -v2 * 0.3
                        else:  # 输出光/倍频光
                            offset_a = v3 * 0.3
                        
                        major_label_pos = major_point * 1.15
                        fig.add_trace(go.Scatter3d(
                            x=[major_label_pos[0] + offset_a[0]],
                            y=[major_label_pos[1] + offset_a[1]],
                            z=[major_label_pos[2] + offset_a[2]],
                            mode='text',
                            text=[f'a={major_radius:.3f}'],
                            textfont=dict(size=10, color=axis_color, family='Arial'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # 标注短轴值 - 使用相应的偏移方向
                        if wavelength_val == user_config.wavelength1_nm:
                            offset_b = v2 * 0.3
                        elif wavelength_val == user_config.wavelength2_nm:
                            offset_b = -v2 * 0.3
                        else:  # 输出光/倍频光
                            offset_b = v3 * 0.3
                        
                        minor_label_pos = minor_point_neg * 1.05
                        fig.add_trace(go.Scatter3d(
                            x=[minor_label_pos[0] + offset_b[0]],
                            y=[minor_label_pos[1] + offset_b[1]],
                            z=[minor_label_pos[2] + offset_b[2]],
                            mode='text',
                            text=[f'b={minor_radius:.3f}'],
                            textfont=dict(size=10, color=axis_color, family='Arial'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                # endregion

            # region 7. 设置图形布局和保存

            fig.update_layout(
                scene = dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data',  # 保证坐标轴比例一致
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)  # 设置视角
                    ),
                    bgcolor='rgba(240, 240, 250, 0.9)'  # 浅色背景
                ),
                width=900,
                height=700,
                margin=dict(r=20, l=10, b=10, t=50),
                title=dict(
                    text=f'{user_config.crystal_name} 晶体折射率椭球示意图<br><sub>相位匹配模式: {target_mode_3d} | X,Y,Z为晶体光学主轴</sub>',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18)
                ),
                showlegend=True,
                legend=dict(x=0.7, y=0.95)
            )
            
            # 保存到session_state
            st.session_state['3d_fig'] = fig
            st.session_state['3d_config'] = {
                'n_x_w': n_x_w1, 'n_y_w': n_y_w1, 'n_z_w': n_z_w1,
                'n_x_out': n_x_out, 'n_y_out': n_y_out, 'n_z_out': n_z_out,
                'wavelength1_nm': user_config.wavelength1_nm,
                'wavelength_out_nm': user_config.wavelength_out_nm,
                'process_type': user_config.process_type
            }
            
            # SFG模式：额外保存第二束光信息
            if user_config.process_type == 'SFG':
                st.session_state['3d_config'].update({
                    'n_x_w1': n_x_w1, 'n_y_w1': n_y_w1, 'n_z_w1': n_z_w1,
                    'n_x_w2': n_x_w2, 'n_y_w2': n_y_w2, 'n_z_w2': n_z_w2,
                    'wavelength2_nm': user_config.wavelength2_nm
                })
            # endregion
        
        # region 8. 显示保存的3D图
        if '3d_fig' in st.session_state:
            st.plotly_chart(st.session_state['3d_fig'], use_container_width=True)
            
            # 添加说明
            st.caption(r"""
            * **k 矢量 ($\overrightarrow{k}$)**：波矢方向。
            * **S 矢量 ($\overrightarrow{S}$)**：能量流/光线方向。
            * **走离角 ($\rho$)**：为了方便展示，视觉上夸大了 3 倍。
            * **长方体**：晶体几何示意，其端面垂直于 $\overrightarrow{k}$ 方向。
            * **截面椭圆**：表示垂直于 $\overrightarrow{k}$ 方向的折射率分布。
            """)
            
            # 显示折射率数值信息
            config = st.session_state['3d_config']
            
            # SHG模式：两列布局
            if config['process_type'] == 'SHG':
                col1, col2 = st.columns(2)
                with col1:
                    st.error(f"**输入光 ({config['wavelength1_nm']:.1f} nm)**")
                    st.write(f"n_x = {config['n_x_w']:.5f}")
                    st.write(f"n_y = {config['n_y_w']:.5f}")
                    st.write(f"n_z = {config['n_z_w']:.5f}")
                with col2:
                    st.info(f"**输出光 ({config['wavelength_out_nm']:.1f} nm)**")
                    st.write(f"n_x = {config['n_x_out']:.5f}")
                    st.write(f"n_y = {config['n_y_out']:.5f}")
                    st.write(f"n_z = {config['n_z_out']:.5f}")
            
            # SFG模式：三列布局，短波长在中间用黄色
            else:
                col1, col2, col3 = st.columns(3)
                
                # 确定哪束光更短
                if config['wavelength2_nm'] < config['wavelength1_nm']:
                    # λ2更短
                    with col1:
                        st.error(f"**输入光1 ({config['wavelength1_nm']:.1f} nm)**")
                        st.write(f"n_x = {config['n_x_w1']:.5f}")
                        st.write(f"n_y = {config['n_y_w1']:.5f}")
                        st.write(f"n_z = {config['n_z_w1']:.5f}")
                    with col2:
                        st.warning(f"**输入光2 ({config['wavelength2_nm']:.1f} nm)**")
                        st.write(f"n_x = {config['n_x_w2']:.5f}")
                        st.write(f"n_y = {config['n_y_w2']:.5f}")
                        st.write(f"n_z = {config['n_z_w2']:.5f}")
                    with col3:
                        st.info(f"**输出光 ({config['wavelength_out_nm']:.1f} nm)**")
                        st.write(f"n_x = {config['n_x_out']:.5f}")
                        st.write(f"n_y = {config['n_y_out']:.5f}")
                        st.write(f"n_z = {config['n_z_out']:.5f}")
                else:
                    # λ1更短
                    with col1:
                        st.error(f"**输入光2 ({config['wavelength2_nm']:.1f} nm)**")
                        st.write(f"n_x = {config['n_x_w2']:.5f}")
                        st.write(f"n_y = {config['n_y_w2']:.5f}")
                        st.write(f"n_z = {config['n_z_w2']:.5f}")
                    with col2:
                        st.warning(f"**输入光1 ({config['wavelength1_nm']:.1f} nm)**")
                        st.write(f"n_x = {config['n_x_w1']:.5f}")
                        st.write(f"n_y = {config['n_y_w1']:.5f}")
                        st.write(f"n_z = {config['n_z_w1']:.5f}")
                    with col3:
                        st.info(f"**输出光 ({config['wavelength_out_nm']:.1f} nm)**")
                        st.write(f"n_x = {config['n_x_out']:.5f}")
                        st.write(f"n_y = {config['n_y_out']:.5f}")
                        st.write(f"n_z = {config['n_z_out']:.5f}")

         # endregion

        # endregion
  
        # ============================================================================
        # 模块3: 接受带宽分析
        # ============================================================================
        st.subheader("📈 3. 接受带宽分析")    
        
        if not valid_modes:
            st.warning("当前没有有效的相位匹配模式，无法进行带宽分析。")
        else:
            # 让用户选择一个模式进行深入分析
            col_sel, _ = st.columns([1, 2])
            with col_sel:
                target_mode_bandwidth = st.selectbox("👉 请选择要分析的模式:", valid_modes, key='mode_bandwidth')
            
            # 初始化当前激活的标签页
            if 'active_bandwidth_tab' not in st.session_state:
                st.session_state['active_bandwidth_tab'] = "角度带宽"
            
            # 扫描精度设置（必须在按钮前定义）
            st.markdown("##### 扫描精度设置")
            col_set1, col_set2, col_set3 = st.columns(3)
            
            with col_set1:
                st.markdown("**角度扫描**")
                scan_step_angle = st.slider("步数", 100, 5000, 1000, key="step_ang")
                scan_res_angle = st.number_input("精度 (mrad)", 0.001, 1.0, 0.001, step=0.001, format="%.3f", key="res_ang")
            
            with col_set2:
                st.markdown("**波长扫描**")
                scan_step_wave = st.slider("步数", 100, 5000, 1000, key="step_wav")
                scan_res_wave = st.number_input("精度 (nm)", 0.001, 1.0, 0.001, step=0.001, format="%.3f", key="res_wav")
            
            with col_set3:
                st.markdown("**温度扫描**")
                scan_step_temp = st.slider("步数", 100, 5000, 1000, key="step_tem")
                scan_res_temp = st.number_input("精度 (°C)", 0.01, 10.0, 0.1, key="res_tem")
            
            st.divider()
            
            # 添加一键计算所有带宽按钮
            if st.button("一键计算所有带宽", key="btn_calc_all", type="primary", use_container_width=True):
                with st.spinner("正在计算所有带宽..."):
                    try:
                        # 计算角度带宽
                        fig_ang, val_mrad, val_deg = simulation.acceptance_angle(
                            theta_dict, target_mode_bandwidth, step=scan_step_angle, res=scan_res_angle
                        )
                        st.session_state['res_angle_fig'] = fig_ang
                        st.session_state['res_angle_val_mrad'] = val_mrad
                        st.session_state['res_angle_val_deg'] = val_deg
                        
                        # 计算波长带宽
                        fig_wav, val_nm, val_ghz = simulation.acceptance_wavelength(
                            theta_dict, target_mode_bandwidth, step=scan_step_wave, res=scan_res_wave
                        )
                        st.session_state['res_wave_fig'] = fig_wav
                        st.session_state['res_wave_val_nm'] = val_nm
                        st.session_state['res_wave_val_ghz'] = val_ghz
                        
                        # 计算温度带宽
                        fig_temp, val_temp = simulation.acceptance_temperature(
                            theta_dict, target_mode_bandwidth, step=scan_step_temp, res=scan_res_temp
                        )
                        st.session_state['res_temp_fig'] = fig_temp
                        st.session_state['res_temp_val_temp'] = val_temp
                        
                        st.session_state['all_bandwidths_calculated'] = True
                        st.success("✅ 所有带宽计算完成！")
                    except Exception as e:
                        st.error(f"计算出错: {e}")
            
            # 显示计算结果
            if st.session_state.get('all_bandwidths_calculated', False):
                st.write("---")
                st.write("**所有带宽结果：**")
                col_all1, col_all2, col_all3 = st.columns(3)
                
                with col_all1:
                    if 'res_angle_fig' in st.session_state:
                        st.pyplot(st.session_state['res_angle_fig'])
                        st.metric("角度带宽 (FWHM)", f"{st.session_state['res_angle_val_mrad']:.4f} mrad·cm")
                        st.caption(f"约 {st.session_state['res_angle_val_deg']:.4f}°·cm")
                
                with col_all2:
                    if 'res_wave_fig' in st.session_state:
                        st.pyplot(st.session_state['res_wave_fig'])
                        st.metric("波长带宽 (FWHM)", f"{st.session_state['res_wave_val_nm']:.4f} nm·cm")
                        st.caption(f"频率: {st.session_state['res_wave_val_ghz']:.2f} GHz·cm")
                
                with col_all3:
                    if 'res_temp_fig' in st.session_state:
                        st.pyplot(st.session_state['res_temp_fig'])
                        st.metric("温度带宽 (FWHM)", f"{st.session_state['res_temp_val_temp']:.4f} K·cm")
                
                st.write("---")
                st.info(
                    "💡 **关于带宽单位 ·cm 的说明**\n\n"
                    "带宽结果的单位带有 **·cm**（例如 mrad·cm、nm·cm、K·cm），表示该值已归一化到 **1 cm 晶体长度**。\n\n"
                    "实际带宽 = 显示值 ÷ 晶体长度 (cm)\n\n"
                    "**举例：** 若角度带宽为 **0.4880 mrad·cm**，晶体长度为 **2 cm**，则实际角度带宽为：\n\n"
                    "  0.4880 ÷ 2 = **0.2440 mrad**\n\n"
                    "同理，若波长带宽为 **0.1260 nm·cm**，晶体长度为 **0.5 cm**，则实际波长带宽为：\n\n"
                    "  0.1260 ÷ 0.5 = **0.2520 nm**\n\n"
                    "晶体越长，带宽越窄；晶体越短，带宽越宽。"
                )

    else:
        # ============================================================================
        # 温度匹配结果展示
        # ============================================================================
        st.subheader("🌡️ 温度匹配结果")
        
        if 'temp_match_results' in st.session_state:
            temp_match_results = st.session_state['temp_match_results']
            fixed_axis = st.session_state.get('fixed_axis_sidebar', '')
            
            st.info(f"**传播轴**: {fixed_axis} 轴")
            
            # 准备表格数据
            table_data = []
            for mode, result in temp_match_results.items():
                if 'error' in result:
                    # 计算出错
                    table_data.append({
                        "模式": mode,
                        "匹配温度点": "计算出错",
                        "匹配点数量": 0
                    })
                elif result['matching_temperatures']:
                    # 找到匹配点
                    temps = result['matching_temperatures']
                    temps_str = ", ".join([f"{t:.2f}°C" for t in temps])
                    table_data.append({
                        "模式": mode,
                        "匹配温度点": temps_str,
                        "匹配点数量": len(temps)
                    })
                else:
                    # 未找到匹配点
                    table_data.append({
                        "模式": mode,
                        "匹配温度点": "❌ 无匹配点",
                        "匹配点数量": 0
                    })
            
            # 展示表格
            df = pd.DataFrame(table_data)
            st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "模式": st.column_config.TextColumn(width="large"),
                    "匹配温度点": st.column_config.TextColumn(width="large"),
                    "匹配点数量": st.column_config.NumberColumn(width="small"),
                }
            )
            
            st.caption("**说明**: 表格显示了所选传播轴方向上所有可能模式的温度匹配结果")
            
            # 选择一个模式进行带宽分析
            modes_with_match = [mode for mode, result in temp_match_results.items() 
                              if not 'error' in result and result['matching_temperatures']]
            
            if modes_with_match:
                st.write("---")
                st.write("**接受带宽分析**")
                
                # 让用户选择一个有匹配点的模式进行带宽分析
                col_mode_sel, _ = st.columns([1, 2])
                with col_mode_sel:
                    selected_mode_for_bandwidth = st.selectbox(
                        "👉 请选择要分析带宽的模式:", 
                        modes_with_match, 
                        key='mode_bandwidth_temp'
                    )
                
                # 获取该模式的结果
                selected_result = temp_match_results[selected_mode_for_bandwidth]
                matching_temp = selected_result['matching_temperatures'][0]
                
                st.info(f"**分析模式**: {selected_mode_for_bandwidth} | **匹配温度**: {matching_temp:.2f}°C")
                
                # 扫描精度设置
                st.markdown("##### 扫描精度设置")
                col_set1, col_set2, col_set3 = st.columns(3)
                
                with col_set1:
                    st.markdown("**温度扫描**")
                    temp_step_bw = st.slider("步数", 100, 5000, 1000, key="temp_step_bw")
                    temp_res_bw = st.number_input("精度 (K)", 0.01, 10.0, 0.1, key="temp_res_bw")
                
                with col_set2:
                    st.markdown("**波长扫描**")
                    wl_step_bw = st.slider("步数", 100, 5000, 1000, key="wl_step_bw")
                    wl_res_bw = st.number_input("精度 (nm)", 0.001, 1.0, 0.001, step=0.001, format="%.3f", key="wl_res_bw")
                
                with col_set3:
                    st.markdown("**角度扫描**")
                    ang_step_bw = st.slider("步数", 100, 5000, 1000, key="ang_step_bw")
                    ang_res_bw = st.number_input("精度 (mrad)", 0.001, 1.0, 0.001, step=0.001, format="%.3f", key="ang_res_bw")
                
                # 需要临时修改配置的温度为匹配温度
                original_temp = simulation.cfg.temperature
                simulation.cfg.temperature = matching_temp
                
                # 一键计算所有带宽按钮
                if st.button("一键计算所有带宽", key="btn_calc_all_ncpm", type="primary", use_container_width=True):
                    with st.spinner("正在计算所有带宽..."):
                        try:
                            fake_theta_dict = {selected_mode_for_bandwidth: 0.0}
                            
                            # 计算温度带宽
                            fig_temp, acc_temp = simulation.acceptance_temperature(
                                fake_theta_dict, selected_mode_for_bandwidth, 
                                step=temp_step_bw, res=temp_res_bw
                            )
                            st.session_state['ncpm_res_temp_fig'] = fig_temp
                            st.session_state['ncpm_res_temp_val'] = acc_temp
                            
                            # 计算波长带宽
                            fig_wl, acc_wl, acc_bw = simulation.acceptance_wavelength(
                                fake_theta_dict, selected_mode_for_bandwidth,
                                step=wl_step_bw, res=wl_res_bw
                            )
                            st.session_state['ncpm_res_wl_fig'] = fig_wl
                            st.session_state['ncpm_res_wl_val'] = acc_wl
                            st.session_state['ncpm_res_wl_bw'] = acc_bw
                            
                            # 计算角度带宽（俯仰和偏航两个方向）
                            # 需要将XYZ模式转换为OE模式，因为XYZ模式不考虑角度变化
                            original_plane = simulation.cfg.plane
                            
                            # 从XYZ模式中提取输入和输出的偏振方向
                            mode_parts = selected_mode_for_bandwidth.split('→')
                            input_part = mode_parts[0].strip()
                            output_part = mode_parts[1].strip()
                            
                            # 提取输入偏振（假设是两个相同的，如"𝐙 + 𝐙"）
                            input_pols = [p.strip() for p in input_part.split('+')]
                            input_pol = input_pols[0].split('(')[0].strip()  # 取第一个，去掉波长
                            
                            # 提取输出偏振
                            output_pol = output_part.split('(')[0].strip() # 去掉波长
                            
                            # 根据传播轴确定两个垂直平面和对应的theta基准角
                            if fixed_axis == 'X':
                                planes = ['XY', 'XZ']  # X轴传播
                            elif fixed_axis == 'Y':
                                planes = ['XY', 'YZ']  # Y轴传播
                            else:  # Z
                                planes = ['XZ', 'YZ']  # Z轴传播
                            
                            # 分别计算两个平面的角度带宽
                            angle_results = {}
                            
                            for plane in planes:
                                # 设置平面并更新对应的轴配置
                                simulation.cfg.plane = plane
                                simulation.key_static, simulation.key_cos, simulation.key_sin = simulation.plane_config[plane]
                                
                                # 根据平面确定哪个偏振方向是O光（垂直于平面）、哪个是E光（在平面内）
                                # plane_config: XY→('n_z',n_x,n_y), XZ→('n_y',n_z,n_x), YZ→('n_x',n_z,n_y)
                                # key_static是垂直于平面的轴（O光方向）
                                plane_to_static_axis = {
                                    'XY': '𝐙',
                                    'XZ': '𝐘',
                                    'YZ': '𝐗'
                                }
                                o_light_axis = plane_to_static_axis[plane] # 获取O光对应的轴
                                
                                # 确定输入光的偏振类型
                                input_is_o = (input_pol == o_light_axis)
                                # 确定输出光的偏振类型
                                output_is_o = (output_pol == o_light_axis)
                                
                                # 构建该平面对应的OE模式
                                input_oe = '𝐎' if input_is_o else '𝐄'
                                output_oe = '𝐎' if output_is_o else '𝐄'
                                
                                # 重建OE模式字符串（保留波长信息）
                                oe_mode = selected_mode_for_bandwidth.replace(input_pol, input_oe).replace(output_pol, output_oe)

                                # 角度带宽计算：根据平面和传播轴确定theta的计算方式
                                angle_offset = np.arange(-ang_step_bw, ang_step_bw) * ang_res_bw * 1e-3  # 偏移角（弧度）
                                
                                # 确定theta的基准值和计算方式
                                # Z轴传播时：offset为theta，XZ平面phi为0，YZ平面phi为pi/2
                                # Y轴传播时：XY平面时phi为pi/2 - offset，YZ平面时theta为pi/2 - offset，phi为pi/2；
                                # X轴传播时：XY平面时phi为offset，XZ平面时theta为 pi/2 - offset，phi为0；
                                if fixed_axis == 'X':
                                    # X轴传播
                                    if plane == 'XY':
                                        # XY平面：phi = offset
                                        theta_axis = np.abs(angle_offset)
                                    elif plane == 'XZ':
                                        # XZ平面：theta = pi/2 - offset
                                        theta_axis = np.pi / 2 - angle_offset
                                elif fixed_axis == 'Y':
                                    # Y轴传播
                                    if plane == 'XY':
                                        # XY平面：phi = pi/2 - offset
                                        theta_axis = np.pi/2 - np.abs(angle_offset)
                                    elif plane == 'YZ':
                                        # YZ平面：theta = pi/2 - offset
                                        theta_axis = np.pi / 2 - angle_offset
                                else:
                                    # Z轴传播
                                    if plane == 'XZ':
                                        # XZ平面：theta = offset
                                        theta_axis = np.abs(angle_offset)
                                    elif plane == 'YZ':
                                        # YZ平面：phi = pi/2 + offset
                                        theta_axis = np.abs(angle_offset)
                                
                                # 使用OE模式计算每个角度的delta_n
                                delta_n_array = np.array([
                                    simulation.delta_n(oe_mode, theta=t)
                                    for t in theta_axis
                                ])
                                angle_axis = angle_offset
                                
                                # 计算Δk = 2π/λ_out × Δn
                                delta_k = (np.pi * 2 / simulation.cfg.wavelength_out_um) * delta_n_array
                                
                                # 计算效率 η(Δk) = sinc²(Δk × L/2)
                                efficiency = (np.sinc(delta_k * 1e4 / (2 * np.pi)))**2
                                
                                # 绘图
                                fig_ang, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(angle_axis * 1000, efficiency, 'r-', linewidth=1.5)
                                ax.set_xlabel('Angle Deviation / mrad', fontsize=12)
                                # 根据过程类型设置纵轴标题
                                ylabel = 'SHG Efficiency' if simulation.cfg.process_type == 'SHG' else 'SFG Efficiency'
                                ax.set_ylabel(ylabel, fontsize=12)
                                display_mode = selected_mode_for_bandwidth.replace('𝐗', 'X').replace('𝐘', 'Y').replace('𝐙', 'Z')
                                ax.set_title(f'Acceptance Angle Curve for {simulation.cfg.crystal_name} ({plane} plane)\n({display_mode})', fontsize=14)
                                ax.grid(True, alpha=0.3)
                                
                                # 计算FWHM
                                half_max = 0.5
                                indices_above_half = np.where(efficiency >= half_max)[0]
                                
                                if len(indices_above_half) > 0:
                                    lower_index = indices_above_half[0]
                                    upper_index = indices_above_half[-1]
                                    acc_ang = (angle_axis[upper_index] - angle_axis[lower_index]) * 1000  # mrad
                                    acc_ang_deg = np.rad2deg(angle_axis[upper_index] - angle_axis[lower_index])
                                else:
                                    acc_ang = np.nan
                                    acc_ang_deg = np.nan
                                
                                angle_results[plane] = {
                                    'fig': fig_ang,
                                    'acc_ang': acc_ang,
                                    'acc_ang_deg': acc_ang_deg
                                }
                            
                            # 恢复原始plane配置和对应的轴
                            simulation.cfg.plane = original_plane
                            simulation.key_static, simulation.key_cos, simulation.key_sin = simulation.plane_config[original_plane]
                            
                            # 保存两个平面的结果
                            st.session_state['ncpm_res_ang_results'] = angle_results
                            st.session_state['ncpm_res_ang_planes'] = planes
                            
                            st.session_state['ncpm_all_calculated'] = True
                            st.success("✅ 所有带宽计算完成！")
                        except Exception as e:
                            st.error(f"带宽计算出错: {e}")
                        finally:
                            simulation.cfg.temperature = original_temp
                            # 确保plane也被恢复
                            if 'original_plane' in locals():
                                simulation.cfg.plane = original_plane
                
                # 显示所有计算结果
                if st.session_state.get('ncpm_all_calculated', False):
                    st.write("---")
                    st.write("**所有带宽结果：**")
                    
                    # 第一行：温度带宽和波长带宽
                    col_row1_1, col_row1_2 = st.columns(2)
                    
                    with col_row1_1:
                        if 'ncpm_res_temp_fig' in st.session_state:
                            st.pyplot(st.session_state['ncpm_res_temp_fig'])
                            st.metric("温度带宽 (FWHM)", 
                                    f"{st.session_state['ncpm_res_temp_val']:.4f} K·cm" 
                                    if not np.isnan(st.session_state['ncpm_res_temp_val']) else "N/A")
                    
                    with col_row1_2:
                        if 'ncpm_res_wl_fig' in st.session_state:
                            st.pyplot(st.session_state['ncpm_res_wl_fig'])
                            st.metric("波长带宽 (FWHM)", 
                                    f"{st.session_state['ncpm_res_wl_val']:.4f} nm·cm" 
                                    if not np.isnan(st.session_state['ncpm_res_wl_val']) else "N/A")
                            st.caption(f"频率: {st.session_state['ncpm_res_wl_bw']:.2f} GHz·cm" 
                                     if not np.isnan(st.session_state['ncpm_res_wl_bw']) else "")
                    
                    # 第二行：两个平面的角度带宽
                    st.write("**角度带宽 (FWHM)**")
                    if 'ncpm_res_ang_results' in st.session_state:
                        planes = st.session_state['ncpm_res_ang_planes']
                        results = st.session_state['ncpm_res_ang_results']
                        
                        col_row2_1, col_row2_2 = st.columns(2)
                        
                        with col_row2_1:
                            plane = planes[0]
                            st.markdown(f"**{plane}平面**")
                            st.pyplot(results[plane]['fig'])
                            
                            acc_ang = results[plane]['acc_ang']
                            acc_ang_deg = results[plane]['acc_ang_deg']
                            
                            st.metric(f"{plane}平面角度带宽", 
                                    f"{acc_ang:.4f} mrad·cm" if not np.isnan(acc_ang) else "N/A")
                            st.caption(f"约 {acc_ang_deg:.4f}°·cm" if not np.isnan(acc_ang_deg) else "")
                        
                        with col_row2_2:
                            plane = planes[1]
                            st.markdown(f"**{plane}平面**")
                            st.pyplot(results[plane]['fig'])
                            
                            acc_ang = results[plane]['acc_ang']
                            acc_ang_deg = results[plane]['acc_ang_deg']
                            
                            st.metric(f"{plane}平面角度带宽", 
                                    f"{acc_ang:.4f} mrad·cm" if not np.isnan(acc_ang) else "N/A")
                            st.caption(f"约 {acc_ang_deg:.4f}°·cm" if not np.isnan(acc_ang_deg) else "")
                    
                    st.write("---")
                    st.info(
                        "💡 **关于带宽单位 ·cm 的说明**\n\n"
                        "带宽结果的单位带有 **·cm**（例如 mrad·cm、nm·cm、K·cm），表示该值已归一化到 **1 cm 晶体长度**。\n\n"
                        "实际带宽 = 显示值 ÷ 晶体长度 (cm)\n\n"
                        "**举例：** 若温度带宽为 **5.8000 K·cm**，晶体长度为 **2 cm**，则实际温度带宽为：\n\n"
                        "  5.8000 ÷ 2 = **2.9000 K**\n\n"
                        "同理，若波长带宽为 **0.1260 nm·cm**，晶体长度为 **0.5 cm**，则实际波长带宽为：\n\n"
                        "  0.1260 ÷ 0.5 = **0.2520 nm**\n\n"
                        "晶体越长，带宽越窄；晶体越短，带宽越宽。"
                    )

