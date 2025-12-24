"""
非线性晶体二次谐波(SHG)模拟器
功能：相位匹配计算、3D可视化、接受带宽分析
"""

import numpy as np
import matplotlib.pyplot as plt
import configuration
import simulation
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from simulation import Solver
from configuration import SimulationConfig
from scipy.optimize import fsolve
from matplotlib.ticker import FuncFormatter  

# ============================================================================
# 页面配置与样式
# ============================================================================
st.set_page_config(
    page_title="非线性晶体SHG模拟 V1.0",
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

st.title("非线性晶体二次谐波 (SHG) 模拟器")

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

# 侧边栏：参数输入
with st.sidebar:
    st.header("仿真参数设置")
    
    # 基础参数
    crystal_name = st.selectbox("晶体类型", ["LBO", "BBO", "CLBO","KTP","KDP","DKDP"], index=0, 
                               help="Sellmeier方程来源有所差异，计算可能有微小差别") 
    wavelength_nm = st.number_input("基频波长 (nm)", value=1064.0, step=0.1, help="精度为0.1nm")
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
        # 双轴晶体：平面可选，角度根据平面自动锁定
        plane = st.selectbox("k矢量所在平面", ["XY", "YZ", "XZ"], index=2, 
                            help="双轴晶体可选择不同平面")
        
        if plane == "XY":
            phi = 90.0
            st.number_input("θ角 (度)", value=90.0, step=0.1, disabled=True, 
                           help="XY平面时θ角锁定为90°")
        else:  # YZ或XZ
            phi = 0.0
            st.number_input("φ角 (度)", value=0.0, step=0.1, disabled=True, 
                           help="YZ/XZ平面时φ角锁定为0°")

    st.divider()
    
    # 扫描精度设置（用于带宽分析）
    st.markdown("扫描精度设置")
    
    with st.expander("角度扫描", expanded=True):
        scan_step_angle = st.slider("步数", 100, 5000, 1000, key="step_ang")
        scan_res_angle = st.number_input("精度 (mrad)", 0.001, 1.0, 0.001, step=0.001, format="%.3f", key="res_ang")
 
    with st.expander("波长扫描", expanded=False):
        scan_step_wave = st.slider("步数", 100, 5000, 1000, key="step_wav")
        scan_res_wave = st.number_input("精度 (nm)", 0.001, 1.0, 0.001, step=0.001, format="%.3f", key="res_wav")
        
    with st.expander("温度扫描", expanded=False):
        scan_step_temp = st.slider("步数", 100, 5000, 1000, key="step_tem")
        scan_res_temp = st.number_input("精度 (°C)", 0.01, 10.0, 0.1, key="res_tem")

# ============================================================================
# 初始化计算核心
# ============================================================================

try:
    user_config = SimulationConfig(crystal_name=crystal_name, 
                                   wavelength=wavelength_nm, 
                                   temperature=temperature, 
                                   plane=plane) 
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
    keys_to_clear = ['res_angle_fig', 'res_wave_fig', 'res_temp_fig']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    with st.spinner("正在求解相位匹配方程..."):
        try:
            # 1. 计算临界角
            st.session_state.theta_dict = simulation.criticalangle()
            # 2. 计算走离角
            st.session_state.walkoff_dict = simulation.walkoff_angle(st.session_state.theta_dict)
            # 3. 计算有效非线性系数
            st.session_state.d_eff_dict = simulation.d_eff(st.session_state.theta_dict, phi)
            # 标记运行完成
            st.session_state.has_run = True
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

    # ============================================================================
    # 模块2: 3D折射率椭球示意图
    # ============================================================================
    st.subheader("🎨 2. 折射率椭球与相位匹配示意图 (3D)")

    if not valid_modes:
        st.warning("当前没有有效的相位匹配模式，无法进行3D可视化。")
    else:
        # 用户选择显示选项
        col_sel1, col_sel2 = st.columns([1, 1])
        with col_sel1:
            target_mode_3d = st.selectbox("👉 请选择要可视化的模式:", valid_modes, key='mode_3d')
        with col_sel2:
            display_option = st.radio("显示模式:", ["仅基频光 (ω)", "仅倍频光 (2ω)", "两者对比"], index=2, horizontal=True, key='display_opt')

        # 生成3D图按钮
        if st.button("生成3D图", type="secondary", key="btn_3d"):
            # --- 创建3D折射率椭球示意图 (夸大视觉效果) ---
            fig = go.Figure()

            # region 1. 数据获取和缩放系数设置
            # 获取基频光(ω)的折射率
            indices_w = user_config.get_indices()
            n_x_w = indices_w['n_x']
            n_y_w = indices_w['n_y']
            n_z_w = indices_w['n_z']

            # 获取倍频光(2ω)的折射率
            indices_2w = user_config.get_indices(wavelength_nm / 2)
            n_x_2w = indices_2w['n_x']
            n_y_2w = indices_2w['n_y']
            n_z_2w = indices_2w['n_z']

            # === 使用真实折射率值，不进行缩放 ===
            # 直接使用折射率的真实值，这样椭圆截面的长轴和短轴标注就是真实的折射率
            scale_w_x = n_x_w
            scale_w_y = n_y_w
            scale_w_z = n_z_w
            
            scale_2w_x = n_x_2w
            scale_2w_y = n_y_2w
            scale_2w_z = n_z_2w
            # endregion

            # endregion

            # region 2. 生成折射率椭球
            # 创建球坐标系的网格 (theta: 0到π, phi: 0到2π)
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            
            # 生成基频光折射率椭球的坐标 (示意图)
            x_w = scale_w_x * np.outer(np.cos(u), np.sin(v))
            y_w = scale_w_y * np.outer(np.sin(u), np.sin(v))
            z_w = scale_w_z * np.outer(np.ones(np.size(u)), np.cos(v))

            # 生成倍频光折射率椭球的坐标 (示意图)
            x_2w = scale_2w_x * np.outer(np.cos(u), np.sin(v))
            y_2w = scale_2w_y * np.outer(np.sin(u), np.sin(v))
            z_2w = scale_2w_z * np.outer(np.ones(np.size(u)), np.cos(v))

            # 根据用户选择添加椭球
            if display_option in ["仅基频光 (ω)", "两者对比"]:
                # 添加基频光椭球
                fig.add_trace(go.Surface(
                    x=x_w, y=y_w, z=z_w,
                    colorscale=[[0, 'rgb(50, 100, 255)'], [1, 'rgb(100, 150, 255)']],
                    showscale=False,
                    opacity=0.35 if display_option == "两者对比" else 0.75,
                    name=f'基频光 (ω) {wavelength_nm:.1f}nm',
                    hovertemplate='基频光 (ω)<br>n_x=%.4f<br>n_y=%.4f<br>n_z=%.4f<extra></extra>' % (n_x_w, n_y_w, n_z_w),
                    contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                    hidesurface=False
                ))

            if display_option in ["仅倍频光 (2ω)", "两者对比"]:
                # 添加倍频光椭球
                fig.add_trace(go.Surface(
                    x=x_2w, y=y_2w, z=z_2w,
                    colorscale=[[0, 'rgb(255, 80, 80)'], [1, 'rgb(255, 150, 150)']],
                    showscale=False,
                    opacity=0.35 if display_option == "两者对比" else 0.75,
                    name=f'倍频光 (2ω) {wavelength_nm/2:.1f}nm',
                    hovertemplate='倍频光 (2ω)<br>n_x=%.4f<br>n_y=%.4f<br>n_z=%.4f<extra></extra>' % (n_x_2w, n_y_2w, n_z_2w),
                    contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                    hidesurface=False
                ))
            # endregion

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
                name='Z轴(光轴)',
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

            # region 4. 添加k矢量和S矢量
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
                
                # 使用标准球坐标转笛卡尔坐标公式
                k_x = vector_length * np.sin(theta_rad) * np.cos(phi_rad)
                k_y = vector_length * np.sin(theta_rad) * np.sin(phi_rad)
                k_z = vector_length * np.cos(theta_rad)
                
                # 获取实际走离角数值（用于显示）
                walkoff_str = walkoff_dict[target_mode_3d]
                # 从字符串中提取走离角度数（示例："E  (0.1234° / 2.1543 mrad)"）
                import re
                walkoff_match = re.search(r'([+-]?\d+\.\d+)°', walkoff_str)
                if walkoff_match:
                    walkoff_deg = float(walkoff_match.group(1))
                else:
                    walkoff_deg = 0.0
                
                # S 矢量：走离角方向的正确处理（夸大3倍以便观察）
                exaggerated_walkoff_rad = np.deg2rad(walkoff_deg * 3)  # 夸大3倍
                
                # 根据平面确定走离方向
                if user_config.plane in ["XZ", "YZ"]:
                    # XZ或YZ平面：走离角沿Z轴偏离（正值靠近Z轴，即theta变小）
                    s_theta_rad = theta_rad - exaggerated_walkoff_rad  # 注意是减号
                    s_x = vector_length * np.sin(s_theta_rad) * np.cos(phi_rad)
                    s_y = vector_length * np.sin(s_theta_rad) * np.sin(phi_rad)
                    s_z = vector_length * np.cos(s_theta_rad)
                else:  # XY平面
                    # XY平面：走离角沿X轴偏离（正值靠近X轴，即phi变小）
                    s_phi_rad = phi_rad - exaggerated_walkoff_rad  # 注意是减号
                    s_x = vector_length * np.sin(theta_rad) * np.cos(s_phi_rad)
                    s_y = vector_length * np.sin(theta_rad) * np.sin(s_phi_rad)
                    s_z = vector_length * np.cos(theta_rad)
                
                # 绘制 k 矢量 (波矢量) - 金黄色箭头
                fig.add_trace(go.Scatter3d(
                    x=[0, k_x], y=[0, k_y], z=[0, k_z],
                    mode='lines',
                    line=dict(color='gold', width=5),
                    name=f'k矢量 (θ={display_theta:.2f}°, φ={display_phi:.1f}°)',
                    showlegend=True,
                    hovertemplate='k 矢量<br>θ=%.2f°<br>φ=%.1f°<extra></extra>' % (display_theta, display_phi)
                ))
                
                # 使用 Cone 绘制 k 矢量箭头
                fig.add_trace(go.Cone(
                    x=[k_x], y=[k_y], z=[k_z],
                    u=[k_x*0.1], v=[k_y*0.1], w=[k_z*0.1],
                    colorscale=[[0, 'gold'], [1, 'gold']],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.12,
                    name='k矢量箭头',
                    showlegend=False
                ))
                
                # 在k矢量旁边添加标注
                fig.add_trace(go.Scatter3d(
                    x=[k_x*1.15], y=[k_y*1.15], z=[k_z*1.15],
                    mode='text',
                    text=['k'],
                    textfont=dict(size=16, color='gold', family='Arial Black'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # 绘制 S 矢量 (能量流/坡印廷矢量) - 深橙色实线箭头
                fig.add_trace(go.Scatter3d(
                    x=[0, s_x], y=[0, s_y], z=[0, s_z],
                    mode='lines',
                    line=dict(color='darkorange', width=5),
                    name=f'S矢量 (能量流)',
                    showlegend=True,
                    hovertemplate='S 矢量 (能量流)<br>实际走离角=%.4f°<extra></extra>' % walkoff_deg
                ))
                
                # 使用 Cone 绘制 S 矢量箭头
                fig.add_trace(go.Cone(
                    x=[s_x], y=[s_y], z=[s_z],
                    u=[s_x*0.1], v=[s_y*0.1], w=[s_z*0.1],
                    colorscale=[[0, 'darkorange'], [1, 'darkorange']],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.12,
                    name='S矢量箭头',
                    showlegend=False
                ))
                
                # 在S矢量旁边添加标注
                fig.add_trace(go.Scatter3d(
                    x=[s_x*1.15], y=[s_y*1.15], z=[s_z*1.15],
                    mode='text',
                    text=['S'],
                    textfont=dict(size=16, color='darkorange', family='Arial Black'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            # endregion
                
            # endregion
                
                # region 5. 添加角度标注 (走离角、theta角、phi角)
                # === 用弧线标注走离角（k矢量和S矢量之间的角度）===
                # 归一化k和S方向
                k_norm = np.array([k_x, k_y, k_z]) / np.linalg.norm([k_x, k_y, k_z])
                s_norm = np.array([s_x, s_y, s_z]) / np.linalg.norm([s_x, s_y, s_z])
                
                # 计算从k到S的弧线（使用球面线性插值）
                arc_radius_walkoff = 1.5  # 弧线半径
                n_points_walkoff = 25
                
                # 使用球面线性插值生成k到S之间的弧线点
                walkoff_arc_x = []
                walkoff_arc_y = []
                walkoff_arc_z = []
                
                for i in range(n_points_walkoff):
                    t = i / (n_points_walkoff - 1)
                    # 球面线性插值 (slerp)
                    theta_interp = np.arccos(np.dot(k_norm, s_norm))
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
                
                fig.add_trace(go.Scatter3d(
                    x=walkoff_arc_x, y=walkoff_arc_y, z=walkoff_arc_z,
                    mode='lines',
                    line=dict(color='darkred', width=3),
                    name='走离角弧线',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # 走离角标注文字位置（弧线中点）
                mid_direction = (k_norm + s_norm) / 2
                mid_direction = mid_direction / np.linalg.norm(mid_direction)
                text_x = mid_direction[0] * 1.8
                text_y = mid_direction[1] * 1.8
                text_z = mid_direction[2] * 1.8
                
                fig.add_trace(go.Scatter3d(
                    x=[text_x], y=[text_y], z=[text_z],
                    mode='text',
                    text=[f'走离角<br>{walkoff_deg:.4f}°'],
                    textfont=dict(size=10, color='darkred', family='Arial Black'),
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
                k_proj_x = k_x
                k_proj_y = k_y
                k_proj_z = 0
                
                # 从k矢量到其投影的虚线
                fig.add_trace(go.Scatter3d(
                    x=[k_x, k_proj_x], y=[k_y, k_proj_y], z=[k_z, k_proj_z],
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
                
                # endregion
                
                # region 6. 绘制晶体长方体
                # === 绘制晶体长方体（端面垂直于k矢量）===
                # k矢量方向的单位向量
                k_unit = np.array([k_x, k_y, k_z]) / np.linalg.norm([k_x, k_y, k_z])
                
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
                
                # endregion
                
                # region 7. 绘制截面椭圆
                # === 绘制垂直于k矢量的截面与折射率椭球的交线（椭圆）===
                # 截面位置在原点（晶体后端面）
                cross_section_center = np.array([0.0, 0.0, 0.0])
                
                # 在截面上绘制折射率椭球的交线（椭圆）
                n_ellipse_points = 150
                angles = np.linspace(0, 2*np.pi, n_ellipse_points)
                
                # 根据选择的显示模式确定要绘制的椭圆（使用缩放后的尺寸）
                ellipses_to_draw = []
                if display_option in ["仅基频光 (ω)", "两者对比"]:
                    ellipses_to_draw.append(('基频光', scale_w_x, scale_w_y, scale_w_z, 'rgba(0, 0, 139, 0.4)', 6))
                if display_option in ["仅倍频光 (2ω)", "两者对比"]:
                    ellipses_to_draw.append(('倍频光', scale_2w_x, scale_2w_y, scale_2w_z, 'rgba(139, 0, 0, 0.4)', 6))
                
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
                        axis_color = 'rgb(0, 0, 139)' if label == '基频光' else 'rgb(139, 0, 0)'
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
                        
                        # 标注长轴值（倍频光距离更远以避免重叠）
                        distance_factor_a = 1.3 if label == '倍频光' else 1.1
                        major_label_pos = major_point * distance_factor_a
                        # 添加偏移量：沿k_unit方向偏移0.15
                        offset_a = k_unit * 0.15
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
                        
                        # 标注短轴值（b标注更靠近椭圆）
                        distance_factor_b = 1.1 if label == '倍频光' else 1.02
                        minor_label_pos = minor_point_neg * distance_factor_b
                        # 添加偏移量：沿k_unit反方向偏移0.15
                        offset_b = -k_unit * 0.15
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

                # endregion

            # region 8. 设置图形布局和保存
            # 设置 3D 场景的基本外观
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
                    text=f'{crystal_name} 晶体折射率椭球示意图<br><sub>基频光 λ={wavelength_nm:.1f}nm (蓝色) | 倍频光 λ={wavelength_nm/2:.1f}nm (红色) | 相位匹配模式: {target_mode_3d} | X,Y,Z为晶体光学主轴</sub>',
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
                'n_x_w': n_x_w, 'n_y_w': n_y_w, 'n_z_w': n_z_w,
                'n_x_2w': n_x_2w, 'n_y_2w': n_y_2w, 'n_z_2w': n_z_2w,
                'wavelength_nm': wavelength_nm
            }
            # endregion
        
        # 显示保存的3D图
    if '3d_fig' in st.session_state:
        st.plotly_chart(st.session_state['3d_fig'], use_container_width=True)
        
        # 添加说明
        st.caption("**说明**: k矢量为波矢方向，S矢量为能量流方向。走离角为了方便展示，并没有显示实际角度。长方体为晶体示意，端面垂直于k矢量方向。截面椭圆表示垂直于k矢量方向的折射率分布。")
        
        # 显示折射率数值信息和差异
        config = st.session_state['3d_config']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**基频光 ({config['wavelength_nm']:.1f} nm)**")
            st.write(f"n_x = {config['n_x_w']:.5f}")
            st.write(f"n_y = {config['n_y_w']:.5f}")
            st.write(f"n_z = {config['n_z_w']:.5f}")
        with col2:
            st.error(f"**倍频光 ({config['wavelength_nm']/2:.1f} nm)**")
            st.write(f"n_x = {config['n_x_2w']:.5f}")
            st.write(f"n_y = {config['n_y_2w']:.5f}")
            st.write(f"n_z = {config['n_z_2w']:.5f}")
        with col3:
            st.warning("**折射率差异 Δn**")
            st.write(f"Δn_x = {abs(config['n_x_2w'] - config['n_x_w']):.5f}")
            st.write(f"Δn_y = {abs(config['n_y_2w'] - config['n_y_w']):.5f}")
            st.write(f"Δn_z = {abs(config['n_z_2w'] - config['n_z_w']):.5f}")

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
        
        # 使用 Tabs 分开三个维度的分析
        tab1, tab2, tab3 = st.tabs(["角度带宽", "波长带宽", "温度带宽"])
        
        # --- Tab 1: 角度带宽 ---
        with tab1:
            if st.button("计算角度带宽", key="btn_ang"):

                # 调用修改后的 Solver 函数，传入 sidebar 设置的 scan_res_angle 和 scan_step_angle
                fig, val_mrad, val_deg = simulation.acceptance_angle(
                    theta_dict, target_mode_bandwidth, step=scan_step_angle, res=scan_res_angle
                )
                # 存储结果到 session state，防止切换 Tab 时丢失
                st.session_state['res_angle_fig'] = fig
                st.session_state['res_angle_val_mrad'] = val_mrad
                st.session_state['res_angle_val_deg'] = val_deg

            if 'res_angle_fig' in st.session_state:
                c1, c2 = st.columns([3, 1])
                with c1: st.pyplot(st.session_state['res_angle_fig']) # 显示图表
                with c2: 
                    st.success(f"**角度带宽 (FWHM)**")
                    st.metric("mrad·cm", f"{st.session_state['res_angle_val_mrad']:.4f}")
                    st.metric("deg·cm", f"{st.session_state['res_angle_val_deg']:.4f}")

        # --- Tab 2: 波长带宽 ---
        with tab2:
            if st.button("计算波长带宽", key="btn_wav"):
                fig, val_nm, val_ghz = simulation.acceptance_wavelength(
                    theta_dict, target_mode_bandwidth, step=scan_step_wave, res=scan_res_wave
                )
                
                st.session_state['res_wave_fig'] = fig
                st.session_state['res_wave_val_nm'] = val_nm
                st.session_state['res_wave_val_ghz'] = val_ghz

            if 'res_wave_fig' in st.session_state:
                c1, c2 = st.columns([3, 1])
                with c1: st.pyplot(st.session_state['res_wave_fig'])
                with c2: 
                    st.info(f"**波长带宽 (FWHM)**")
                    st.metric("nm·cm", f"{st.session_state['res_wave_val_nm']:.4f}")
                    st.metric("GHz·cm", f"{st.session_state['res_wave_val_ghz']:.4f}")

        # --- Tab 3: 温度带宽 ---
        with tab3:
            if st.button("计算温度带宽", key="btn_tem"):
                fig, val_temp = simulation.acceptance_temperature(
                    theta_dict, target_mode_bandwidth, step=scan_step_temp, res=scan_res_temp
                )
                
                st.session_state['res_temp_fig'] = fig
                st.session_state['res_temp_val_temp'] = val_temp
                
            if 'res_temp_fig' in st.session_state:
                c1, c2 = st.columns([3, 1])
                with c1: st.pyplot(st.session_state['res_temp_fig'])
                with c2: 
                    st.warning(f"**温度带宽 (FWHM)**")

                    st.metric("°C·cm", f"{st.session_state['res_temp_val_temp']:.4f}")
