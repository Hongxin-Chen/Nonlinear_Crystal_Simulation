"""
非线性晶体二谐波产生(SHG)相位匹配模拟器

该模块用于计算非线性光学晶体(如CLBO、LBO等)的相位匹配角度和接受角。
主要功能包括:
   - 计算不同相位匹配类型(Type I, Type II)的临界角度
    - 计算光束偏离角(Walk-off Angle)
    - 计算相位匹配接受角(Angle Acceptance)
    - 计算相位匹配接受波长(Wavelength Acceptance)
    - 计算相位匹配接受温度(Temperature Acceptance)
主要类:
    - Solver: 封装了所有计算功能的核心类
主要函数:
    - neff_func: 计算单轴晶体中的有效折射率
    - criticalangle: 计算相位匹配的临界角度
    - walkoff_angle: 计算光束偏离角
    - acceptance_angle: 计算相位匹配接受角
    - acceptance_wavelength: 计算相位匹配接受波长
    - acceptance_temperature: 计算相位匹配接受温度
该模块依赖于 numpy、matplotlib 和 scipy 库进行数值计算和绘图。

作者：陈泓鑫
"""
import numpy as np
import matplotlib.pyplot as plt
import configuration
from configuration import SimulationConfig
from scipy.optimize import fsolve
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go

class Solver():
    """
    非线性晶体相位匹配求解器
    
    该类封装了与晶体相位匹配相关的所有计算，包括折射率有效值计算、
    相位匹配角度求解以及接受角分析。
    
    属性:
        cfg (SimulationConfig): 用户配置参数，包含晶体类型、波长、温度等
        indices (dict): 基频光在各个方向上的折射率 {n_x, n_y, n_z}
        indices2w (dict): 倍频光(二次谐波)在各个方向上的折射率
        plane_config (dict): 定义不同平面(XZ, YZ, XY)中轴的对应关系
        key_static (str): 在所选平面中不变的轴对应的折射率键
        key_cos (str): 在所选平面中与cos²θ相关的轴对应的折射率键
        key_sin (str): 在所选平面中与sin²θ相关的轴对应的折射率键
            major_axis (float): 主要轴的折射率(与cos²θ相关)
            minor_axis (float): 次要轴的折射率(与sin²θ相关)
        nw_static, nw_eff_func: 基频光的静态折射率和有效折射率函数
        n2w_static, n2w_eff_func: 倍频光的静态折射率和有效折射率函数
        equations_deltan (dict): 四种相位匹配模式对应的Δn方程
    """
    def __init__(self, config):
        """
        初始化求解器，加载晶体配置和折射率数据
        
        参数:
            config (SimulationConfig): 用户配置对象，包含晶体参数、波长、温度等信息
        
        初始化步骤:
            1. 从配置中获取基频光和倍频光的折射率张量
            2. 根据所选平面(XZ/YZ/XY)确定不动轴和变动轴的对应关系
            3. 构建四种相位匹配模式的Δn方程(目标是求解使Δk=0的角度)
        """
        # 保存配置参数
        self.cfg = config
        self.crystal_db = config.crystal_db
        # 获取基频光(ω)在三个方向的折射率 n_x, n_y, n_z
        # 这些数据从晶体的色散方程中计算得出(参考 getusers.py)
        self.indices = self.cfg.get_indices()
        
        # 获取倍频光(2ω)的折射率，波长为基频光的一半
        # 由于色散效应，高频光的折射率会不同
        self.indices2w = self.cfg.get_indices(self.cfg.wavelength_nm / 2)

        # 根据所选的非共线平面定义轴的对应关系
        # 平面定义: (不动轴, cos²轴, sin²轴)
        # 在计算有效折射率时，某一轴不随角度变化，另外两个轴的贡献通过cos²θ和sin²θ混合
        self.plane_config = {
            "XZ": ('n_y', 'n_z', 'n_x'), # XZ平面：Y轴方向不变，角度在Z-X平面内扫描
            "YZ": ('n_x', 'n_z', 'n_y'), # YZ平面：X轴方向不变，角度在Z-Y平面内扫描
            "XY": ('n_z', 'n_x', 'n_y')  # XY平面：Z轴方向不变，角度在X-Y平面内扫描
        }
        
        # 根据所选平面获取对应的轴信息
        self.key_static, self.key_cos, self.key_sin = self.plane_config[self.cfg.plane]

        # 定义主轴和次轴的折射率
        self.major_axis = self.indices[self.key_cos]
        self.minor_axis = self.indices[self.key_sin]
        self.major_axis2w = self.indices2w[self.key_cos]
        self.minor_axis2w = self.indices2w[self.key_sin]

        # ===== 基频光(Fundamental wave, ω) 的折射率 =====
        # 在选定平面内，有一个轴方向的折射率保持不变(o光或e光)
        self.nw_static = self.indices[self.key_static]
        # 有效折射率是另外两个方向折射率的函数，随扫描角度θ变化
        self.nw_eff_func = self.neff_func(self.major_axis, self.minor_axis)
        
        # ===== 倍频光(Second Harmonic wave, 2ω) 的折射率 =====
        # 倍频光的折射率同样需要分解为静态部分和角度相关的有效折射率
        self.n2w_static = self.indices2w[self.key_static]
        self.n2w_eff_func = self.neff_func(self.major_axis2w, self.minor_axis2w)

        # region ===== 定义四种相位匹配类型的Δk方程 =====
        # 相位匹配条件: Δk = k₂ω - 2k_ω = 0，即 n₂ω = 2n_ω (无量纲形式)
        # 
        # Type I 相位匹配(o+o->e 或 e+e->o):
        #   - 基频光两个光子偏振相同
        #   - 满足条件: n₂ω(Type I) = 2n_ω(avg)
        #
        # Type II 相位匹配(o+e->e 或 o+e->o):
        #   - 基频光两个光子偏振不同
        #   - 需要使用平均折射率: n_ω(avg) = (n_o + n_e) / 2
        #
        # 每个方程是关于扫描角θ的函数，求解使该函数=0的θ值
        # endregion
                    
        self.equations_deltan = {
            "O + O -> E (Type I) ": lambda theta: self.nw_static - self.n2w_eff_func(theta),
            "E + E -> O (Type I) ": lambda theta: self.nw_eff_func(theta) - self.n2w_static,
            "O + E -> E (Type II)": lambda theta: 0.5 * (self.nw_static + self.nw_eff_func(theta)) - self.n2w_eff_func(theta),
            "O + E -> O (Type II)": lambda theta: 0.5 * (self.nw_static + self.nw_eff_func(theta)) - self.n2w_static
                     }

    def neff_func(self, n_cos, n_sin):
        """
        计算单轴晶体中的有效折射率
        
        对于单轴晶体中的角度相关传播，有效折射率由两个主折射率通过椭球方程混合计算。
        这是晶体光学中的基本公式。
        
        参数:
            n_cos (float): 与cos²θ相关联的折射率(通常为e光的折射率或n_max)
            n_sin (float): 与sin²θ相关联的折射率(通常为o光的折射率或n_min)
        
        返回:
            function: 返回一个关于角度θ的函数 n_eff(θ)
        
        公式推导 (单轴晶体椭球方程):
            1/n_eff²(θ) = cos²(θ)/n_cos² + sin²(θ)/n_sin²
            
            求解得: n_eff(θ) = √[ (n_cos² * n_sin²) / (n_cos² * cos²θ + n_sin² * sin²θ) ]
        
        物理意义:
            - 当θ=0°时,n_eff = n_cos (纯o光)
            - 当θ=90°时,n_eff = n_sin (纯e光)
            - 中间值通过椭球插值计算
        
        应用:
            在非线性光学中，通过改变传播方向(扫描θ)来改变有效折射率，
            从而调整相位匹配条件
        """
        return lambda theta: np.sqrt(
            (n_cos**2 * n_sin**2) / 
            (n_cos**2 * np.cos(theta)**2 + n_sin**2 * np.sin(theta)**2)
        )

    def criticalangle(self):
        """
        计算相位匹配的临界角度
        
        该方法是核心计算函数,用于求解相位匹配条件Δk=0下的光线传播角度。
        对于给定的晶体、波长和温度，存在特定的传播角度使得二次谐波产生效率最高。
        
        工作流程:
            1. 定义robust的非线性方程求解器 (robust_solve)
            2. 遍历所有四种相位匹配模式
            3. 对每种模式求解相应的Δk方程
            4. 进行物理合理性检验（角度范围、残差大小）
            5. 输出并返回所有结果
        
        返回值:
            dict: 四种匹配模式及其对应的相位匹配角度(单位:度)
                  例: {'O + O -> E (Type I) ': 23.5, 'E + E -> O (Type I) ': nan, ...}
                  无解时值为 nan
        """
        
        # ===== 内部求解函数: robust_solve =====
        def robust_solve(equation_func, guess=np.pi/4):
            """
            数值求解器：尝试找到方程的根，失败或无解时返回 np.nan
            
            参数:
                equation_func: 目标方程 f(θ)，当 f(θ)=0 时满足相位匹配
                guess: 初始猜测值,默认45°(π/4弧度)，这是比较合理的起点
            
            返回:
                float: 求解得到的角度(弧度)，或 np.nan(无解)
            
            鲁棒性保证:
                1. fsolve 返回信息 ier=1 表示成功收敛
                2. 解必须在物理范围 [0°, 90°] = [0, π/2] 内
                3. 将解代回原方程验证，残差 |f(θ_solution)| < 1e-4
                4. 严格检验防止伪收敛和数值不稳定
            
            参数说明:
                - full_output=1: 让 fsolve 返回详细信息，包括收敛标志 ier
                - ier=1: 收敛成功
                - ier≠1: 求解失败或不收敛
            """
            # 调用 scipy 的 fsolve 非线性方程求解器
            # full_output=1 可以获得收敛信息
            root, _, ier, _ = fsolve(equation_func, guess, full_output=1)      
                     
            if ier == 1:
                theta_res = root[0]
                # ===== 双重检查 =====
                # 检验 1: 解必须在物理范围内 (0° 到 90°)
                if 0 <= theta_res <= np.pi/2:
                    # 检验 2: 把解代回方程，计算残差
                    # 如果残差太大，说明是伪收敛，应该舍弃
                    residual = abs(equation_func(theta_res))
                    if residual < 1e-4:
                        return theta_res
            
            # 如果求解失败或未通过检验，返回 NaN
            return np.nan

        # ===== 主计算流程 =====

        # 用字典存储所有四种模式的求解结果
        theta_critical_dict_results = {}

        # 遍历四种相位匹配模式
        for mode_name, eq_func in self.equations_deltan.items():
            # 调用求解器，初始猜测为 45°
            theta_val = robust_solve(eq_func, guess=np.pi/4)
            
            # 将结果从弧度转换为度数，便于用户理解
            theta_deg = np.rad2deg(theta_val) if not np.isnan(theta_val) else np.nan
            
            # 保存结果用于后续计算
            theta_critical_dict_results[mode_name] = theta_deg
                
        # 返回包含四个结果的字典，方便后续画图使用
        return theta_critical_dict_results

    def walkoff_angle(self, theta_critical_dict):
        """
        计算和显示非线性晶体的光束偏离角(Walk-off Angle)
        
        光束偏离角是指在双折射晶体中，由于不同偏振态的光速不同，
        导致能量传播方向与波矢方向不一致，从而产生的偏离现象。
        
        工作流程:
            1. 遍历所有四种相位匹配模式
            2. 对每种模式，使用临界角计算光束偏离角
            3. 输出结果，显示每个光子偏振态的偏离角

            值得注意的是，走离角的正负意义取决于晶体的具体光学轴方向和入射角度，
            正值意味着靠近major axis方向偏离,负值则相反.
        """

        walkoff_angle_results = {
            "O + O -> E (Type I) ": lambda theta: self.nw_static - self.n2w_eff_func(theta),
            "E + E -> O (Type I) ": lambda theta: self.nw_eff_func(theta) - self.n2w_static,
            "O + E -> E (Type II)": lambda theta: 0.5 * (self.nw_static + self.nw_eff_func(theta)) - self.n2w_eff_func(theta),
            "O + E -> O (Type II)": lambda theta: 0.5 * (self.nw_static + self.nw_eff_func(theta)) - self.n2w_static
                     }

        for mode_name, theta_deg in theta_critical_dict.items():
            if np.isnan(theta_deg):
                walkoff_angle_results[mode_name] = np.nan
            else:
               
                theta_rad = np.deg2rad(theta_deg)

                # 计算光束偏离角公式
                # 根据椭球方程，tan(θ_s) = (a² / b²) * tan(θ_k)

                phow = theta_deg - np.rad2deg(np.arctan(self.major_axis**2 / self.minor_axis**2 * np.tan(theta_rad)))
                pho2w = theta_deg - np.rad2deg(np.arctan(self.major_axis2w**2 / self.minor_axis2w**2 * np.tan(theta_rad)))
                phow_rad = np.deg2rad(phow) * 1e3  # 转换为毫弧度
                pho2w_rad = np.deg2rad(pho2w) * 1e3  # 转换为毫弧度

                if "O + O -> E" in mode_name:
                 walkoff_angle_results[mode_name] = " O  (0°)  |  O  (0°)  |  E  ({:.4f}° / {:.4f} mrad)".format(pho2w, pho2w_rad)
                elif "E + E -> O" in mode_name:
                 walkoff_angle_results[mode_name] = " E  ({:.4f}° / {:.4f} mrad)  |  E  ({:.4f}° / {:.4f} mrad)  |  O  (0°)".format(phow, phow_rad, phow, phow_rad)
                elif "O + E -> E" in mode_name:
                 walkoff_angle_results[mode_name] = " O  (0°)  |  E  ({:.4f}° / {:.4f} mrad)  |  E  ({:.4f}° / {:.4f} mrad)".format(phow, phow_rad, pho2w, pho2w_rad)
                elif "O + E -> O" in mode_name:
                 walkoff_angle_results[mode_name] = " O  (0°)  |  E  ({:.4f}° / {:.4f} mrad)  |  O  (0°)".format(phow, phow_rad)
        return walkoff_angle_results

    def d_eff(self, theta_critical_dict, selected_phi= None):
        """
        计算非线性晶体的有效非线性系数 d_eff
        
        有效非线性系数 d_eff 是描述非线性光学过程强度的关键参数。它将晶体的
        非线性张量与具体的相位匹配几何构型结合，反映了在特定传播方向和偏振
        组合下，晶体的实际二次谐波转换效率。
        
        工作流程:
            1. 从晶体数据库中获取非线性系数张量 d
            2. 根据所选平面确定角度参数 (theta, phi) 的对应关系
            3. 根据晶体点群对称性选择相应的 d_eff 计算公式
            4. 对四种相位匹配模式分别计算 d_eff 值
            5. 返回包含所有模式的 d_eff 结果字典
        
        参数:
            theta_critical_dict (dict): 由 criticalangle() 返回的相位匹配角度字典
            selected_phi (float, optional): 用户指定的方位角 φ (度)，默认为 None
        
        返回:
            dict: 四种相位匹配模式及其对应的 d_eff 值(单位: pm/V)
                  例: {'O + O -> E (Type I) ': 0.85, 'E + E -> O (Type I) ': nan, ...}
        
        物理原理:
            - 非线性极化强度: P^(2ω) = ε₀ * d_eff * E^(ω) * E^(ω)
            - d_eff 取决于:
                1) 晶体的非线性张量分量 (d₁₁, d₁₅, d₃₁, d₃₂, d₃₃, d₃₆ 等)
                2) 传播方向角度 (θ, φ)
                3) 偏振组合类型 (Type I 或 Type II)
            - 不同晶体点群有不同的计算公式:
                * 4̄2m 群 (如 BBO): 依赖 d₃₆ 和三角函数组合
                * 3m 群 (如 LiNbO₃): 依赖 d₁₁, d₂₂, d₃₁ 等多个分量
                * mm2 群 (如 LBO, KTP): 分平面讨论，XY/YZ/XZ 有不同公式
        
        角度参数处理逻辑:
            根据所选平面，theta 和 phi 的含义不同:
            - XY 平面: theta 固定为 90°，扫描角对应 phi
            - XZ 平面: phi 固定为 0°，扫描角对应 theta
            - YZ 平面: phi 固定为 90°，扫描角对应 theta
            这样可以统一处理不同平面的几何关系。
        
        应用:
            d_eff 越大，二次谐波转换效率越高。在设计非线性光学系统时，
            需要综合考虑 d_eff 大小、相位匹配条件和其他接受角等因素。
        """
        # 获取晶体的非线性系数张量
        crystal_info = self.crystal_db[self.cfg.crystal_name]
        if not crystal_info: return 0.0
        d_tensor = crystal_info["d"]

        angle_dict_phi = {"O + O -> E (Type I) ": np.deg2rad(selected_phi),
                          "E + E -> O (Type I) ": np.deg2rad(selected_phi),
                          "O + E -> E (Type II)": np.deg2rad(selected_phi),
                          "O + E -> O (Type II)": np.deg2rad(selected_phi)}
        
        if self.cfg.plane == "XY":
            # 在 XY 平面，theta 固定 90度，phi = 扫描角
            theta_vals = angle_dict_phi
            phi_vals = theta_critical_dict
        
        elif self.cfg.plane == "XZ":
            # 在 XZ 平面，phi 固定 0度，theta = 扫描角
            theta_vals = theta_critical_dict
            phi_vals = angle_dict_phi
            
        elif self.cfg.plane == "YZ":
            # 在 YZ 平面，phi 固定 90度，theta = 扫描角
            theta_vals = theta_critical_dict
            phi_vals = angle_dict_phi


        if crystal_info["group"] == "4bar2m":  # 例如 BBO

            d_eff_dict = {"O + O -> E (Type I) ": lambda theta,phi: d_tensor['d36'] * np.sin(theta) * np.sin(2*phi),
                          "E + E -> O (Type I) ": lambda theta,phi: d_tensor['d36'] * np.sin(2*theta) * np.cos(2*phi),
                          "O + E -> E (Type II)": lambda theta,phi: d_tensor['d36'] * np.sin(2*theta) * np.cos(2*phi),
                          "O + E -> O (Type II)": lambda theta,phi: d_tensor['d36'] * np.sin(theta) * np.sin(2*phi)}
            
            for mode, theta_deg in theta_vals.items():
                phi = phi_vals[mode]
                d_eff_dict[mode] = d_eff_dict[mode](np.deg2rad(theta_deg), phi)
        
        elif crystal_info["group"] == "3m":  # 例如 BBO

            d_eff_dict = {"O + O -> E (Type I) ": lambda theta,phi: d_tensor['d31'] * np.sin(theta) + (d_tensor['d11'] * np.cos(3*phi)-d_tensor['d22']*np.sin(3*phi)) * np.cos(theta),
                          "E + E -> O (Type I) ": lambda theta,phi: d_tensor['d31'] * np.sin(theta) + (d_tensor['d22'] * np.sin(3*phi)-d_tensor['d11']*np.cos(3*phi)) * np.cos(theta),
                          "O + E -> E (Type II)": lambda theta,phi: (d_tensor['d11'] * np.sin(3*phi) + d_tensor['d22'] * np.cos(3*phi)) * np.cos(theta)**2,
                          "O + E -> O (Type II)": lambda theta,phi: d_tensor['d15'] * np.sin(theta) + (d_tensor['d11'] * np.cos(3*phi) - d_tensor['d22'] * np.sin(3*phi)) * np.cos(theta)}
        
            for mode, theta_deg in theta_vals.items():
                phi = phi_vals[mode]
                d_eff_dict[mode] = d_eff_dict[mode](np.deg2rad(theta_deg), phi)
                
        elif crystal_info["group"] == "mm2":  # 例如 LBO, KTP
            # 1. 获取基础系数 (注意单位一致性)
            d31 = d_tensor.get('d31', 0)
            d32 = d_tensor.get('d32', 0)
            d33 = d_tensor.get('d33', 0) # 很少用
            d15 = d_tensor.get('d15', 0) # 也就是 d31 (Kleinman)
            d24 = d_tensor.get('d24', 0) # 也就是 d32 (Kleinman)

            # 2. 根据平面“分而治之”
            # mm2 的特点是：不同平面公式结构完全不同
            
            if self.cfg.plane == "XY":
                # === XY 平面 (Theta=90, 扫描 Phi) ===
                # 这是 LBO 最经典的 NCPM 平面
                # 特点：公式依赖 Phi，Type II 通常为 0
                
                d_eff_dict = {
                    # Type I: d31*cos²φ + d32*sin²φ (也有文献定义相反，视XY轴定义而定，工程上通常取绝对值大的)
                    "O + O -> E (Type I) ": lambda t, p: d31 * (np.cos(p)**2) + d32 * (np.sin(p)**2),
                    "E + E -> O (Type I) ": lambda t, p: d33, # 极小，几乎不用
                    
                    # XY平面内 Type II 无效 (对称性抵消)
                    "O + E -> E (Type II)": lambda t, p: 0.0,
                    "O + E -> O (Type II)": lambda t, p: 0.0
                }

            elif self.cfg.plane == "YZ":
                # === YZ 平面 (Phi=90, 扫描 Theta) ===
                # 特点：公式依赖 Theta
                
                d_eff_dict = {
                    # Type I (o+o->e): d31 * cos(theta)
                    "O + O -> E (Type I) ": lambda t, p: d31 * np.cos(t),
                    "E + E -> O (Type I) ": lambda t, p: 0.0,
                    
                    # Type II (o+e->e): d31 * sin(theta) (或者是 d15)
                    "O + E -> E (Type II)": lambda t, p: d31 * np.sin(t),
                    "O + E -> O (Type II)": lambda t, p: d31 * np.sin(t)
                }

            elif self.cfg.plane == "XZ":
                # === XZ 平面 (Phi=0, 扫描 Theta) ===
                # 特点：公式依赖 Theta，但系数变成了 d32
                
                d_eff_dict = {
                    # Type I (o+o->e): d32 * cos(theta)
                    "O + O -> E (Type I) ": lambda t, p: d32 * np.cos(t),
                    "E + E -> O (Type I) ": lambda t, p: 0.0,
                    
                    # Type II (o+e->e): d32 * sin(theta) (或者是 d24)
                    "O + E -> E (Type II)": lambda t, p: d32 * np.sin(t),
                    "O + E -> O (Type II)": lambda t, p: d32 * np.sin(t)
                }

            else:
                # 其它任意切面的通用公式过于复杂且工程不常用
                # 返回 0 防止报错
                d_eff_dict = {
                    "O + O -> E (Type I) ": lambda t, p: 0.0,
                    "E + E -> O (Type I) ": lambda t, p: 0.0,
                    "O + E -> E (Type II)": lambda t, p: 0.0,
                    "O + E -> O (Type II)": lambda t, p: 0.0
                }

            # 3. 执行计算 (复用之前的逻辑)
            for mode, theta_deg in theta_vals.items():
                phi = phi_vals[mode]
                # 计算并取绝对值 (d_eff 只看大小)
                val = d_eff_dict[mode](np.deg2rad(theta_deg), phi)
                d_eff_dict[mode] = abs(val)
        
        return d_eff_dict

    def acceptance_angle(self, theta_critical_dict, target_mode, step=1000, res=0.1):

        """
        计算和可视化相位匹配接受角(相位匹配角带宽)
        
        相位匹配接受角是指在相位匹配条件附近，光转换效率仍能保持在一定水平
        (通常为最大值的50%)时的角度偏差范围。这是判断晶体实用性的重要参数。
        
        工作流程:
            1. 与用户交互，获取扫描精度、步数和匹配模式
            2. 在临界角附近进行微小角度扫描
            3. 计算每个角度的相位失配 Δk 和转换效率
            4. 绘制接受角曲线
            5. 计算半高全宽(FWHM)对应的接受角
        
        参数:
            theta_critical_dict (dict): 由 criticalangle() 返回的相位匹配角度字典
        
        返回值:
            tuple: (接受角_毫弧度, 接受角_度)
        
        物理原理:
            - 相位失配: Δk = 4π/λ * f(θ)  其中f(θ)是相位匹配方程的偏离
            - 转换效率: η ∝ sinc²(Δk * L/2)  其中L是晶体长度(注意代码中我直接设置L=1cm, 方便计算,实际值将该值除以L即可)
            - sinc函数在Δk=0时最大,随|Δk|增大而衰减
        """
        # ===== 构建角度扫描数组 =====
        # 获取所选模式的相位匹配方程
        equation_func = self.equations_deltan[target_mode]
        
        # 以临界角为中心，前后各扫描 step 个点
        # 单位变换: mrad × 1e-3 = rad
        theta_axis = np.deg2rad(theta_critical_dict[target_mode]) + np.arange(-step, step) * res * 1e-3 
       
        # ===== 计算相位失配和转换效率 =====

        delta_k_angle = (np.pi * 4 / self.cfg.wavelength_um) * equation_func(theta_axis)
        
        # 转换效率: η(Δk) = sinc²(Δk × L/2)
        # sinc(x) = sin(x)/x
        efficiency_angle = (np.sinc(delta_k_angle * 1e4 / (2 * np.pi)))**2

        # ===== 绘制接受角曲线 =====
        fig,ax = plt.subplots(figsize=(10, 6))
        ax.plot(theta_axis * 1000, efficiency_angle, 'r-', linewidth=1.5)
        ax.set_xlabel('Angle Deviation / mrad', fontsize=12)  # X轴: 角度偏差(毫弧度)
        ax.set_ylabel('SHG Efficiency', fontsize=12)          # Y轴: 二次谐波转换效率
        ax.set_title(f'Acceptance Angle Curve for {self.cfg.crystal_name} ({target_mode})', fontsize=14)
        ax.grid(True, alpha=0.3)

        # ===== 计算接受角(FWHM, 半高全宽) =====
        # FWHM 定义: 效率降到最大值50%时的角度范围
        half_max = 0.5
        
        # 找出所有效率≥50%的点
        indices_above_half = np.where(efficiency_angle >= half_max)[0]
        
        if len(indices_above_half) > 0:
            # 最小角度对应的索引(左边界)
            lower_index = indices_above_half[0]
            # 最大角度对应的索引(右边界)
            upper_index = indices_above_half[-1]
            
            # 计算接受角(毫弧度)
            acceptance_angle = (theta_axis[upper_index] - theta_axis[lower_index]) * 1000
            
            # 转换为度数便于理解
            acceptance_angle_deg = np.rad2deg(theta_axis[upper_index] - theta_axis[lower_index])



        return fig, acceptance_angle, acceptance_angle_deg

    def acceptance_wavelength(self, theta_critical_dict, target_mode, step, res):
        """
        计算和可视化相位匹配接受波长(相位匹配波长带宽)
        
        相位匹配接受波长是指在相位匹配条件附近，光转换效率仍能保持在一定水平
        (通常为最大值的50%)时的波长偏差范围。这是判断晶体实用性的重要参数。
        
        工作流程:
            1. 与用户交互，获取扫描精度、步数和匹配模式
            2. 在临界波长附近进行微小波长扫描
            3. 计算每个波长的相位失配 Δk 和转换效率
            4. 绘制接受波长曲线
            5. 计算半高全宽(FWHM)对应的接受波长
        """
        
        # 以临界波长为中心，前后各扫描 step 个点
        wavelength_axis = self.cfg.wavelength_nm + np.arange(-step, step) * res 

        # ===== 对每个波长计算折射率和相位失配 =====
        # 获取基频光在所有波长下的折射率 (温度固定)
        tem_indices = self.cfg.get_indices(target_wavelength=wavelength_axis, target_temperature=self.cfg.temperature)
        # 获取倍频光在所有波长下的折射率 (波长为基频光的一半，温度固定)
        tem_indices2w = self.cfg.get_indices(target_wavelength=wavelength_axis / 2, target_temperature=self.cfg.temperature)

        # 在选定平面内，有一个轴方向的折射率保持不变(o光或e光)
        tem_nw_static = tem_indices[self.key_static]
        # 有效折射率是另外两个方向折射率的函数，随扫描角度θ变化
        tem_nw_eff_func = self.neff_func(tem_indices[self.key_cos], tem_indices[self.key_sin])

        tem_n2w_static = tem_indices2w[self.key_static]
        # 有效折射率是另外两个方向折射率的函数，随扫描角度θ变化
        tem_n2w_eff_func = self.neff_func(tem_indices2w[self.key_cos], tem_indices2w[self.key_sin])

        tem_theta = np.deg2rad(theta_critical_dict[target_mode])

        tem_equations_deltan = {
            "O + O -> E (Type I) ": tem_nw_static - tem_n2w_eff_func(tem_theta),
            "E + E -> O (Type I) ": tem_nw_eff_func(tem_theta) - tem_n2w_static,
            "O + E -> E (Type II)": 0.5 * (tem_nw_static + tem_nw_eff_func(tem_theta)) - tem_n2w_eff_func(tem_theta),
            "O + E -> O (Type II)": 0.5 * (tem_nw_static + tem_nw_eff_func(tem_theta)) - tem_n2w_static
                        }

        delta_k_wavelength = (np.pi * 4 / self.cfg.wavelength_um) * tem_equations_deltan[target_mode]

        # 转换效率: η(Δk) = sinc²(Δk × L/2)
        efficiency_wavelength = (np.sinc(delta_k_wavelength * 1e4 / (2 * np.pi)))**2

        # ===== 绘制接受波长曲线 =====
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wavelength_axis, efficiency_wavelength, 'g-', linewidth=1.5)
        ax.set_xlabel('Wavelength Deviation / nm', fontsize=12)  # X轴: 波长偏差(nm)
        ax.set_ylabel('SHG Efficiency', fontsize=12)          # Y轴: 二次谐波转换效率
        ax.set_title(f'Acceptance Wavelength Curve for {self.cfg.crystal_name} ({target_mode})', fontsize=14)
        ax.grid(True, alpha=0.3)
    
        # ===== 计算接受波长(FWHM, 半高全宽) =====
        half_max = 0.5  
        indices_above_half = np.where(efficiency_wavelength >= half_max)[0]
        
        acceptance_wavelength = np.nan  # 默认值
        acceptance_bandwidth = np.nan   # 默认值
        if len(indices_above_half) > 0:
            lower_index = indices_above_half[0]
            upper_index = indices_above_half[-1]
            
            # 计算波长接受范围 (单位: nm)
            acceptance_wavelength = (wavelength_axis[upper_index] - wavelength_axis[lower_index])
    
            # 根据公式 c = λν 转换为带宽: Δν = c × Δλ / λ²  (单位: GHz) 
            acceptance_bandwidth = 299792458 / (self.cfg.wavelength_nm**2) * acceptance_wavelength 

        return fig, acceptance_wavelength, acceptance_bandwidth

    def acceptance_temperature(self, theta_critical_dict ,target_mode, step, res):
        """
        计算和可视化相位匹配接受温度(相位匹配温度带宽)
        
        相位匹配接受温度是指在相位匹配条件附近，光转换效率仍能保持在一定水平
        (通常为最大值的50%)时的温度偏差范围。这是判断晶体实用性的重要参数。
        
        工作流程:
            1. 与用户交互，获取扫描精度、步数和匹配模式
            2. 在临界温度附近进行微小温度扫描
            3. 计算每个温度的相位失配 Δk 和转换效率
            4. 绘制接受温度曲线
            5. 计算半高全宽(FWHM)对应的接受温度
            
        关键修复:
            - 需要对温度数组中的每一个温度值，分别调用 get_indices() 获取该温度下的折射率
            - 然后计算对应的有效折射率和相位失配
            - 使用循环或数组处理来处理多个温度值
        """
        
        # 以临界温度为中心，前后各扫描 step 个点
        temperature_axis = self.cfg.temperature + np.arange(-step, step) * res 

        # ===== 对每个温度计算折射率和相位失配 =====
        # 由于 get_indices() 可以处理数组，我们直接传入温度数组
        # 获取基频光在所有温度下的折射率 (波长固定)
        tem_indices = self.cfg.get_indices(target_wavelength=self.cfg.wavelength_nm, target_temperature=temperature_axis)
        # 获取倍频光在所有温度下的折射率 (波长为基频光的一半)
        tem_indices2w = self.cfg.get_indices(target_wavelength=self.cfg.wavelength_nm / 2, target_temperature=temperature_axis)

        # 在选定平面内，有一个轴方向的折射率保持不变(o光或e光)
        tem_nw_static = tem_indices[self.key_static]
        # 有效折射率是另外两个方向折射率的函数，随扫描角度θ变化
        tem_nw_eff_func = self.neff_func(tem_indices[self.key_cos], tem_indices[self.key_sin])

        tem_n2w_static = tem_indices2w[self.key_static]
        # 有效折射率是另外两个方向折射率的函数，随扫描角度θ变化
        tem_n2w_eff_func = self.neff_func(tem_indices2w[self.key_cos], tem_indices2w[self.key_sin])

        # 使用在原始温度下计算的相位匹配角度
        # 注意: 严格来说，每个温度下的相位匹配角都会略有变化，但为简化计算，
        # 我们使用初始温度下的相位匹配角。如需精确结果，应对每个温度重新计算相位匹配角。
        tem_theta = np.deg2rad(theta_critical_dict[target_mode])

        # 对每个温度计算对应的Δn值
        tem_equations_deltan = {
            "O + O -> E (Type I) ": tem_nw_static - tem_n2w_eff_func(tem_theta),
            "E + E -> O (Type I) ": tem_nw_eff_func(tem_theta) - tem_n2w_static,
            "O + E -> E (Type II)": 0.5 * (tem_nw_static + tem_nw_eff_func(tem_theta)) - tem_n2w_eff_func(tem_theta),
            "O + E -> O (Type II)": 0.5 * (tem_nw_static + tem_nw_eff_func(tem_theta)) - tem_n2w_static
                        }   
        
        # 计算相位失配: Δk = 4π/λ × Δn
        # 这里 λ 是基频光波长
        delta_k_temperature = (np.pi * 4 / self.cfg.wavelength_um) * tem_equations_deltan[target_mode]
        
        # 计算转换效率: η(Δk) = sinc²(Δk × L/2)
        # 因子 1e4/(2π) 对应于假设的晶体长度和系数
        efficiency_temperature = (np.sinc(delta_k_temperature * 1e4 / (2 * np.pi)))**2

        # ===== 绘制接受温度曲线 =====
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(temperature_axis, efficiency_temperature, 'b-', linewidth=1.5)
        ax.set_xlabel('Temperature Deviation / °C', fontsize=12)  # X轴: 温度偏差(°C)
        ax.set_ylabel('SHG Efficiency', fontsize=12)          # Y轴: 二次谐波转换效率
        ax.set_title(f'Acceptance Temperature Curve for {self.cfg.crystal_name} ({target_mode})', fontsize=14)
        ax.grid(True, alpha=0.3) 
    
        # ===== 计算接受温度(FWHM, 半高全宽) =====
        # FWHM: 效率下降到最大值50%时的温度范围
        half_max = 0.5  
        indices_above_half = np.where(efficiency_temperature >= half_max)[0]
        
        acceptance_temperature = np.nan  # 默认值
        if len(indices_above_half) > 0:
            lower_index = indices_above_half[0]
            upper_index = indices_above_half[-1]
            
            # 计算温度接受范围 (单位: K)
            acceptance_temperature = (temperature_axis[upper_index] - temperature_axis[lower_index])
            print(f"\n接受温度(Acceptance Temperature (FWHM)): {acceptance_temperature:.4f} K·cm")
        else:
            print("No points found above half maximum efficiency.")

        return fig, acceptance_temperature