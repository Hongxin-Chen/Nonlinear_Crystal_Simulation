import numpy as np

class SimulationConfig:
    """
    这个类用来存储一次模拟的所有配置信息
    """
    def __init__(self, crystal_name, wavelength, temperature, plane):
        self.crystal_name = crystal_name  # 存储晶体名
        self.wavelength_nm = wavelength      # 基频波长
        self.wavelength_um = wavelength / 1000.0  # 转换为微米
        self.plane = plane                # 存储平面
        self.temperature = temperature    # 存储温度
        self.crystal_db = {
            "BBO":  {"group": "3m",     "d": {"d22": 2.2, "d31": 0.04, "d15":0.04, "d11":0.02} },
            "KDP":  {"group": "4bar2m", "d": {"d36": 0.39, "d14": 0.39}     },
            "DKDP": {"group": "4bar2m", "d": {"d36": 0.37, "d14": 0.37}    },
            "CLBO": {"group": "4bar2m", "d": {"d36": 0.95, "d14": 0.95}  },
            "LBO":  {"group": "mm2",    "d": {"d31": 1.05, "d32": 0.85, "d33": 0.05, "d15":1.05, "d24":0.85}},
            "KTP":  {"group": "mm2",    "d": {"d31": 2.20, "d32": 3.70, "d33": 14.6, "d15": 2.2, "d24": 3.7}}
        }

    def get_indices(self, target_wavelength=None, target_temperature=None):
        """
        获取晶体在指定波长和温度下的折射率
        
        参数:
            target_wavelength (float or array): 目标波长(nm)，若为None则使用配置中的波长
            target_temperature (float or array): 目标温度(°C)，若为None则使用配置中的温度
        
        返回:
            dict: 包含折射率的字典 {'n_x': ..., 'n_y': ..., 'n_z': ...}
                  注意: 如果输入是数组，返回的折射率也是数组
        """

        if target_wavelength is None:
            wavelength = self.wavelength_um
        else:
            wavelength = target_wavelength / 1000.0  # 转换为微米

        if target_temperature is None:
            dtemp = self.temperature - 20.0  # 假设20°C为参考温度
        else:
            dtemp = target_temperature - 20.0  # 温度相对于20°C的偏差

        if self.crystal_name == "CLBO":
            n_x = np.sqrt(2.2145 + 0.00890 / (wavelength**2 - 0.02051) - 0.01413 * wavelength**2) - 1.9e-6 * dtemp
            n_y = n_x
            n_z = np.sqrt(2.0588 + 0.00866 / (wavelength**2 - 0.01202) - 0.00607 * wavelength**2) - 0.5e-6 * dtemp

        elif self.crystal_name == "BBO":
            n_x = np.sqrt((0.90291 * wavelength**2) / (wavelength**2 - 0.003926) + (0.83155 * wavelength**2) / (wavelength**2 - 0.018786) + (0.76536 * wavelength**2) / (wavelength**2 - 60.01) + 1) - 16.6e-6 * dtemp
            n_y = n_x
            n_z = np.sqrt((1.151075 * wavelength**2) / (wavelength**2 - 0.007142) + (0.21803 * wavelength**2) / (wavelength**2 - 0.02259) + (0.656 * wavelength**2) / (wavelength**2 - 263) + 1) - 9.3e-6 * dtemp

        elif self.crystal_name == "LBO":

            n_x = np.sqrt(2.4542 + 0.01125 / (wavelength**2 - 0.01135) - 0.01388 * wavelength**2) + ((dtemp + 29.13e-3 * dtemp**2) * ((-3.76 * wavelength + 2.30) * 1e-6))
            n_y = np.sqrt(2.5390 + 0.01277 / (wavelength**2 - 0.01189) - 0.01849 * wavelength**2 + (4.3025e-5) * wavelength**4 - (2.9131e-5) * wavelength**6) +((dtemp - (32.89e-4) * dtemp**2) * (6.01 * wavelength - 19.40) * 1e-6)
            n_z = np.sqrt(2.5865 + 0.01310 / (wavelength**2 - 0.01223) - 0.01862 * wavelength**2 + (4.5778e-5) * wavelength**4 - 3.2526e-5 * wavelength**6) + ((dtemp - (74.49e-4) * dtemp**2) * (1.50 * wavelength - 9.70)* 1e-6)
        
        elif self.crystal_name == "KTP":
            #福晶官网数据
            n_x = np.sqrt(3.0065 + 0.03901 / (wavelength**2 - 0.04251) - 0.01327 * wavelength**2) + 1.1e-5 * dtemp
            n_y = np.sqrt(3.0333 + 0.04154 / (wavelength**2 - 0.04547) - 0.01408 * wavelength**2) + 1.3e-5 * dtemp
            n_z = np.sqrt(3.3134 + 0.05694 / (wavelength**2 - 0.05658) - 0.01682 * wavelength**2) + 1.6e-5 * dtemp    

        elif self.crystal_name == "KDP":
            n_x = np.sqrt(2.259276 + 0.01008956 / (wavelength**2 - 0.012942625) + (13.00522 * wavelength**2) / (wavelength**2 - 400))
            n_y = n_x
            n_z = np.sqrt(2.132668 + 0.008637494 / (wavelength**2 - 0.012281043) + (3.2279924 * wavelength**2) / (wavelength**2 - 400))

        elif self.crystal_name == "DKDP":
            n_x = np.sqrt(1.9575544 + (0.2901391 * wavelength**2) / (wavelength**2 - 0.0281399) - 0.02824391 * wavelength**2 + 0.004977826 * wavelength**4)
            n_y = n_x
            n_z = np.sqrt(1.5057799 + (0.6276034 * wavelength**2) / (wavelength**2 - 0.0131558) - 0.01054063 * wavelength**2 + 0.002243821 * wavelength**4)


        return {"n_x": n_x, "n_y": n_y, "n_z": n_z}

