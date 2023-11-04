"""
Модуль, для расчета PVT свойств многофазных смесей
"""
from typing import Optional

import numpy as np
import scipy.interpolate as interp

import unifloc.pvt.black_oil_model as bl
import unifloc.service._constants as const


class FluidFlow:
    """
    Класс для расчета PVT-модели
    """

    def __init__(
        self,
        q_fluid: float,
        pvt_model_data: Optional[dict] = None,
        fluid_type: str = "liquid",
    ):
        """

        Parameters
        ----------
        :param q_fluid: дебит флюида, м3/с
        :param pvt_model_data: словарь, содержащий исходные данные для расчета PVT модели
        :param fluid_type: тип флюида, для которого подается дебит - q_fluid
                            по умолчанию - fluid_type="liquid": pvt свойства рассчитываются
                            для нефти, газа и воды;
                            при fluid_type="gas" pvt свойства рассчитываются только для газа

        Examples:
        --------
        >>> import pandas as pd
        >>>
        >>> from unifloc.pvt.fluid_flow import FluidFlow
        >>> # Инициализация исходных данных класса FluidFlow
        >>> q_fluid = 100 / 86400
        >>> dmuo = {"p": [1, 3398818, 3898840, 4898883, 5898926, 6898978, 7699015],
        ...         "274.15": [1876, 1348.31, 1274.77, 1138.84, 1017.39, 909.557, 832.379],
        ...         "278.15": [1180, 848.084, 801.83, 716.331, 639.933, 572.11, 523.565],
        ...         "283.15": [828, 595.097, 562.64, 502.646, 449.039, 401.446, 367.383],
        ...         "288.15": [537, 385.95, 364.901, 325.991, 291.224, 260.358, 238.266],
        ...         "293.15": [250.341, 179.924, 170.111, 151.972, 135.764, 121.375, 111.076],}
        >>> dfmuo = pd.DataFrame(dmuo)
        >>> dmul = {"wct": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ...         "274.15": [1876, 2000, 2200, 2650, 3500, 5800, 7000, 4200, 1500, 700],
        ...         "278.15": [1180, 1300, 1500, 2000, 2900, 4700, 5600, 3100, 1000, 600],
        ...         "283.15": [828, 1000, 1500, 1700, 2500, 4200, 5200, 3000, 700, 500],
        ...         "288.15": [537, 700, 1000, 1200, 1800, 2600, 3000, 2000, 800, 250],
        ...         "293.15": [250.341, 500, 600, 900, 1200, 2200, 2800, 1500, 420, 240],}
        >>> dfmul = pd.DataFrame(dmul)
        >>> pvt_model_data = {"black_oil": {"gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.8,
        ...                                 "wct": 0, "phase_ratio": {"type": "GOR", "value": 50},
        ...                                 "oil_correlations":
        ...                                  {"pb": "Standing", "rs": "Standing",
        ...                                   "rho": "Standing","b": "Standing",
        ...                                   "mu": "Beggs", "compr": "Vasquez"},
        ...                     "gas_correlations": {"ppc": "Standing", "tpc": "Standing",
        ...                                          "z": "Dranchuk", "mu": "Lee"},
        ...                     "water_correlations": {"b": "McCain", "compr": "Kriel",
        ...                                            "rho": "Standing", "mu": "McCain"},
        ...                     "rsb": {"value": 50, "p": 10000000, "t": 303.15},
        ...                     "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
        ...                     "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
        ...                     "table_model_data": None, "use_table_model": False,
        ...                     "table_mu": {"mul": dfmul, "muo": dfmuo}}}
        >>>
        >>> # Инициализация исходных данных метода расчета pvt-свойств флюидов
        >>> p = 4 * (10 ** 6)
        >>> t = 350
        >>>
        >>> # Инициализация объекта pvt-модели
        >>> fluid_flow = FluidFlow(q_fluid, pvt_model_data)
        >>>
        >>> # Пересчет всех свойств для данного давления и температуры
        >>> fluid_flow.calc_flow(p, t)
        >>>
        >>> # Вывод параметра вязкости жидкости
        >>> mul = fluid_flow.mul
        """
        self.pvt_model_data = pvt_model_data

        # Определение типа рассчитываемой модели и инициализация соответствующего класса
        if "black_oil" in self.pvt_model_data:
            self.pvt_model_data["black_oil"].update({"fluid_type": fluid_type})
            self.pvt_model = bl.BlackOilModel(**self.pvt_model_data["black_oil"])

            # Инициализация переменных класса BlackOil
            self.gamma_oil = self.pvt_model.gamma_oil
            self.gamma_gas = self.pvt_model.gamma_gas
            self.gamma_wat = self.pvt_model.gamma_wat
            self.wct = self.pvt_model.wct
            self.phase_ratio = self.pvt_model.phase_ratio
            self.salinity = self.pvt_model.salinity
            self.table_model_data = self.pvt_model.table_model_data
            self.use_table_model = self.pvt_model.use_table_model

        # Инициализация переменных, подаваемых на вход FluidFlow
        self._q_fluid = q_fluid
        self.__q_fluid_init = q_fluid
        self.q_gas_free = 0
        self.fluid_type = fluid_type.lower()

        # Инициализация атрибутов класса BlackOilModel и TableModel
        self.pb = None
        self.rs = None
        self.bo = None
        self.bg = None
        self.bw = None
        self.muo = None
        self.mug = None
        self.muw = None
        self.rho_oil = None
        self.rho_gas = None
        self.rho_wat = None
        self.z = None
        self.co = None
        self.salinity = None
        self.stwg = None
        self.stog = None
        self.heat_capacity_oil = None
        self.heat_capacity_wat = None
        self.heat_capacity_gas = None

    def reinit(self):
        """
        Метод для сброса флагов для реинициализации расчетов
        """
        self.q_gas_free = 0
        self.wct = self.pvt_model.wct_init
        self.q_fluid = self.__q_fluid_init
        self.pvt_model.reinit_calibrations()
        self.pvt_model.flag_calc_calibrations = self.pvt_model.flag_calc_calibrations_init
        self.pvt_model.rsb_calibr_dict = self.pvt_model.rsb_calibr_dict_init
        self.pvt_model.bo_calibr_dict = self.pvt_model.bo_calibr_dict_init
        self.pvt_model.muo_calibr_dict = self.pvt_model.muo_calibr_dict_init
        self.pvt_model.phase_ratio = self.pvt_model.phase_ratio_init
        self.pvt_model.__bob = None

    @property
    def q_fluid(self):
        return self._q_fluid

    @q_fluid.setter
    def q_fluid(self, new_value):
        """
        Parameters
        ----------
        :param new_value: дебит жидкости, м3/с

        :return: дебит жидкости, м3/с
        -------
        """
        self._q_fluid = new_value

    @property
    def fluid_type(self):
        return self._fluid_type

    @fluid_type.setter
    def fluid_type(self, new_fluid_type):
        """
        Изменение типа флюида в FluidFlow и дочернем классе BlackOil

        Parameters
        ----------
        :param new_fluid_type: новый тип флюида, объект класса Fluid_Flow

        """
        self._fluid_type = new_fluid_type
        self.pvt_model.fluid_type = self._fluid_type

    @property
    def qo(self):
        """
        Свойство расчета текущего дебита нефти

        :return: дебит нефти, м3/с
        -------
        """
        return self.__calc_qo(self.q_fluid, self.wct, self.bo)

    @property
    def qg(self):
        """
        Свойство расчета текущего дебита газа

        :return: дебит газа, м3/с
        -------
        """
        return self.__calc_qg(self.q_fluid, self.wct, self.bg, self.phase_ratio, self.rs, self.q_gas_free)

    @property
    def qw(self):
        """
        Свойство расчета текущего дебита воды

        :return: дебит воды, м3/с
        -------
        """
        return self.__calc_qw(self.q_fluid, self.wct, self.bw)

    @property
    def ql(self):
        """
        Свойство расчета текущего дебита жидкости

        :return: дебит жидкости, м3/с
        -------
        """

        return self.qw + self.qo

    @property
    def qm(self):
        """
        Свойство расчета текущего дебита смеси

        :return: дебит смеси, м3/с
        -------
        """
        return self.qw + self.qo + self.qg

    @property
    def mul(self):
        """
        Свойство расчета вязкости жидкости

        :return: вязкость жидкости, сПз
        -------
        """
        if self.ql > 0:
            wc_rc = self.qw / self.ql
        else:
            wc_rc = self.wct
        return self.__calc_mul(self.muo, wc_rc, self.muw)

    @property
    def mum(self):
        """
        Свойство расчета вязкости смеси

        :return: вязкость смеси, сПз
        -------
        """
        return self.__calc_mum(self.gf, self.mug, self.mul)

    @property
    def gf(self):
        """
        Свойство расчета доли газа

        :return: доля газа, д.ед.
        -------
        """
        q_mix = self.qw + self.qo + self.qg
        if q_mix > 0:
            return self.qg / q_mix
        return 0

    @property
    def wf(self):
        """
        Свойство расчета доли воды

        :return: доля воды, (д.ед.)
        -------
        """
        return self.__calc_wf(self.wct, self.bw, self.bo)

    @property
    def ro(self):
        """
        Свойство расчета текущей плотности нефти

        :return: плотность нефти, кг/м3
        -------
        """
        return self.__calc_ro(self.gamma_oil, self.rs, self.gamma_gas, self.bo)

    @property
    def rw(self):
        """
        Свойство расчета текущей плотности воды

        :return: плотность воды, кг/м3
        -------
        """
        return self.__calc_rw(self.gamma_wat, self.bw)

    @property
    def rg(self):
        """
        Свойство расчета текущей плотности газа

        :return: плотность газа, кг/м3
        -------
        """
        return self.__calc_rg(self.gamma_gas, self.bg)

    @property
    def rl(self):
        """
        Свойство расчета текущей плотности жидкости

        :return: плотность жидкости, кг/м3
        -------
        """
        return self.__calc_rl(self.qo, self.qw, self.ro, self.rw)

    @property
    def rm(self):
        """
        Свойство расчета текущей плотности смеси

        :return: плотность смеси, кг/м3
        -------
        """
        return self.__calc_rm(self.qg, self.qm, self.rl, self.rg)

    @property
    def stlg(self):
        """
        Свойство расчета коэффициента поверхностного натяжения на границе газ-жидкость

        :return: поверхностное натяжение, Н/м
        -------
        """
        return self.__calc_stlg(self.qw, self.ql, self.stog, self.stwg)

    @property
    def mass_q_oil(self):
        """
        Свойство расчета массового дебита нефти

        :return: массовый дебит нефти, кг/с
        -------
        """
        return self.ro * self.qo if self.ro is not None else 0

    @property
    def mass_q_wat(self):
        """
        Свойство расчета массового дебита воды

        :return: массовый дебит воды, кг/с
        -------
        """
        return self.rw * self.qw if self.rw is not None else 0

    @property
    def mass_q_gas(self):
        """
        Свойство расчета массового дебита газа

        :return: массовый дебит газа, кг/с
        -------
        """
        return self.rg * self.qg if self.rg is not None else 0

    @property
    def mass_q_liq(self):
        """
        Свойство расчета массового дебита жидкости

        :return: массовый дебит жидкости, кг/с
        -------
        """
        return self.mass_q_oil + self.mass_q_wat

    @property
    def mass_q_mix(self):
        """
        Свойство расчета массового дебита смеси

        :return: массовый дебит смеси, кг/с
        -------
        """
        return self.mass_q_oil + self.mass_q_wat + self.mass_q_gas

    @property
    def heat_capacity_liq(self):
        """
        Свойство расчета удельной теплоемкости жидкости

        :return: Удельная теплоемкость жидкости, Дж/(кг*К)
        -------
        """
        return self.__calc_heat_capacity_liq(self.mass_q_oil, self.mass_q_wat)

    @property
    def heat_capacity_mixture(self):
        """
        Свойство расчета удельной теплоемкости смеси

        Returns
        -------
        Удельная теплоемкость смеси, Дж/(кг*К)
        """
        return self.__calc_heat_capacity_mixture(self.mass_q_liq, self.mass_q_gas)

    def __calc_qo(self, q_liq, wct, bo):
        """
        Метед расчета дебита нефти в пластовых условиях

        Parameters
        ----------
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param bo: объемный коэффициент нефти, м3/м3

        :return: дебит нефти, м3/с
        -------
        """
        if bo is None:
            bo = 1
        return 0 if self.fluid_type == "gas" else q_liq * (1 - wct) * bo

    def __calc_qg(self, q_fluid, wct, bg, phase_ratio, rs, q_gas_free):
        """
        Метод расчета дебита газа в пластовых условиях

        Parameters
        ----------
        :param q_fluid: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param bg: объемный коэффициент газа, м3/м3
        :param phase_ratio: словарь объемного соотношение фоз
            * "type": тип объемного соотношения фаз
                * "GOR": gas-oil ratio
                * "GlR": gas-liquid ratio (если wct = 1: liquid = water + gas, то работает как GWR (gas-water ratio))
            * "value": объемное соотношение фаз, м3/м3
        :param rs: газосодержание, м3/м3
        :param q_gas_free: дебит свободного газа, м3/с

        :return: дебит газа, м3/с
        -------
        """
        if bg is None:
            bg = 0
        if rs is None:
            rs = 0
        if self.fluid_type == "gas":
            q_gas = q_fluid * bg
        elif self.fluid_type == "water":
            q_gas = 0
        else:
            if phase_ratio["type"].lower() == "gor":
                q_gas = bg * (q_fluid * (1 - wct) * (phase_ratio["value"] - rs) + q_gas_free)
            else:
                q_gas = bg * (q_fluid * (phase_ratio["value"] - rs * (1 - wct)) + q_gas_free)
            # Бывают проблемы из-за интерполяции при модификации
            q_gas = max(q_gas, 0)
        return q_gas

    def __calc_qw(self, q_liq, wct, bw):
        """
        Расчет дебита воды в пластовых условиях

        Parameters
        ----------
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param bw: объемный коэффициент воды, м3/м3

        :return: дебит воды, м3/с
        -------
        """
        return 0 if bw is None or self.fluid_type == "gas" else q_liq * wct * bw

    @staticmethod
    def __calc_wf(wct, bw, bo):
        """
        Расчет доли воды для заданных давления и температуры

        Parameters
        ----------
        :param wct: обводненнность, д.ед.
        :param bw: объемный коэффициент воды, м3/м3
        :param bo: объемный коэффициент нефти, м3/м3

        :return: доля воды, д.ед.
        -------
        """
        if bw is not None and bo is not None and wct is not None:
            liq_fraction = (1 - wct) * bo + wct * bw
            if liq_fraction > 0:
                return (wct * bw) / liq_fraction
        return 0

    def __calc_mul(
        self,
        muo: float,
        wc_rc: float,
        muw: float,
        method: str = "continuous",
    ):
        """
        Метод расчета вязкости жидкости

        Parameters
        ----------
        :param muo: вязкость нефти, сПз
        :param wc_rc: обводненность, д.ед
        :param muw: вязкость воды, сПз

        :return: вязкость жидкости, сПз
        -------
        """
        if self.pvt_model.mul is not None:
            return self.pvt_model.mul
        else:
            if muo is None:
                muo = 0
            if muw is None:
                muw = 0
            if method == "continuous":
                return muo if wc_rc < 0.6 else muw
            else:
                return muo * (1 - wc_rc) + muw * wc_rc

    @staticmethod
    def __calc_mum(gf, mug, mul):
        """
        Метод расчета вязкости ГЖС (при дебите жидкости > 0)

        Parameters
        ----------
        :param gf: доля газа, д.ед
        :param mug: вязкость газа, сПз
        :param mul: вязкость жидкости, сПз

        :return: вязкость ГЖС, сПз
        -------
        """
        if mug is None:
            mug = 0
        return mul * (1 - gf) + mug * gf

    def __calc_ro(self, gamma_oil, rs, gamma_gas, bo):
        """
        Метод расчета плотности нефти

        Parameters
        ----------
        :param gamma_oil: относительная плотность нефти, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param rs: газосодержание, м3/м3
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param bo: объемный коэффициент нефти, м3/м3

        :return: плотность нефти, кг/м3
        -------
        """
        if self.rho_oil is None:
            if bo is None:
                bo = 1
            if rs is None:
                rs = 0
            return 1000 * (gamma_oil + rs * gamma_gas * 0.0012217) / bo
        else:
            return self.pvt_model.rho_oil

    def __calc_rw(self, gamma_wat, bw):
        """
        Метод расчета плотности воды

        Parameters
        ----------
        :param gamma_wat: относительная плотность воды, доли,
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param bw: объемный коэффициент воды, м3/м3

        :return: плотность воды, кг/м3
        -------
        """
        if self.rho_wat is None:
            if bw is None:
                bw = 1
            return 1000 * gamma_wat / bw
        else:
            return self.pvt_model.rho_wat

    def __calc_rg(self, gamma_gas, bg):
        """
        Метод расчета плотности газа

        Parameters
        ----------
        :param gamma_gas: относительная плотность газа, доли,
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param bg: объемный коэффициент газа, м3/м3

        :return: плотность газа, кг/м3
        -------
        """
        if self.rho_gas is None:
            if bg is None:
                bg = 1
            return gamma_gas * 1.2217 / bg
        else:
            return self.pvt_model.rho_gas

    def __calc_rl(self, qo, qw, ro, rw):
        """
        Метод расчета плотности жидкости

        Parameters
        ----------
        :param qo: дебит нефти, м3/с
        :param qw: дебит воды, м3/с
        :param ro: плотность нефти, кг/м3
        :param rw: плотность воды, кг/м3

        :return: плотность жидкости, кг/м3
        -------
        """
        ql = qw + qo
        if ql > 0:
            wc_rc = qw / ql
        else:
            wc_rc = self.wct
        return (1 - wc_rc) * ro + wc_rc * rw

    def __calc_rm(self, qg, qm, rl, rg):
        """
        Метод расчета плотности смеси

        Parameters
        ----------
        :param qg: дебит газа, м3/с
        :param qm: дебит смеси, м3/с
        :param rl: плотность жидкости, кг/м3
        :param rg: плотность газа, кг/м3

        :return: плотность смеси, кг/м3
        -------
        """
        if qm > 0 and rl is not None:
            f_g = qg / qm
            rm = rl * (1 - f_g) + rg * f_g
        elif self.fluid_type in ["liquid", "water"]:
            rm = rl
        elif self.fluid_type == "gas":
            rm = rg
        else:
            rm = None
        return rm

    def __calc_stlg(self, qw, ql, sto, stw):
        """
        Метод расчета поверхностного натяжения на границе газ-жидкость

        Parameters
        ----------
        :param qw: дебит воды, м3/с
        :param ql: дебит жидкости, м3/с
        :param sto: поверхностное натяжение на границе газ-нефть, Н/м
        :param stw: поверхностное натяжение на границе газ-вода, Н/м

        :return: st (superficial tension) - поверхностное натяжение, Н/м
        -------
        """
        if sto is not None and stw is not None:
            if ql > 0:
                wc_rc = qw / ql
            else:
                wc_rc = self.wct
            # В пайпсим написано что есть переход в 60 %,
            # но там ошибка в мануале на самом деле всегда так
            st = sto * (1 - wc_rc) + stw * wc_rc
        else:
            st = None
        return st

    def __transfer_pvt_attributes(self, p: float, t: float):
        """
        Метод вызова атрибутов класса BlackOil/Table,
        необходимых для расчета PVT-параметров потока
        Parameters
        ----------
        :param p: Давление, Па
        :param t: Температура, К

        :return: атрибуты класса BlackOil/Table
        -------
        """
        self.pb = self.pvt_model.pb
        self.rs = self.pvt_model.rs
        self.bo = self.pvt_model.bo
        self.bg = self.pvt_model.bg
        self.bw = self.pvt_model.bw
        self.muo = self.pvt_model.muo
        self.mug = 0 if not self.pvt_model.mug else self.pvt_model.mug
        self.muw = self.pvt_model.muw
        self.rho_oil = self.pvt_model.rho_oil
        self.rho_gas = self.pvt_model.rho_gas
        self.rho_wat = self.pvt_model.rho_wat
        self.co = self.pvt_model.compro
        self.z = self.pvt_model.z
        self.salinity = self.pvt_model.salinity
        self.co = self.pvt_model.compro
        self.phase_ratio = self.pvt_model.phase_ratio
        self.stwg = self.pvt_model.st_wat_gas if self.pvt_model.st_wat_gas is not None else 0
        self.stog = self.pvt_model.st_oil_gas if self.pvt_model.st_oil_gas is not None else 0
        self.heat_capacity_oil = self.pvt_model.heat_capacity_oil
        self.heat_capacity_wat = self.pvt_model.heat_capacity_wat
        self.heat_capacity_gas = self.pvt_model.heat_capacity_gas

    def calc_flow(self, p: float, t: float):
        """
        Метод расчета PVT-параметров потока (дебиты, вязкости, плотности, поверхностные натяжения)

        Parameters
        ----------
        :param p: Давление, Па
        :param t: Температура, К

        :return: PVT-параметры потока
        -------
        """
        # Обращение к классу BlackOil/Table и запуск метода calc_pvt
        self.pvt_model.calc_pvt(p, t)
        # Вызов атрибутов класса BlackOil/Table, необходимых для расчета PVT-параметров потока
        self.__transfer_pvt_attributes(p, t)

    def modify(
        self,
        p: float,
        t: float,
        k_sep: float,
        calc_type: str = "tubing",
    ):
        """
        Метод модификации свойств флюида после сепарации

        Parameters
        ----------
        :param p: давление сепарации, Па
        :param t: температура сепарации, К
        :param k_sep: коэффициент сепарации
        :param calc_type: тип расчета: для НКТ: "tubing", иначе - для затруба

        :return: модифицирует модель после сепарации
        -------
        """
        self.pvt_model.calc_pvt(p, t)
        self.__transfer_pvt_attributes(p, t)

        if self.phase_ratio["type"].lower() == "gor":
            rp = self.phase_ratio["value"]
        else:
            rp = self.phase_ratio["value"] / (1 - self.wct)

        # Расчет нового газового фактора в зависимости от типа расчета (для НКТ/для затруба)
        if calc_type.lower() == "tubing":
            # Новый газовый фактор с учетом сепарации газа
            rp_new = rp - (rp - self.rs) * k_sep
        else:
            # Новый газовый фактор с учетом сепарации газа для затруба
            rp_new = (rp - self.rs) * k_sep

        if rp_new == 0:
            self.pvt_model.rsb_calibr_dict = None
            self.pvt_model.bo_calibr_dict = None
            self.pvt_model.muo_calibr_dict = None
            self.pvt_model.flag_calc_calibrations = False
        elif rp_new < rp:
            # Если газовый фактор становится меньше газосодержания, тогда надо
            # скорректировать газосодержание
            # и давление насыщения, которое будет от него зависеть

            n = 10
            rs_curve = np.empty(n)
            pb_rs_curve = np.linspace(const.ATM, self.pb, n)
            bo_rs_curve = np.empty(n)
            muo_rs_curve = np.empty(n)

            for i, p in enumerate(pb_rs_curve):
                # Пересчитываются все pvt-параметры по новому давлению и температуре сепарации
                self.pvt_model.calc_pvt(p, t)

                rs_curve[i] = self.pvt_model.rs
                bo_rs_curve[i] = self.pvt_model.bo
                muo_rs_curve[i] = self.pvt_model.muo

            pb = self.__interpolation_for_modify(rs_curve, pb_rs_curve, rp_new)
            # Внесение калибровочных значений для газосодержания в соответствующий словарь
            self.pvt_model.rsb_calibr_dict = {"value": rp_new, "p": pb, "t": t}

            bob = self.__interpolation_for_modify(rs_curve, bo_rs_curve, rp_new)
            # Внесение калибровочных значений для объемного коэффициента нефти
            # в соответствующий словарь
            self.pvt_model.bo_calibr_dict = {"value": bob, "p": pb, "t": t}

            muob = self.__interpolation_for_modify(rs_curve, muo_rs_curve, rp_new)
            # Внесение калибровочных значений для вязкости нефти в соответствующий словарь
            self.pvt_model.muo_calibr_dict = {"value": muob, "p": pb, "t": t}

            self.pvt_model.flag_calc_calibrations = True

        # Итоговый газовый фактор всегда с учетом сепарации
        if self.phase_ratio["type"].lower() == "gor":
            self.pvt_model.phase_ratio = {"type": "GOR", "value": rp_new}
            self.phase_ratio = {"type": "GOR", "value": rp_new}
        else:
            self.pvt_model.phase_ratio = {
                "type": "GLR",
                "value": rp_new * (1 - self.wct),
            }
            self.phase_ratio = {"type": "GLR", "value": rp_new * (1 - self.wct)}

    @staticmethod
    def __interpolation_for_modify(x_curve, y_curve, x_point):
        """
        Метод интерполяционного расчета параметра по заданным зависимостям
        для функции модификации флюида после сепарации

        Parameters
        ----------
        :param x_curve: массив значений x
        :param y_curve: массив значений y
        :param x_point: точка x, для которой необходимо определить y

        :return: искомое значение y
        -------
        """
        # Определение интерполяционной функции
        interp_func = interp.interp1d(x_curve, y_curve, kind="linear", fill_value="extrapolate")
        result = interp_func(x_point).item()
        return result

    def __calc_heat_capacity_liq(self, mass_q_oil, mass_q_wat):
        """
        Расчет удельной теплоемкости жидкости

        Parameters
        ----------
        :param mass_q_oil: массовый дебит нефти, кг/с
        :param mass_q_wat: массовый дебит воды, кг/с

        Returns
        -------
        Удельная теплоемкость жидкости, Дж/(кг*К)
        """

        # Расчет теплоемкости для жидкости
        return (
            0
            if mass_q_oil == 0 and mass_q_wat == 0
            else (self.heat_capacity_oil * mass_q_oil + self.heat_capacity_wat * mass_q_wat) / (mass_q_oil + mass_q_wat)
        )

    def __calc_heat_capacity_mixture(self, mass_q_liq, mass_q_gas):
        """
        Расчет удельной теплоемкости смеси

        Parameters
        ----------
        :param mass_q_liq: массовый дебит жидкости, кг/с
        :param mass_q_gas: массовый дебит газа, кг/с

        Returns
        -------
        Удельная теплоемкость смеси, Дж/(кг*К)
        """

        # Расчет теплоемкости для смеси
        return (
            0
            if mass_q_liq == 0 and mass_q_gas == 0
            else (self.heat_capacity_liq * mass_q_liq + self.heat_capacity_gas * mass_q_gas) / (mass_q_liq + mass_q_gas)
        )

    def __dz_dt_p(self, t: float, p: float) -> float:
        """
        Метод расчета производной z - фактора по температуре при постоянном давлении

        Parameters
        ----------
        :param t: температура, К
        :param p: давление, Па

        :return: производная z - фактора по температуре
        -------
        """
        z1 = self.pvt_model.z_func(p=p, t=t - 0.1, gamma_gas=self.pvt_model.gamma_gas, pvt_property="z")
        z2 = self.pvt_model.z_func(p=p, t=t + 0.1, gamma_gas=self.pvt_model.gamma_gas, pvt_property="z")
        return (z2 - z1) / 0.2

    @staticmethod
    def __calc_jt(
        t: float,
        cp_n: float,
        x: float,
        rho_liq: float,
        rho_gas: float,
        dz_dt: float,
        z: float,
    ) -> float:
        """
        Функция расчета коэффициента Джоуля-Томсона

        Parameters
        ----------
        :param t: температура, К
        :param cp_n: теплоемкость смеси, Дж/(кг*К)
        :param x: доля газа, д.ед
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3
        :param dz_dt: производная z - фактора по температуре
        :param z: z - фактор

        :return: коэффициент Джоуля-Томсона
        -------
        """
        jt = 1 / cp_n * (x * t * dz_dt / (rho_gas * z) - (1 - x) / rho_liq)
        return jt

    def calc_joule_thomson_coeff(self, t: float, p: float) -> float:
        """
        Метод расчета коэффициента Джоуля-Томсона

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К

        :return: коэффициент Джоуля-Томсона
        -------
        """
        self.calc_flow(p, t)
        cp_n = self.__calc_heat_capacity_mixture(self.mass_q_liq, self.mass_q_gas)
        mass_mix = self.mass_q_liq + self.mass_q_gas
        if mass_mix != 0:
            x = self.mass_q_gas / mass_mix
        else:
            x = 0
        dz_dt = self.__dz_dt_p(t, p)
        return self.__calc_jt(t, cp_n, x, self.rl, self.rg, dz_dt, self.z)
