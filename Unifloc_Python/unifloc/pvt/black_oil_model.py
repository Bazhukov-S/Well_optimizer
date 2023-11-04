"""
Модуль, для расчета PVT свойств по модели Black Oil
"""
from typing import Optional

import numpy as np

from unifloc.pvt import _gas_correlations as gas
from unifloc.pvt import _oil_correlations as oil
from unifloc.pvt import _water_correlations as wat
from unifloc.pvt import pvt_model as pvt
from unifloc.pvt import table_model as tbl
from unifloc.service import _constants as const
from unifloc.tools import exceptions as exc


class BlackOilModel(pvt.PvtModel):
    """
    Класс расчета PVT-параметров по модели BlackOil
    """

    def __init__(
        self,
        gamma_gas: float,
        gamma_oil: float,
        gamma_wat: float,
        wct: float,
        phase_ratio: Optional[dict] = None,
        oil_correlations: Optional[dict] = None,
        gas_correlations: Optional[dict] = None,
        water_correlations: Optional[dict] = None,
        salinity: Optional[float] = None,
        rsb: Optional[dict] = None,
        bob: Optional[dict] = None,
        muob: Optional[dict] = None,
        table_model_data: Optional[dict] = None,
        use_table_model: bool = False,
        table_mu: Optional[dict] = None,
        fluid_type: str = "liquid",
    ):
        """

        Parameters
        ----------
        :param gamma_gas: относительная плотность газа,
                          (относительно воздуха с плотностью 1.2217 кг/м3 при с.у.)
        :param gamma_oil: относительная плотность нефти,
                          (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_wat: относительная плотность воды,
                          (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param wct: обводненность, д.ед.
        :param phase_ratio: словарь объемного соотношение фаз

            * "type": тип объемного соотношения фаз
                * "GOR": gas-oil ratio
                * "GlR": gas-liquid ratio (if wct = 1: liquid = water + gas, то работает как GWR (gas-water ratio))
            * "value": объемное соотношение фаз, м3/м3
        :param oil_correlations: корреляции для нефти
        :param gas_correlations: корреляции для газа
        :param water_correlations: корреляции для воды
        :param salinity: минерализация воды, ppm
        :param rsb: словарь с калибровочным значением газосодержания при давлении насыщения

            * "value": калибровочное значение газо-ния при дав-ии нас-я, ст. м3 газа/ст. м3 нефти
            * "p": давление насыщения, Па абс.
            * "t": температура калибровки газосодержания при давлении насыщения, К
        :param bob: словарь с калибровочным значением объемного
                    коэффициента нефти при давлении насыщения

                    * "value": калибровочное значение объемного коэффициента нефти, ст.м3/ст.м3
                    * "p": давление калибровки, Па абс.
                    * "t": температура калибровки объемного коэффициента нефти
                    при давлении насыщения, К
        :param muob: словарь с калибровочным значением вязкости нефти при давлении насыщения

                    * "value": калибровочное значение вязкости нефти, сПз
                    * "p": давление калибровки, Па абс.
                    * "t": температура калибровки вязкости нефти при давлении насыщения, К
        :param table_model_data: словарь с DataFrame с данными табличной модели
        :param use_table_model: флаг использования расчета по табличной модели
        :param table_mu: словарь с DataFrame с данными табличной модели для muo, mul

                    * "mul": Dataframe с табличными значениями для вязкости жидкости, см.пример во FluidFlow
                    * "muo": Dataframe с табличными значениями для вязкости нефти, см.пример во FluidFlow
        :param fluid_type: тип флюида, для которого подается дебит - q_fluid
                            по умолчанию - fluid_type="liquid": pvt свойства
                            рассчитываются для нефти, газа и воды;
                            при fluid_type="gas": pvt свойства рассчитываются только для газа
        -------
        """
        if water_correlations is None:
            water_correlations = const.WAT_CORRS
        if gas_correlations is None:
            gas_correlations = const.GAS_CORRS
        if oil_correlations is None:
            oil_correlations = const.OIL_CORRS

        # Проверка на корректный ввод типа фазового соотношения
        if phase_ratio:
            if phase_ratio["type"].lower() not in ["gor", "glr"]:
                raise exc.NotImplementedPhaseRatioError(
                    "Данное фазовое соотношение недоступно. Воспользуйтесь GOR или GLR.",
                    phase_ratio,
                )

        # Инициализация переменных, подаваемых на вход BlackOilModel
        self.gamma_gas = gamma_gas
        self.gamma_oil = gamma_oil
        self.gamma_wat = gamma_wat
        self.wct = wct
        self.wct_init = wct
        self.phase_ratio = phase_ratio
        self.phase_ratio_init = phase_ratio
        self._oil_correlations = self.__check_correlations(
            {k: v.lower() for k, v in oil_correlations.items() if v is not None},
            const.OIL_CORRS,
        )
        self._gas_correlations = self.__check_correlations(
            {k: v.lower() for k, v in gas_correlations.items() if v is not None},
            const.GAS_CORRS,
        )
        self._water_correlations = self.__check_correlations(
            {k: v.lower() for k, v in water_correlations.items() if v is not None},
            const.WAT_CORRS,
        )
        self.salinity = salinity
        self.table_model_data = table_model_data
        self.use_table_model = use_table_model
        self.table_mu = table_mu
        self.fluid_type = fluid_type

        # Исходные значения калибровок
        self.rsb_calibr_dict_init = rsb
        self.bo_calibr_dict_init = bob
        self.muo_calibr_dict_init = muob

        # Инициализация исходных словарей с заданными калибровочными коэффициентами
        self.rsb_calibr_dict = rsb
        self.bo_calibr_dict = bob
        self.muo_calibr_dict = muob

        # Считаем калибровочные коэффициенты только при наличии калибровки по газосодержанию
        if self.rsb_calibr_dict is not None and None not in self.rsb_calibr_dict.values():
            self.flag_calc_calibrations = True
            self.flag_calc_calibrations_init = True
        else:
            self.flag_calc_calibrations = False
            self.flag_calc_calibrations_init = False

        # Инициализация исходных словарей с табличными значениями вязкости
        if self.table_mu:
            (
                self.p_table,
                self.wct_table,
                self.muo_tables,
                self.mul_tables,
                self.t_list,
                self.t_diap,
            ) = self.__viscositydata()

        # Инициализация калибровочных коэффициентов
        self.calibr_rs = 1
        self.calibr_bo = 1
        self.calibr_mu = 1
        self.calibr_pb = 1

        # Инициализация атрибутов класса BlackOilModel
        self.pb = None
        self.rs = None
        self.muo = None
        self.mul = None
        self.rho_oil = None
        self.bo = None
        self.__bob = None
        self.compro = None
        self.z = None
        self.bg = None
        self.rho_gas = None
        self.mug = None
        self.bw = None
        self.comrw = None
        self.rho_wat = None
        self.muw = None
        self.st_wat_gas = None
        self.st_oil_gas = None
        self.heat_capacity_wat = None
        self.heat_capacity_gas = None
        self.heat_capacity_oil = None
        self.oil_corrs = oil.OilCorrelations(self._oil_correlations)
        self.gas_corrs = gas.GasCorrelations(self._gas_correlations)
        self.wat_corrs = wat.WaterCorrelations(self._water_correlations)

        self.__define_pvt_funcs()

    def __define_pvt_funcs(self):
        """
        Определение функций для расчета PVT-свойств в зависимости от количества поданных таблично
        """
        # TODO: v1.5.0 - убрать рефлексию
        if self.table_model_data is not None and self.use_table_model:
            self.table_model = tbl.TableModel(self.table_model_data)

            for k in self.table_model_data:
                if k in const.ALL_PROPERTIES:
                    setattr(self, k + "_func", self.table_model.calc_property)
        else:
            self.table_model = None

        for k in const.ALL_PROPERTIES:
            if not hasattr(self, k + "_func"):
                setattr(
                    self,
                    k + "_func",
                    getattr(
                        getattr(self, const.ALL_PROPERTIES[k][0]),
                        const.ALL_PROPERTIES[k][1],
                    ),
                )

    def __viscositydata(self):
        """
        Метод для инициализации данных для расчета вязкости флюида по табличным значениям

        Parameters
        ----------
        :return p_table: массив с таблиными значениями давления, Па
        :return wct_table: массив со значенияими табличной обводненности, дол.ед
        :return muo_tables: массив со значенияими табличной вязкости нефти, сП
        :return mul_tables: массив со значенияими табличной вязкости жидкости, сП
        :return t_list: список со значенияими температур измерений, К
        :return t_diap: список с диапазонами температур измерений, К
        ----------
        """
        p_table = self.table_mu["muo"]["p"].values
        wct_table = self.table_mu["mul"]["wct"].values
        df_muo2 = self.table_mu["muo"].iloc[:, 1:]
        df_mul2 = self.table_mu["mul"].iloc[:, 1:]
        muo_tables = []
        mul_tables = []
        for i, j in zip(df_muo2.columns, df_mul2.columns):
            muo_nam = self.table_mu["muo"][i].values
            muo_nam.flatten()
            muo_tables.append([float(i), muo_nam])
            wct_nam = self.table_mu["mul"][j].values
            wct_nam.flatten()
            mul_tables.append([float(j), wct_nam])
        t_list = [i[0] for i in muo_tables]
        t_diap = [(t_list[i - 1], val) for i, val in (enumerate(t_list))]
        del t_diap[0]
        return p_table, wct_table, muo_tables, mul_tables, t_list, t_diap

    @staticmethod
    def __make_interp_func(x_array, y_array, n):
        """
        Общая функция для создания интерполяционного полинома n-ой степени

        Parameters
        ----------
        :x_array: массив иксов
        :y_array: массив игреков
        :n: степень полинома

        :return: интерполяционный полином n-ой степени
        ----------
        """
        func = np.polyfit(x_array, y_array, n)
        func = np.poly1d(func)
        return func

    def viscousertable(self, p: float, t: float, wct: float):
        """
        Расчет вязкости жидкости для данного давления и температуры

        Parameters
        ----------
        :param p: давление в тех же единицах измерения что и в исходной таблице
        :param t: температура в тех же единицах измерения  что и в исходной таблице
        :param wct: обводненность в тех же единицах измерения  что и в исходной таблице

        :return mul: вязкость жидкости при p, t, wct, сП
        :return muo: вязкость нефти при p, t, сП
        ----------
        """
        if t > self.t_list[-1]:
            t = min(t, self.t_list[-1])
        elif t < self.t_list[0]:
            t = max(t, self.t_list[0])

        table_rash = np.array([])
        if t in self.t_list:
            muo_t = self.muo_tables[self.t_list.index(t)]
            mu_func = self.__make_interp_func(self.p_table, muo_t[1], 3)
            muo = mu_func(p)
            if wct != 0:
                table_wct = self.mul_tables[self.t_list.index(t)]
                muo_wct = np.interp(wct, self.wct_table, table_wct[1])
                mul_func = mu_func - muo_t[1][0] + muo_wct
                mul = mul_func(p)
        else:
            for t_d in self.t_diap:
                if t_d[0] < t < t_d[1]:
                    k = (abs(t_d[1] - t)) / (abs(t_d[0] - t_d[1]))
                    table_t1 = self.muo_tables[self.t_list.index(t_d[0])]
                    table_t2 = self.muo_tables[self.t_list.index(t_d[1])]
                    muo_func_t1 = self.__make_interp_func(self.p_table, table_t1[1], 3)
                    muo_func_t2 = self.__make_interp_func(self.p_table, table_t2[1], 3)
                    muo = (muo_func_t1(p) + muo_func_t2(p)) * k
                    if wct != 0:
                        for i in self.p_table:
                            muot = (muo_func_t1(i) + muo_func_t2(i)) * k
                            table_rash = np.append(table_rash, [muot])
                        table_wct1 = self.mul_tables[self.t_list.index(t_d[0])]
                        table_wct2 = self.mul_tables[self.t_list.index(t_d[1])]
                        mul_wct1 = np.interp(wct, self.wct_table, table_wct1[1]) / t_d[0] * t
                        mul_wct2 = np.interp(wct, self.wct_table, table_wct2[1]) / t_d[1] * t
                        mul1 = (mul_wct1 + mul_wct2) / 2
                        mu_func = self.__make_interp_func(self.p_table, table_rash, 3)
                        mul_func = mu_func - table_rash[0] + mul1
                        mul = mul_func(p)
        if wct == 0:
            mul = muo
        mul = max(mul, 0.2)
        muo = max(muo, 0.2)
        return mul, muo

    def __muoviscuserdata(
        self,
        p: float,
        t: float,
    ):
        """
        Расчет вязкости нефти для данного давления и температуры

        Parameters
        ----------
        :param p: давление в тех же единицах измерения что и в исходной таблице
        :param t: температура в тех же единицах измерения  что и в исходной таблице

        :return muo_userdata: вязкость нефти при p, t, сП
        ----------
        """
        if p <= self.p_table[-1]:
            muo_userdata = self.viscousertable(p, t, 0)[1]
        elif self.pb <= self.p_table[-1]:
            muo_pb = self.viscousertable(self.pb, t, 0)[1]
            muo_userdata = self.oil_corrs._oil_viscosity_vasquez_beggs(muo_pb, p, self.pb)
        else:
            muo_pb = self.viscousertable(self.p_table[-1], t, 0)[1]
            muo_userdata = self.oil_corrs._oil_viscosity_vasquez_beggs(muo_pb, p, self.pb)
        return muo_userdata

    @staticmethod
    def __check_correlations(correlations: dict, correlations_default: dict) -> dict:
        """ "
        Функция проверки корреляций свойств

        :param correlations: словарь корреляций для проверки
        :param correlations_default: словарь корреляций по умолчанию

        :return: correlations - скорректированный словарь корреляций
        """
        for key in correlations_default:
            if correlations.get(key) is None:
                correlations.update({key: correlations_default[key]})
        return correlations

    @property
    def oil_correlations(self):
        """
        read-only атрибут с набором корреляций для расчета нефти

        :return: словарь с набором корреляций для расчета нефти
        -------

        """
        return self._oil_correlations

    @property
    def gas_correlations(self):
        """
        read-only атрибут с набором корреляций для расчета газа

        :return: словарь с набором корреляций для расчета газа
        -------

        """
        return self._gas_correlations

    @property
    def water_correlations(self):
        """
        read-only атрибут с набором корреляций для расчета воды

        :return: словарь с набором корреляций для расчета воды
        -------

        """
        return self._water_correlations

    def reinit_calibrations(self):
        """
        Метод для сброса калибровочных коэффициентов
        """
        self.calibr_rs = 1
        self.calibr_bo = 1
        self.calibr_mu = 1
        self.calibr_pb = 1

    def calc_pvt(self, p: float, t: float):
        """
        Метод расчета PVT-параметров нефти, газа и воды

        Parameters
        ----------
        Функция для расчета всех физико-химических свойств
        :param p: давление, Па
        :param t: температура, К

        :return: pvt-параметры для нефти, газа и воды
        -------
        """
        # TODO: v1.5.0 - сделать удобные интерфейсы для функций свойств
        # Вызов метода расчета калибровочных коэффициентов
        if self.fluid_type == "gas":
            self.__calc_gas_pvt_parameters(p, t)
        elif self.fluid_type == "water":
            self.__calc_water_pvt_parameters(p, t)
        elif self.phase_ratio["type"].lower() == "glr" and self.wct == 1:
            self.__calc_gas_pvt_parameters(p, t)
            self.__calc_water_pvt_parameters(p, t)
        else:
            self.reinit_calibrations()
            # Вызов метода расчета калибровочных коэффициентов
            if self.flag_calc_calibrations:
                self.__calc_all_calibrations(self.rsb_calibr_dict, self.muo_calibr_dict, self.bo_calibr_dict, t)

            self.__calc_water_pvt_parameters(p, t)
            self.__calc_gas_pvt_parameters(p, t)
            self.__calc_oil_pvt_parameters(p, t)

    def __calc_water_pvt_parameters(self, p: float, t: float):
        """
        Метод расчета PVT - параметров для воды

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :return: метод изменяет атрибуты класса BlackOil, отвечающие за pvt-свойства воды
        -------
        """

        self.bw = self.bw_func(p=p, t=t, pvt_property="bw")
        self.salinity = self.salinity_func(p=p, t=t, gamma_wat=self.gamma_wat, pvt_property="salinity")
        self.rho_wat = self.rho_wat_func(
            t=t,
            p=p,
            pvt_property="rho_wat",
            gamma_wat=self.gamma_wat,
            bw=self.bw,
            salinity=self.salinity,
        )
        self.muw = self.muw_func(p=p, t=t, pvt_property="muw", salinity=self.salinity)
        self.heat_capacity_wat = self.hc_wat_func(t=t, p=p, pvt_property="hc_wat")
        self.st_wat_gas = self.st_wat_gas_func(p=p, t=t, pvt_property="st_wat_gas")

    def __calc_gas_pvt_parameters(self, p: float, t: float):
        """
        Метод расчета PVT - параметров для газа

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :return:метод изменяет атрибуты класса BlackOil, отвечающие за pvt-свойства газа
        -------
        """
        self.z = self.z_func(p=p, t=t, gamma_gas=self.gamma_gas, pvt_property="z")
        self.bg = self.bg_func(p=p, t=t, z=self.z, pvt_property="bg")
        self.rho_gas = self.rho_gas_func(p=p, t=t, pvt_property="rho_gas", gamma_gas=self.gamma_gas, bg=self.bg)
        self.mug = self.mug_func(p=p, t=t, pvt_property="mug", gamma_gas=self.gamma_gas, rho_gas=self.rho_gas)
        self.heat_capacity_gas = self.hc_gas_func(p=p, t=t, gamma_gas=self.gamma_gas, pvt_property="hc_gas")

    def __calc_oil_pvt_parameters(self, p: float, t: float):
        """
        Метод расчета PVT - параметров для нефти

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :return: метод изменяет атрибуты класса BlackOil, отвечающие за pvt-свойства нефти
        -------
        """

        if self.phase_ratio["type"].lower() == "gor":
            rp = self.phase_ratio["value"]
        else:
            rp = self.phase_ratio["value"] / (1 - self.wct)

        if rp == 0 and self.rsb_calibr_dict is None:
            self.pb = const.ATM
            self.rs = 0
            rsb = 0
        else:
            # Вызов метода расчета давления насыщения
            if self.use_table_model and self.table_model_data is not None and "pb" in self.table_model_data:
                self.pb = self.table_model.calc_property(p, t, "pb")
                rsb = self.oil_corrs.calc_rs(self.pb, t, self.gamma_oil, self.gamma_gas)
            else:
                if self.flag_calc_calibrations:
                    self.pb = self.rsb_calibr_dict["p"]
                    rsb = rp
                else:
                    self.pb = min(
                        self.oil_corrs.calc_pb(t, rp, self.gamma_oil, self.gamma_gas),
                        const.PB_MAX,
                    )
                    rp = self.oil_corrs.calc_rs(self.pb, t, self.gamma_oil, self.gamma_gas)

                    if self.phase_ratio["type"].lower() == "gor":
                        self.phase_ratio = {"type": "GOR", "value": rp}
                    else:
                        self.phase_ratio = {"type": "GLR", "value": rp * (1 - self.wct)}

                    rsb = self.oil_corrs.calc_rs(self.pb, t, self.gamma_oil, self.gamma_gas)
            self.pb = min(self.pb * self.calibr_pb, const.PB_MAX)

            # Вызов метода расчета газосодержания
            if self.use_table_model and self.table_model_data is not None and "rs" in self.table_model_data:
                self.rs = self.table_model.calc_property(p, t, "rs")
            else:
                if p < self.pb:
                    self.rs = self.oil_corrs.calc_rs(p, t, self.gamma_oil, self.gamma_gas)
                    self.rs *= self.calibr_rs
                else:
                    self.rs = rsb

        self.compro = self.compro_func(
            t=t,
            p=p,
            pvt_property="compro",
            gamma_oil=self.gamma_oil,
            gamma_gas=self.gamma_gas,
            rsb=self.rs,
        )

        # Вызов метода расчета объемного коэффициента нефти
        if self.use_table_model and self.table_model_data is not None and "bo" in self.table_model_data:
            self.bo = self.table_model.calc_property(p, t, "bo")
        else:
            if p <= self.pb:
                self.bo = self.oil_corrs.calc_oil_fvf(
                    p, t, self.rs, self.gamma_oil, self.gamma_gas, self.compro, self.pb
                )
                if self.bo_calibr_dict is not None and None not in self.bo_calibr_dict.values():
                    rs_tm_pm = (
                        self.oil_corrs.calc_rs(
                            self.bo_calibr_dict["p"],
                            self.bo_calibr_dict["t"],
                            self.gamma_oil,
                            self.gamma_gas,
                        )
                        * self.calibr_rs
                    )
                    self.bo = self.bo + (1 - self.calibr_bo) * min(self.rs / rs_tm_pm, 1)
            else:
                self.bo = self.oil_corrs.calc_oil_fvf(
                    p,
                    t,
                    self.rs,
                    self.gamma_oil,
                    self.gamma_gas,
                    self.compro,
                    self.pb,
                    self.bob(t, self.pb, rsb, self.compro),
                )

        self.rho_oil = self.rho_oil_func(
            p=p,
            t=t,
            pvt_property="rho_oil",
            rs=self.rs,
            bo=self.bo,
            gamma_oil=self.gamma_oil,
            gamma_gas=self.gamma_gas,
        )
        if self.table_mu:
            self.muo = self.__muoviscuserdata(p=p, t=t)
            if p <= self.p_table[-1]:
                self.mul = self.viscousertable(p=p, t=t, wct=self.wct)[0]
        else:
            self.muo = self.muo_func(
                p=p,
                t=t,
                pvt_property="muo",
                pb=self.pb,
                gamma_oil=self.gamma_oil,
                rs=self.rs,
                calibr_mu=self.calibr_mu,
            )
        self.heat_capacity_oil = self.hc_oil_func(t=t, gamma_oil=self.gamma_oil, p=p, pvt_property="hc_oil")
        self.st_oil_gas = self.st_oil_gas_func(
            t=t, p=p, pvt_property="st_oil_gas", gamma_oil=self.gamma_oil, rs=self.rs
        )

    def bob(self, t: float, pb: float, rsb: float, compro: float) -> float:
        """
        Метод расчета объемного коэффициента нефти при давлении насыщения

        Parameters
        ----------
        :param t: температура, К
        :param pb: давление насыщения, Па
        :param rsb: газосодержание при давлении насыщения, м3/м3
        :param compro: сжимаемость нефти, 1/Па

        :return: объемный коэффициент нефти при давлении насыщения, м3/м3
        -------
        """
        self.__bob = self.oil_corrs.calc_oil_fvf(pb, t, rsb, self.gamma_oil, self.gamma_gas, compro, pb)
        if self.bo_calibr_dict is not None and None not in self.bo_calibr_dict.values():
            rs_tm_pm = (
                self.oil_corrs.calc_rs(
                    self.bo_calibr_dict["p"],
                    self.bo_calibr_dict["t"],
                    self.gamma_oil,
                    self.gamma_gas,
                )
                * self.calibr_rs
            )
            self.__bob = self.__bob + (1 - self.calibr_bo) * min(rsb / rs_tm_pm, 1)
        return self.__bob

    def __calc_all_calibrations(self, rsb: dict, muob: dict, bob: dict, t: float):
        """
        Метод расчета калибровочных коэффициентов для газосодержания, давления насыщения,
        объемного коэффициента нефти и вязкости нефти

        Parameters
        ----------
        :param rsb: словарь с калибровочным значением газосодержания при давлении насыщения
                    "value" - калибровочное значение газосодержания при давлении насыщения,
                    ст. м3 газа/ст. м3 нефти
                    "p" - давление насыщения, Па абс.
                    "t" - температура калибровки газосодержания при давлении калибровки, К
        :param bob: словарь с калибровочным значением объемного
        коэффициента нефти при давлении калибровки
                    "value" - калибровочное значение
                    объемного коэффициента нефти при давлении насыщения, ст.м3/ст.м3
                    "p" - давление калибровки, Па абс.
                    "t" - температура калибровки объемного
                    коэффициента нефти при давлении насыщения, К
        :param muob: словарь с калибровочным значением вязкости нефти при давлении калибровки
                    "value" - калибровочное значение вязкости нефти при давлении калибровки, сПз
                    "p" - давление калибровки, Па абс.
                    "t" - температура калибровки вязкости нефти при давлении насыщения, К
        :param t: температура, К
        :return: метод изменяет атрибуты класса BlackOil,
        отвечающие за калибровочные коэффициенты для
        давления насыщения, газосодержания, объемного коэффициента нефти и вязкости нефти
        -------
        """

        if self.phase_ratio["type"].lower() == "gor":
            rp = self.phase_ratio["value"]
        else:
            rp = self.phase_ratio["value"] / (1 - self.wct)

        # FIXME: Убрать большое количество комментариев, когда будет документация
        # --- 1. Расчет калибровки для газосодержания и давления насыщения ---
        # 1.1 Расчет калибровочного коэффициента для давления насыщения для случаев,
        # когда rsb != rp, а также
        # при rsb = rp, но t <> rsb["t"]
        if rsb["value"] < rp or rsb["value"] > rp or (rsb["value"] == rp and t != rsb["t"]):
            # Расчетное Рнас при пластовой температуре и rp:
            pb_rp = self.oil_corrs.calc_pb(t, rp, self.gamma_oil, self.gamma_gas)
            # Расчетное Рнас при температуре калибровки и калибровочном газосодержании:
            pb_rsb = self.oil_corrs.calc_pb(
                rsb["t"],
                rsb["value"],
                self.gamma_oil,
                self.gamma_gas,
            )
            # Расчет калибровочного коэффициента:
            self.calibr_pb = pb_rp / pb_rsb
        else:
            self.calibr_pb = 1

        # Расчетное газосодержание при заданных p и t калибровки:
        rs_calc = self.oil_corrs.calc_rs(rsb["p"], rsb["t"], self.gamma_oil, self.gamma_gas)
        # 1.2 Расчет калибровочного коэффициента для газосодержания
        # с учетом заданного калибровочного значения газосодержания
        self.calibr_rs = rsb["value"] / rs_calc

        # --- 2. Расчет калибровки для вязкости ---
        if muob is not None and None not in muob.values():
            # Расчетное газосодержание при заданных p и t калибровки для вязкости:
            rs = self.oil_corrs.calc_rs(muob["p"], muob["t"], self.gamma_oil, self.gamma_gas) * self.calibr_rs
            # Рнас принимается равным давлению калибровки для газосодержания:
            pb = rsb["p"] * self.calibr_pb

            # Расчет вязкости газонасыщенной нефти при заданных p и t калибровки для вязкости
            muo_live_tm_pm = self.oil_corrs.calc_oil_viscosity(
                muob["p"] * self.calibr_pb,
                muob["t"],
                pb,
                self.gamma_oil,
                rs,
                self.calibr_mu,
            )
            # Расчет вязкости дегазированной нефти, при заданных p и t калибровки для вязкости
            muo_dead_tm_pm = self.oil_corrs.calc_oil_viscosity(
                pb - const.ATM, muob["t"], pb, self.gamma_oil, 0, self.calibr_mu
            )
            # 2.1 Расчет калибровочного коэффициента для вязкости газонасыщенной нефти
            self.calibr_mu = (1 / muob["value"] - 1 / muo_dead_tm_pm) / (1 / muo_live_tm_pm - 1 / muo_dead_tm_pm)

            # Корректировка калибровочного коэффициента (как в Pipesim)
            self.calibr_mu = max(self.calibr_mu, 0.1)
            self.calibr_mu = min(self.calibr_mu, 10)
        else:
            self.calibr_mu = 1

        # --- 3. Расчет калибровки для объемного коэффициента ---
        if bob is not None and None not in bob.values():
            # Расчетное газосодержание при заданных p и t калибровки для объемного коэффициента:
            rs = self.oil_corrs.calc_rs(bob["p"], bob["t"], self.gamma_oil, self.gamma_gas) * self.calibr_rs
            # Рнас принимается равным давлению калибровки для газосодержания:
            pb = rsb["p"] * self.calibr_pb

            # Расчет объемного коэффициента нефти при заданных
            # p и t калибровки для объемного коэффициента
            bo_tm_pm = self.oil_corrs.calc_oil_fvf(
                bob["p"] * self.calibr_pb,
                bob["t"],
                rs,
                self.gamma_oil,
                self.gamma_gas,
                self.compro,
                pb,
            )
            # 3.1 Расчет калибровочного коэффициента для объемного коэффициента нефти
            self.calibr_bo = 1 - (bob["value"] - bo_tm_pm)

            # Корректировка калибровочного коэффициента (как в Pipesim)
            self.calibr_bo = max(0.3, self.calibr_bo)
            self.calibr_bo = min(3, self.calibr_bo)
        else:
            self.calibr_bo = 1
