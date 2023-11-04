from copy import deepcopy
from typing import Optional, Union

from scipy.interpolate import interp1d

from unifloc.equipment._mech_marquez_corr import MechMarquezCorr
from unifloc.equipment.esp_system import EspSystem
from unifloc.pipe.annulus import Annulus
from unifloc.service._constants import DISTRS
from unifloc.service._tools import make_unified_distributions
from unifloc.tools.common_calculations import adapt_choke
from unifloc.tools.exceptions import OptimizationStatusError, UniflocPyError
from unifloc.well._well import AbstractWell


class EspWell(AbstractWell):
    """
    Класс для расчета скважины с ЭЦН

    Принимает на вход словари с исходными данными.

    Структура словарей исходных данных:

    * fluid_data ("ключ" - определение - тип) - dict
        * "q_fluid" - дебит жидкости в ст.у., ст. м/c - float
        * "wct" - объемная обводненность, доли ед. - float
        * "pvt_model_data" - словарь с флюидами - dict
            * "black_oil" - словарь с свойствами нефти, газа и воды для модели Black Oil - dict
                * "gamma_gas" - относительная плотность газа по воздуху в ст.у.
                (плотность воздуха = 1.2217 кг/м3)-float
                * "gamma_oil" - относительная плотность нефти по воде в ст.у.
                (плотность воды = 1000 кг/м3) - float
                * "gamma_wat" - относительная плотность воды по воде в ст.у.
                (плотность воды = 1000 кг/м3) - float
                * "rp" - газовый фактор, ст. м3 газа/ст. м3 нефти - float
                * "oil_correlations" - словарь с набором корреляций для нефти - dict, optional
                    * "pb" - название корреляции для давления насыщения - string, optional
                        * Возможные значения: "Standing"
                    * "rs" - название корреляции для газосодержания - string, optional
                        * Возможные значения: "Standing"
                    * "rho" - название корреляции для плотности нефти - string, optional
                        * Возможные значения: "Standing"
                    * "mu" - название корреляции для вязкости нефти - string, optional
                        * Возможные значения: "Beggs"
                    * "b" - название корреляции для объемного коэффициента нефти - string, optional
                        * Возможные значения: "Standing"
                    * "compr" - название корреляции для сжимаемости нефти - string, optional
                        * Возможные значения: "Vasquez"
                * "gas_correlations" - словарь с набором корреляций для газа - dict, optional
                    * "ppc" - название корреляции для критического давления - string, optional
                        * Возможные значения: "Standing"
                    * "tpc" - название корреляции для критической температуры - string, optional
                        * Возможные значения: "Standing"
                    * "z" - название корреляции для z-фактора - string, optional
                        * Возможные значения: "Kareem", "Dranchuk"
                    * "mu" - название корреляции для вязкости газа - string, optional
                        * Возможные значения: "Lee"
                * "water_correlations" - словарь с набором корреляций для газа - dict, optional
                    * "b" - название корреляции для объемного коэффициента воды - string, optional
                        * Возможные значения: "McCain"
                    * "rho" - название корреляции для плотности воды - string, optional
                        * Возможные значения: "Standing", "IAPWS"
                    * "mu" - название корреляции для вязкости воды - string, optional
                        * Возможные значения: "McCain", "IAPWS"
                    * "compr" - название корреляции для сжимаемости воды - string, optional
                        * Возможные значения: "Kriel"
                * "salinity" - минерализация воды, ppm - float, optional
                * "rsb" - словарь с калибровочным значением газосодержания при давлении насыщения -
                dict, optional
                    * "value" - калибровочное значение газосодержания при давлении насыщения,
                    ст. м3 газа/ст. м3 нефти - float
                    * "p" - давление калибровки, Па абс. - float
                    * "t" - температура калибровки газосодержания, К - float
                * "bob" - словарь с калибровочным значением объемного коэффициента нефти при
                давлении насыщения - dict, optional
                    * "value" - калибровочное значение объемного коэффициента нефти при давлении
                    насыщения, ст. м3/ст. м3 - float
                    * "p" - давление калибровки, Па изб. - float
                    * "t" - температура калибровки объемного коэффициента нефти, К - float
                * "muob" - словарь с калибровочным значением вязкости нефти при давлении насыщения
                - dict, optional
                    * "value" - калибровочное значение вязкости нефти при давлении насыщения, сПз - float
                    * "p" - давление калибровки, Па изб. - float
                    * "t" - температура калибровки вязкости нефти, К - float
                * "table_model_data" - словарь с исходными данными табличной модели - dict, optional
                    * "pvt_dataframes_dict" - словарь с таблицами с исходными данными -
                     dict of DataFrames
                    * "interp_type" -  тип интерполяции (по умолчанию - линейный) - string, optional
                * "use_table_model" - флаг использования табличной модели - boolean, optional
    * pipe_data ("ключ" - определение - тип) - dict
        * "casing" - словарь с исходными данными для создания ЭК - dict
            * "bottom_depth" - измеренная глубина верхних дыр перфорации, м - float
            * "d" - внутренний диаметр ЭК, м - float, pd.DataFrame("MD", "d")
            * ! можно задавать как числом, так и таблицей с распределением по глубине или словарем,
             см. пример
            * "roughness" - шероховатость, м - float
        * "tubing" - словарь с исходными данными для создания колонны НКТ - dict
            * "bottom_depth" - измеренная глубина спуска колонны НКТ, м - float
            * "d" - внутренний диаметр колонны НКТ, м - float, pd.DataFrame("MD", "d")
            * ! можно задавать как числом, так и таблицей с распределением по глубине или словарем,
             см. пример
            * "roughness" - шероховатость, м - float
            * "s_wall" - толщина стенки, м - float
    * well_trajectory_data ("ключ" - определение - тип) - dict
        * "inclinometry" - таблица с инклинометрией, две колонки: "MD","TVD", индекс по умолчанию,
         см.пример - DataFrame
        * или возможно с помощью dict с ключами "MD", "TVD"
        * Важно!: физичность вводимых данных пока не проверяется, поэтому нужно смотреть чтобы
         TVD <= MD, dTVD <= dMD
    * ambient_temperature_data ("ключ" - определение - тип) - словарь с распределением температуры
     породы по MD - dict
        * обязательные ключи MD, T - list
    * equipment_data ("ключ" - определение - тип) - dict
        * "choke" - словарь с исходными данными для создания объекта штуцера - dict, optional
            * "d" - диаметр штуцера, м - float
        * "packer" - флаг наличия пакера, True/False, optional
        * "esp_system" - словарь с исходными данными для создания объекта системы УЭЦН - dict
            * "esp" - словарь с данными для объекта ЭЦН
                * "h_mes" - глубина на которой установлен насос, м - float
                * "stages" - количество ступеней насоса - integer
                * "esp_data" - паспортные данные ЭЦН из базы насосов - pd.Series или dict,
                 см.ниже пример
                * "viscosity_correction" - флаг учета поправки на вязкость - boolean
                * "gas_correction" - флаг учета поправки на газ - boolean
            * "esp_electric_system" - словарь с исходными данными для электрической части
             УЭЦН - dict, optional
                * "motor_data" - паспортные данные ПЭД - dict
                * "gassep_nom_power" - номинальная мощность газосепаратора, Вт - float, optional
                * "protector_nom_power" - номинальная мощность протектора, Вт - float, optional
                * "transform_eff" - КПД трансформатора, д.ед. - float, optional
                * "cs_eff" - КПД станции управления, д.ед. - float, optional
                * "cable_specific_resistance" - удельное сопротивление  электрического кабеля, Ом/1000 м  - float,
                 optional
                * "cable_length" - длина электрического кабеля, м - float, optional
            * "separator" - словарь с исходными данными для объекта газосепаратор - dict, optional
                * "k_gas_sep" - коэффициент газосепарации сепаратора, доли ед. - float
                * "sep_name" - название сепаратора в соответствии с БД, str
    * nat_separation - флаг, включающий расчет естественной сепарации газа на приеме ЭЦН - bool, optional
    """

    def __init__(
        self,
        fluid_data: dict,
        pipe_data: dict,
        equipment_data: dict,
        well_trajectory_data: dict,
        ambient_temperature_data: dict,
        nat_separation: bool = False,
    ):
        """
        Parameters
        ----------
        :param fluid_data: словарь с исходными данными для создания флюида
        :param pipe_data: словарь с исходными данными для создания колонн труб
        :param equipment_data: словарь с исходными данными для создания различного оборудования
        :param well_trajectory_data: словарь с исходными данными для создания инклинометрии скважины
        :param ambient_temperature_data: словарь с распределением температуры породы по MD
        :param nat_separation: флаг расчета естественной сепарации - bool, optional

        Examples:
        --------
        >>> # Пример исходных данных для насоса
        >>> import pandas as pd
        >>> import unifloc.tools.units_converter as uc
        >>> from unifloc.well.esp_well import EspWell
        >>> # Если мы хотим задать насос вручную, то нам соответственно нужно заполнить все строки
        >>> # Например, при помощи словаря:
        >>> esp_data = {"ID": 99999,
        ...         "source": "legacy",
        ...         "manufacturer": "Reda",
        ...         "name": "DN500",
        ...         "stages_max": 400,
        ...         "rate_nom_sm3day": 30,
        ...         "rate_opt_min_sm3day":20,
        ...         "rate_opt_max_sm3day":40,
        ...         "rate_max_sm3day": 66,
        ...         "slip_nom_rpm": 3000,
        ...         "freq_Hz": 50,
        ...         "eff_max": 0.4,
        ...         "height_stage_m": 0.035,
        ...         "Series": 4,
        ...         "d_od_mm": 86,
        ...         "d_cas_min_mm": 112,
        ...         "d_shaft_mm": 17,
        ...         "area_shaft_mm2": 227,
        ...         "power_limit_shaft_kW": 72,
        ...         "power_limit_shaft_high_kW": 120,
        ...         "power_limit_shaft_max_kW": 150,
        ...         "pressure_limit_housing_atma": 390,
        ...         "d_motor_od_mm": 95,
        ...         "rate_points": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 66],
        ...         "head_points": [4.88, 4.73, 4.66, 4.61, 4.52, 4.35, 4.1, 3.74, 3.28, 2.73, 2.11, 1.45, 0.77, 0],
        ...         "power_points": [0.02, 0.022, 0.025, 0.027, 0.03, 0.032, 0.035, 0.038, 0.041, 0.043, 0.046, 0.049,
        ...                            0.052000000000000005, 0.055],
        ...         "eff_points": [0, 0.12, 0.21, 0.29, 0.35, 0.38, 0.4, 0.39, 0.37, 0.32, 0.26, 0.19, 0.1, 0]
        ...         }
        >>> # Можно задавать с помощью dict - esp_data = data
        >>> esp_data = pd.Series(esp_data,name=esp_data["ID"])
        >>> print(esp_data)
        ID                                                                         99999
        source                                                                    legacy
        manufacturer                                                                Reda
        name                                                                       DN500
        stages_max                                                                   400
        rate_nom_sm3day                                                               30
        rate_opt_min_sm3day                                                           20
        rate_opt_max_sm3day                                                           40
        rate_max_sm3day                                                               66
        slip_nom_rpm                                                                3000
        freq_Hz                                                                       50
        eff_max                                                                      0.4
        height_stage_m                                                             0.035
        Series                                                                         4
        d_od_mm                                                                       86
        d_cas_min_mm                                                                 112
        d_shaft_mm                                                                    17
        area_shaft_mm2                                                               227
        power_limit_shaft_kW                                                          72
        power_limit_shaft_high_kW                                                    120
        power_limit_shaft_max_kW                                                     150
        pressure_limit_housing_atma                                                  390
        d_motor_od_mm                                                                 95
        rate_points                    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55,...
        head_points                    [4.88, 4.73, 4.66, 4.61, 4.52, 4.35, 4.1, 3.74...
        power_points                   [0.02, 0.022, 0.025, 0.027, 0.03, 0.032, 0.035...
        eff_points                     [0, 0.12, 0.21, 0.29, 0.35, 0.38, 0.4, 0.39, 0...
        Name: 99999, dtype: object

        >>> # Исходные данные для ПЭД
        >>> motor_data = {
        ...     "ID": 1,
        ...     "manufacturer": "Centrilift",
        ...     "name": "562Centrilift-KMB-130-2200B",
        ...     "d_motor_mm": 142.7,
        ...     "motor_nom_i": 35,
        ...     "motor_nom_power": 96.98,
        ...     "motor_nom_voltage": 2200,
        ...     "motor_nom_eff": 80,
        ...     "motor_nom_cosf": 0.82,
        ...     "motor_nom_freq": 60,
        ...     "load_points": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        ...     "amperage_points": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        ...     "cosf_points": [0.7, 0.74, 0.77, 0.8, 0.83, 0.85, 0.86, 0.87, 0.88, 0.88],
        ...     "eff_points": [0.78, 0.83, 0.85, 0.88, 0.87, 0.87, 0.87, 0.87, 0.86, 0.86],
        ...     "rpm_points": [3568.604, 3551.63, 3534.656, 3517.682, 3500.708, 3483.734, 3466.76, 3449.786,
        ...                   3432.812, 3415.838]
        ... }
        >>> # Инициализация исходных данных
        >>> df = pd.DataFrame(columns=["MD", "TVD"], data=[[0, 0], [1400, 1200], [1800, 1542.85]])
        ... # Возможный способ задания инклинометрии через dict
        ... # df = {"MD": [0, 1000],
        ... #       "TVD": [0, 900]}
        ... # В словари с калибровками подается давление и температура калибровки.
        ... # Зачастую - это давление насыщения и пластовая температура
        >>> fluid_data = {"q_fluid": uc.convert_rate(40, "m3/day", "m3/s"),
        ...               "pvt_model_data": {
        ...                   "black_oil": {
        ...                       "gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.8,
        ...                       "wct": 0, "phase_ratio": {"type": "GOR", "value": 20},
        ...                       "oil_correlations": {
        ...                             "pb": "Standing",
        ...                             "rs": "Standing",
        ...                             "rho": "Standing",
        ...                             "b": "Standing",
        ...                             "mu": "Beggs",
        ...                             "compr": "Vasquez"
        ...                       },
        ...                       "gas_correlations": {
        ...                             "ppc": "Standing",
        ...                             "tpc": "Standing",
        ...                             "z": "Dranchuk",
        ...                             "mu": "Lee"
        ...                       },
        ...                       "water_correlations": {
        ...                             "b": "McCain",
        ...                             "compr": "Kriel",
        ...                             "rho": "Standing",
        ...                             "mu": "McCain"
        ...                       },
        ...                       "rsb": {"value": 50, "p": 10000000, "t": 303.15},
        ...                       "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
        ...                       "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
        ...                       "table_model_data": None, "use_table_model": False
        ...                   }
        ...               }}
        >>> # Диаметр можно задавать как числом так и таблицей с распределением по глубине
        >>> d = pd.DataFrame(columns=["MD", "d"], data=[[0, 0.062], [1000, 0.082]])
        >>> # Так тоже возможно: d = {"MD": [0, 1000], "d": [0.06, 0.08]}
        >>> pipe_data = {"casing": {"bottom_depth": 1800, "d": 0.146, "roughness": 0.0001},
        ...              "tubing": {"bottom_depth": 1400, "d": d, "roughness": 0.0001, "s_wall": 0.005}}
        >>> well_trajectory_data = {"inclinometry": df}
        >>> ambient_temperature_data = {"MD": [0, 1800], "T": [293.15, 303.15]}
        >>> equipment_data = {"esp_system": {
        ...     "esp":
        ...         {
        ...             "esp_data": esp_data,
        ...             "stages": 345,
        ...             "viscosity_correction": True,
        ...             "gas_correction": True,
        ...         },
        ...         "esp_electric_system": {"motor_data": motor_data,
        ...                                 "gassep_nom_power": 500,
        ...                                 "protector_nom_power": 500,
        ...                                 "transform_eff": 0.97,
        ...                                 "cs_eff": 0.97,
        ...                                 "cable_specific_resistance": 1.18,
        ...                                 "cable_length": 1450,
        ...                                 },
        ...         "separator": {"k_gas_sep": 0.7, "sep_name": '3МНГДБ5'}
        ... },
        ...     "packer": False
        ... }
        >>> # Текущая частота
        >>> freq = 53
        >>> # Затрубное давление
        >>> p_ann = 10 * 101325
        >>> # Инициализация объекта скважины
        >>> well = EspWell(fluid_data, pipe_data, equipment_data, well_trajectory_data,
        ...                ambient_temperature_data)
        >>> # Расчет с сохранением доп. атрибутов распределений свойств
        >>> result = well.calc_pwf_pfl(
        ...     uc.convert_pressure(12, "atm", "Pa"),
        ...     uc.convert_rate(40, "m3/day", "m3/s"),
        ...     0.1,
        ...     freq,
        ...     p_ann=p_ann,
        ...     output_params=True
        ... )
        >>> # Запрос всех значений доп. свойств в виде словаря
        >>> extra_results = well.extra_output
        """
        # Считаем что скважина ЭЦН всегда без пакера и в ней есть естественная сепарация
        super().__init__(
            fluid_data,
            pipe_data,
            equipment_data,
            well_trajectory_data,
            ambient_temperature_data,
        )

        if "esp_system" in equipment_data:
            equipment_data["esp_system"]["esp"].update({"fluid": self.fluid, "h_mes": self.tubing.bottom_depth})

            if "esp_electric_system" in equipment_data["esp_system"]:
                self.__freq_base = equipment_data["esp_system"]["esp"]["esp_data"]["freq_Hz"]
                equipment_data["esp_system"]["esp_electric_system"].update({"pump_nom_freq": self.__freq_base})
                self.__q_liq_max = max(equipment_data["esp_system"]["esp"]["esp_data"]["rate_points"]) / 86400

                if equipment_data["esp_system"]["esp_electric_system"].get("cable_length"):
                    if equipment_data["esp_system"]["esp_electric_system"]["cable_length"] < self.tubing.bottom_depth:
                        equipment_data["esp_system"]["esp_electric_system"]["cable_length"] = self.tubing.bottom_depth
                else:
                    equipment_data["esp_system"]["esp_electric_system"].update(
                        {"cable_length": self.tubing.bottom_depth}
                    )

            if "separator" in equipment_data["esp_system"]:
                if equipment_data["esp_system"]["separator"]:
                    equipment_data["esp_system"]["separator"].update({"h_mes": self.tubing.bottom_depth})

            self.esp_system = EspSystem(**equipment_data["esp_system"])
        else:
            self.esp_system = None

        if self.natural_sep:
            self.annulus = Annulus(
                fluid=self.fluid,
                bottom_depth=pipe_data["tubing"]["bottom_depth"],
                ambient_temperature_distribution=self.amb_temp_dist,
                d_casing=pipe_data["casing"]["d"],
                d_tubing=pipe_data["tubing"]["d"],
                s_wall=pipe_data["tubing"]["s_wall"],
                roughness=pipe_data["casing"]["roughness"],
                trajectory=self.well_trajectory,
            )
            if "d_od_mm" in equipment_data["esp_system"]["esp"]["esp_data"]:
                esp_d_od = equipment_data["esp_system"]["esp"]["esp_data"]["d_od_mm"] / 1000
            else:
                if isinstance(self.annulus.d_tub_out, interp1d):
                    esp_d_od = self.annulus.d_tub_out(self.esp_system.esp.h_mes).item()
                else:
                    esp_d_od = self.annulus.d_tub_out

            if isinstance(self.annulus.d_cas_in, interp1d):
                d_cas = self.annulus.d_cas_in(self.esp_system.esp.h_mes).item()
            else:
                d_cas = self.annulus.d_cas_in
            self.natural_sep = nat_separation
            if nat_separation:
                self.nat_sep_marquez = MechMarquezCorr(esp_d_od, d_cas)
        else:
            self.annulus = None

        self.p_dis = None

    @property
    def extra_output(self):
        """
        Сборный атрибут со всеми требуемыми распределениями
        """
        self._extra_output.update(
            {
                "rate_points_corr": self.esp_system.rate_points_corr.tolist(),
                "head_points_corr": self.esp_system.head_points_corr.tolist(),
                "power_points_corr": self.esp_system.power_points_corr.tolist(),
                "eff_points_corr": self.esp_system.eff_points_corr.tolist(),
            }
        )
        return self._extra_output

    def __make_nodes(self):
        """
        Создание распределений ключевых узлов

        :return: распределение ключевых узлов
        """
        nodes_tub = [None for _ in self.tubing.distributions["depth"]]
        nodes_tub[0] = "Буфер"
        nodes_esp = [None for _ in self.esp_system.distributions["depth"]]
        nodes_esp[0] = "Выкид ЭЦН"
        nodes_esp[-1] = "Прием ЭЦН"
        nodes_cas = [None for _ in self.casing.distributions["depth"]]
        nodes_cas[-1] = "Верхние дыры перфорации"
        result = nodes_tub + nodes_esp + nodes_cas

        if self.choke and "depth" in self.choke.distributions:
            nodes_ch = [None for _ in self.choke.distributions["depth"]]
            nodes_ch[0] = "Линия"
            result = nodes_ch + result
        return result

    def _calc_p_dis(
        self,
        p_in: float,
        t_in: float,
        q_liq: float,
        wct: float,
        freq: float,
        p_ann: float,
        head_factor: float,
        step_length: float,
        output_params: bool,
        finalize: bool,
        optimizer: bool,
        c_pump_power: float,
        c_load_i: float,
        c_transform_voltage: float,
        c_cs_power: float,
    ) -> tuple:
        """
        Расчет параметров ЭЦН и естественной сепарации.

        :param p_in: давление на приеме ЭЦН, Па
        :param t_in: температура на приеме ЭЦН, Па
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param freq: частота вращения вала ЭЦН, Гц
        :param p_ann: давление в затрубе, Па
        :param head_factor: коэффициент поправки на напор, д.ед. - optional
        :param step_length: длина шага интегрирования
        :param output_params: флаг вывода дополнительных параметров
        :param finalize: рассчитать общие распределения для скважины
        :param optimizer: опция считать ли доп. параметры (False)
            или считать только необходимое (для оптимизатора-True)
        :param c_pump_power: адаптационный коэффициент для мощности насоса
        :param c_load_i: адаптационный коэффициент для загрузки по току
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ

        :return: давление на выкиде ЭЦН, Па
        """
        self.fluid.reinit()
        self.fluid.calc_flow(p_in, t_in)
        # Если нет пакера и хотим считать естественную сепарацию
        if self.natural_sep:
            k_sep_nat = self.nat_sep_marquez.calc_k_sep(
                ql=self.fluid.ql,
                qg=self.fluid.qg,
                mul=self.fluid.mul,
                mum=self.fluid.mum,
                rho_gas=self.fluid.rho_gas,
                rho_mix=self.fluid.rm,
                rho_liq=self.fluid.rl,
                stlg=self.fluid.stlg,
            )
        # Если нет пакера и не хотим считать естественную сепарацию
        elif self.annulus:
            k_sep_nat = 0.5
        # Есть пакер >> естественной сепарации нет
        else:
            k_sep_nat = 0

        k_sep = self.esp_system.calc_general_separation(k_sep_nat, self.fluid.gf, q_liq, freq)

        # Модификация флюида и обновление объектов
        self.fluid.modify(p_in, t_in, k_sep)
        if not optimizer:
            self.fluid.calc_flow(p_in, t_in)

        # Доля газа на приеме насоса
        gas_fraction = self.fluid.gf
        # Дебит ГЖС в условиях приема
        qm_in = self.fluid.qm

        self.esp_system.esp.fluid = deepcopy(self.fluid)

        # Расчет давления и температуры на выкиде от приема
        t_cable = t_in

        # Расчет насоса
        results = self.esp_system.calc_esp_system(
            q_liq=q_liq,
            wct=wct,
            p=p_in,
            t=t_in,
            freq=freq,
            t_cable=t_cable,
            direction_to="dis",
            head_factor=head_factor,
            extra_output=output_params,
            c_pump_power=c_pump_power,
            c_load_i=c_load_i,
            c_transform_voltage=c_transform_voltage,
            c_cs_power=c_cs_power,
        )

        # Неуспешный случай расчета ЭЦН
        if results[2] != 0 and optimizer:
            raise OptimizationStatusError

        h_dyn = None

        # Сохранение доп. атрибута
        if finalize:
            self._output_objects["esp_sys"] = self.esp_system
            self.p_dis = results[0]
            if results[2] != 0:
                self._extra_output = make_unified_distributions(**self._output_objects)

        # Дополнительный расчет параметров
        if not optimizer:
            if self.annulus:
                # Расчет динамического уровня
                if p_ann is None:
                    raise UniflocPyError("Не задано затрубное давление. " "Расчет динамического уровня невозможен")
                self.fluid.reinit()
                self.annulus.fluid = deepcopy(self.fluid)
                self.annulus.fluid.modify(p_in, t_in, k_sep, calc_type="annulus")
                h_dyn, _ = self.annulus.calc_hdyn(p_esp=p_in, p_ann=p_ann, wct=wct, step_length=step_length)

        return results, h_dyn, k_sep, gas_fraction, qm_in

    def calc_pwh_pin(
        self,
        p_in: float,
        t_in: float,
        q_liq: float,
        wct: float,
        freq: float,
        p_ann: Optional[float] = None,
        hydr_corr_type: Optional[str] = None,
        head_factor: Optional[float] = None,
        step_length: float = 10,
        output_params: bool = False,
        heat_balance: bool = False,
        optimizer: bool = False,
        finalize: bool = False,
        c_pump_power: float = 1,
        c_load_i: float = 1,
        c_transform_voltage: float = 1,
        c_cs_power: float = 1,
    ):
        """
        Расчет буферного давления по давлению на приеме.

        :param p_in: давление на приеме, Па абс.
        :param t_in: температура на приеме, К
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, доли ед.
        :param freq: частота вращения вала ЭЦН, Гц
        :param p_ann: затрубное давление, Па изб.
        :param hydr_corr_type: тип гидравлической корреляции
        :param head_factor: к-т адаптации на напор ЭЦН
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг для расчета распределений параметров
        :param heat_balance: опция расчета теплопотерь
        :param optimizer: опция считать ли доп. параметры (False)
            или считать только необходимое (для оптимизатора-True)
        :param finalize: рассчитать общие распределения для скважины
        :param c_pump_power: адаптационный коэффициент для мощности насоса
        :param c_load_i: адаптационный коэффициент для загрузки по току
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ

        :return: буферное давление, Па абс.
        :return: температура на буфере, К
        :return: мощность на станции управления, Вт
        :return: динамический уровень, м
        :return: напряжение на трансформаторе, В
        :return: сила тока ПЭД, А
        :return: загрузка ПЭД, доли ед.
        :return: КПД насоса факт., доли ед.
        :return: КПД системы, доли ед.
        :return: Коэффициент сепарации общий, доли ед.
        :return: давление на приеме, Па изб.
        :return: температура ПЭД, К
        :return: мощность ЭЦН, Вт
        :return: мощность ПЭД, Вт
        :return: доля газа на приеме насоса, доли ед.
        :return: статус расчета
        :return: дебит ГЖС в условиях приема насоса, м3/с
        """
        res, h_dyn, k_sep, gas_fraction, qm_in = self._calc_p_dis(
            p_in=p_in,
            t_in=t_in,
            q_liq=q_liq,
            wct=wct,
            freq=freq,
            p_ann=p_ann,
            head_factor=head_factor,
            step_length=step_length,
            output_params=output_params,
            finalize=finalize,
            optimizer=optimizer,
            c_pump_power=c_pump_power,
            c_load_i=c_load_i,
            c_transform_voltage=c_transform_voltage,
            c_cs_power=c_cs_power,
        )

        p_dis, t_dis, _ = res[:3]
        el_params = res[3:]

        self.tubing.fluid = deepcopy(self.fluid)

        # Расчет давления и температуры на буфере
        p_wh, t_wh, status = self.tubing.calc_pt(
            h_start="bottom",
            p_mes=p_dis,
            flow_direction=-1,
            q_liq=q_liq,
            wct=wct,
            phase_ratio_value=None,
            t_mes=t_dis,
            hydr_corr_type=hydr_corr_type,
            step_len=step_length,
            extra_output=output_params,
            heat_balance=heat_balance,
        )

        if finalize:
            # Сохранение доп. атрибутов
            self._output_objects["tubings"] = [self.tubing]

        if status != 0:
            # Случай неуспешного расчета до буфера
            if finalize:
                # Сохранение доп. атрибутов
                self._extra_output = make_unified_distributions(**self._output_objects)
            if optimizer:
                raise OptimizationStatusError
        return (
            p_wh,
            t_wh,
            el_params[9],
            h_dyn,
            el_params[7],
            el_params[5],
            el_params[3],
            el_params[10],
            el_params[11],
            k_sep,
            p_in,
            t_in,
            el_params[2],
            el_params[4],
            gas_fraction,
            status,
            qm_in,
        )

    def calc_pwh_pwf(
        self,
        p_wf: float,
        q_liq: float,
        wct: float,
        freq: float,
        p_ann: Optional[float] = None,
        hydr_corr_type: Optional[str] = None,
        head_factor: Optional[float] = None,
        heat_balance: bool = False,
        optimizer: bool = False,
        c_pump_power: float = 1,
        c_load_i: float = 1,
        c_transform_voltage: float = 1,
        c_cs_power: float = 1,
        step_length: float = 10,
        output_params: bool = False,
        finalize: bool = False,
    ) -> list:
        """
        Расчет буферного давления по забойному с учетом всех гидравлических элементов

        :param p_wf: забойное давление, Па абс.
        :param q_liq: дебит жидкости, ст. м3/с
        :param wct: обводненность, доли ед.
        :param freq: частота работы эцн, Гц
        :param p_ann: затрубное давление, Па изб.
        :param hydr_corr_type: тип гидравлической корреляции
        :param head_factor: к-т адаптации на напор ЭЦН
        :param c_pump_power: адаптационный коэффициент для мощности насоса
        :param c_load_i: адаптационный коэффициент для загрузки по току
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг расчета распределений параметров
        :param heat_balance: опция расчета теплопотерь
        :param optimizer: опция считать ли доп. параметры (False)
            или считать только необходимое (для оптимизатора-True)
        :param finalize: рассчитать общие распределения для скважины

        :return: буферное давление, Па абс.
        :return: температура на буфере, К
        :return: мощность на станции управления, Вт
        :return: динамический уровень, м
        :return: напряжение на трансформаторе, В
        :return: сила тока ПЭД, А
        :return: загрузка ПЭД, доли ед.
        :return: КПД насоса факт., доли ед.
        :return: КПД системы, доли ед.
        :return: коэффициент сепарации общий, доли ед.
        :return: давление на приеме, Па изб.
        :return: температура ПЭД, К
        :return: мощность ЭЦН, Вт
        :return: мощность ПЭД, Вт
        :return: доля газа на приеме насоса, доли ед.
        :return: статус расчёта:
                 0 - расчет успешный;
                 1 - достигнуто минимальное давление, невозможно рассчитать линейное давление;
                 -1 - ошибка интегрирования
        :return: дебит ГЖС в условиях приема насоса, м3/с
        """
        self._reinit(wct, q_liq)
        self.casing.fluid = deepcopy(self.fluid)

        # Расчет давления, температуры на приеме по ЭК вверх
        t_wf = self.amb_temp_dist.calc_temp(self.casing.bottom_depth).item()
        p_in, t_in, status = self.casing.calc_pt(
            h_start="bottom",
            p_mes=p_wf,
            flow_direction=-1,
            q_liq=q_liq,
            wct=wct,
            phase_ratio_value=None,
            t_mes=t_wf,
            hydr_corr_type=hydr_corr_type,
            step_len=step_length,
            extra_output=output_params,
            heat_balance=heat_balance,
        )

        if optimizer and status != 0:
            # Отлавливание ошибки оптимизатора
            raise OptimizationStatusError

        if finalize:
            self._output_objects["casing"] = self.casing
            if status != 0:
                # Сохранение доп. атрибутов
                self._extra_output = make_unified_distributions(**self._output_objects)
                return [None, None, 0, None, 0, 0, 0, 0, 0, 0, p_in, t_in, 0, 0, 0, status, 0]

        # Расчет давления во всех остальных узлах (выкид, буфер, линия)
        results = list(
            self.calc_pwh_pin(
                p_in=p_in,
                t_in=t_in,
                q_liq=q_liq,
                wct=wct,
                freq=freq,
                p_ann=p_ann,
                hydr_corr_type=hydr_corr_type,
                head_factor=head_factor,
                step_length=step_length,
                output_params=output_params,
                heat_balance=heat_balance,
                optimizer=optimizer,
                finalize=finalize,
                c_pump_power=c_pump_power,
                c_load_i=c_load_i,
                c_transform_voltage=c_transform_voltage,
                c_cs_power=c_cs_power,
            )
        )
        return results

    def calc_pfl_pwf(
        self,
        p_wf: float,
        q_liq: float,
        wct: float,
        freq: float,
        p_ann: Optional[float] = None,
        hydr_corr_type: Optional[str] = None,
        head_factor: Optional[float] = None,
        c_choke: Optional[Union[float, dict]] = None,
        c_pump_power: bool = 1,
        c_load_i: float = 1,
        c_transform_voltage: float = 1,
        c_cs_power: float = 1,
        step_length: bool = 10,
        output_params: bool = False,
        heat_balance: bool = False,
    ) -> list:
        """
        Расчет линейного давления по забойному с учетом всех гидравлических элементов

        :param p_wf: забойное давление, Па абс
        :param q_liq: дебит жидкости, ст. м3/с
        :param wct: обводненность, доли ед
        :param freq: частота работы эцн, Гц
        :param p_ann: затрубное давление, Па изб
        :param hydr_corr_type: тип гидравлической корреляции
        :param head_factor: к-т адаптации на напор ЭЦН
        :param c_choke: адаптационный коэффициент штуцера\
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value},\
            где value - постоянный перепад, который будет использоваться как перепад между буферным и линейным давлением
        :param c_pump_power: адаптационный коэффициент для мощности насоса
        :param c_load_i: адаптационный коэффициент для загрузки по току
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг расчета распределений параметров
        :param heat_balance: опция расчета теплопотерь

        :return: линейное давление, Па абс.
        :return: мощность на станции управления, Вт
        :return: динамический уровень, м
        :return: напряжение на трансформаторе, В
        :return: сила тока ПЭД, А
        :return: загрузка ПЭД, доли ед.
        :return: КПД насоса факт., доли ед.
        :return: КПД системы, доли ед.
        :return: коэффициент сепарации общий, доли ед.
        :return: давление на приеме, Па изб.
        :return: температура ПЭД, К
        :return: мощность ЭЦН, Вт
        :return: мощность ПЭД, Вт
        :return: доля газа на приеме насоса, доли ед.
        :return: статус расчёта:
                 0 - расчет успешный;
                 1 - достигнуто минимальное давление, невозможно рассчитать линейное давление;
                 -1 - ошибка интегрирования
        :return: дебит ГЖС в условиях приема насоса, м3/с
        """
        # Установим дебит жидкости и обводненность в флюид и переопределим флюид для всех
        # зависимых от него классов
        self._output_objects = {"params": ["p", "t", "depth"]}

        if output_params:
            self._output_objects["params"] += DISTRS

        results = self.calc_pwh_pwf(
            p_wf=p_wf,
            q_liq=q_liq,
            wct=wct,
            freq=freq,
            p_ann=p_ann,
            hydr_corr_type=hydr_corr_type,
            head_factor=head_factor,
            heat_balance=heat_balance,
            optimizer=False,
            c_pump_power=c_pump_power,
            c_load_i=c_load_i,
            c_transform_voltage=c_transform_voltage,
            c_cs_power=c_cs_power,
            step_length=step_length,
            output_params=output_params,
            finalize=True,
        )

        p_fl, t_fl, node_status = self.calc_p_choke(
            choke=self.choke,
            p_received=results[0],
            t_received=results.pop(1),
            q_liq=q_liq,
            wct=wct,
            flow_direction=-1,
            output_params=output_params,
            c_choke=c_choke,
        )
        if node_status:
            self._output_objects["choke"] = self.choke

        self._extra_output = make_unified_distributions(**self._output_objects)
        results[0] = p_fl
        return results

    def _calc_p_dis_error(
        self,
        p_in: float,
        p_fl: float,
        t_fl: float,
        q_liq: float,
        wct: float,
        freq: float,
        p_ann: float,
        hydr_corr_type: str,
        head_factor: float,
        step_length: float,
        heat_balance: bool,
        c_choke: Union[float, dict],
        c_pump_power: float,
        c_load_i: float,
        c_transform_voltage: float,
        c_cs_power: float,
    ) -> float:
        """
        Расчет ошибки давления на выкиде ЭЦН.

        :param p_in: давление на приеме ЭЦН, Па
        :param p_fl: линейное давление, Па
        :param t_fl: температура в линии, К
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param freq: частота вращения вала ЭЦН, Гц
        :param p_ann: давление в затрубе, Па
        :param hydr_corr_type: тип гидравлической корреляции
        :param head_factor: к-т адаптации на напор ЭЦН
        :param step_length: длина шага интегрирования, м
        :param heat_balance: опция расчета теплопотерь
        :param c_choke: коэффициент калибровки штуцера
        :param c_pump_power: адаптационный коэффициент для мощности насоса
        :param c_load_i: адаптационный коэффициент для загрузки по току
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ

        :return: ошибка при сопоставлении рассчитанного давления на выкиде ЭЦН с полученным итеративно
        """
        try:
            # Расчет давления на выкиде ЭЦН от приема ЭЦН
            esp_res, *_ = self._calc_p_dis(
                p_in=p_in,
                t_in=self.amb_temp_dist.calc_temp(self.tubing.bottom_depth).item(),
                q_liq=q_liq,
                wct=wct,
                freq=freq,
                p_ann=p_ann,
                head_factor=head_factor,
                step_length=step_length,
                output_params=False,
                finalize=False,
                optimizer=True,
                c_pump_power=c_pump_power,
                c_load_i=c_load_i,
                c_transform_voltage=c_transform_voltage,
                c_cs_power=c_cs_power,
            )

            # Расчет давления на выкиде ЭЦН от линии/буфера
            if self.choke:
                self.choke.fluid = deepcopy(self.fluid)

            p_wh_calced, t_wh, node_status = self.calc_p_choke(
                choke=self.choke,
                p_received=p_fl,
                t_received=t_fl,
                q_liq=q_liq,
                wct=wct,
                flow_direction=1,
                output_params=False,
                c_choke=c_choke,
            )

            if node_status:
                self._output_objects["choke"] = self.choke

            p_dis_top, _, status = self.tubing.calc_pt(
                h_start="top",
                p_mes=p_wh_calced,
                flow_direction=1,
                q_liq=q_liq,
                wct=wct,
                phase_ratio_value=None,
                t_mes=t_wh,
                hydr_corr_type=hydr_corr_type,
                step_len=step_length,
                extra_output=False,
                heat_balance=heat_balance,
            )

            if status != 0:
                raise OptimizationStatusError
            else:
                self._output_objects["tubings"] = [self.tubing]

            return p_dis_top - esp_res[0]
        except OptimizationStatusError:
            return -9999999999

    def _calc_p_dis_error_abs(
        self,
        p_in: float,
        p_fl: float,
        t_fl: float,
        q_liq: float,
        wct: float,
        freq: float,
        p_ann: float,
        hydr_corr_type: str,
        head_factor: float,
        step_length: float,
        heat_balance: bool,
        c_choke: Union[float, dict],
        c_pump_power: float,
        c_load_i: float,
        c_transform_voltage: float,
        c_cs_power: float,
    ) -> float:
        """
        Расчет абсолютной ошибки давления на выкиде ЭЦН.

        :param p_in: давление на приеме ЭЦН, Па
        :param p_fl: линейное давление, Па
        :param t_fl: температура в линии, К
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param freq: частота вращения вала ЭЦН, Гц
        :param p_ann: давление в затрубе, Па
        :param hydr_corr_type: тип гидравлической корреляции
        :param head_factor: к-т адаптации на напор ЭЦН
        :param step_length: длина шага интегрирования, м
        :param heat_balance: опция расчета теплопотерь
        :param c_choke: коэффициент калибровки штуцера
        :param c_pump_power: адаптационный коэффициент для мощности насоса
        :param c_load_i: адаптационный коэффициент для загрузки по току
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ

        :return: абсолютная ошибка при сопоставлении рассчитанного давления на выкиде ЭЦН с полученным итеративно
        """
        return abs(
            self._calc_p_dis_error(
                p_in=p_in,
                p_fl=p_fl,
                t_fl=t_fl,
                q_liq=q_liq,
                wct=wct,
                freq=freq,
                p_ann=p_ann,
                hydr_corr_type=hydr_corr_type,
                head_factor=head_factor,
                step_length=step_length,
                heat_balance=heat_balance,
                c_choke=c_choke,
                c_pump_power=c_pump_power,
                c_load_i=c_load_i,
                c_transform_voltage=c_transform_voltage,
                c_cs_power=c_cs_power,
            )
        )

    def _calc_pwf(
        self,
        p_fl: float,
        q_liq: float,
        wct: float,
        freq: float,
        p_ann: float,
        hydr_corr_type: str,
        head_factor: float,
        step_length: float,
        output_params: bool,
        heat_balance: bool,
        c_choke: Union[float, dict],
        c_pump_power: float,
        c_load_i: float = 1,
        c_transform_voltage: float = 1,
        c_cs_power: float = 1,
    ) -> list:
        """
        Функция расчета давления на забое итеративно через давление в линии / давление на выкиде ЭЦН.

        :param p_fl: давление в линии, Па
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param freq: частота вращения вала ЭЦН, Гц
        :param p_ann: давление в затрубе, Па
        :param hydr_corr_type: тип гидравлической корреляции
        :param head_factor: к-т адаптации на напор ЭЦН
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг расчета распределений параметров
        :param heat_balance: опция расчета теплопотерь
        :param c_choke: адаптационный коэффициент штуцера\
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value},\
            где value - постоянный перепад, который будет использоваться как перепад между буферным и линейным давлением
        :param c_pump_power: адаптационный коэффициент для мощности насоса
        :param c_load_i: адаптационный коэффициент для загрузки по току
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ

        :return: давление на приеме ЭЦН / забое, Па
        """
        t_fl = self.amb_temp_dist.calc_temp(self.tubing.top_depth).item()

        if heat_balance:
            error_func = self._calc_pwh_error
            error_func_abs = self._calc_pwh_error_abs
            variables = (
                p_fl,
                q_liq,
                wct,
                c_choke,
                freq,
                p_ann,
                hydr_corr_type,
                head_factor,
                heat_balance,
                True,
                c_pump_power,
                c_load_i,
                c_transform_voltage,
                c_cs_power,
                step_length,
            )
        else:
            error_func = self._calc_p_dis_error
            error_func_abs = self._calc_p_dis_error_abs
            variables = (
                p_fl,
                t_fl,
                q_liq,
                wct,
                freq,
                p_ann,
                hydr_corr_type,
                head_factor,
                step_length,
                heat_balance,
                c_choke,
                c_pump_power,
                c_load_i,
                c_transform_voltage,
                c_cs_power,
            )

        p_bot, convergence = self.calc_p_iter(error_func=error_func, error_func_abs=error_func_abs, variables=variables)

        if heat_balance:
            results = self.calc_pwh_pwf(
                p_wf=p_bot,
                q_liq=q_liq,
                wct=wct,
                freq=freq,
                p_ann=p_ann,
                hydr_corr_type=hydr_corr_type,
                head_factor=head_factor,
                heat_balance=heat_balance,
                optimizer=False,
                c_pump_power=c_pump_power,
                c_load_i=c_load_i,
                c_transform_voltage=c_transform_voltage,
                c_cs_power=c_cs_power,
                step_length=step_length,
                output_params=output_params,
                finalize=True,
            )
            del results[1]
            results[0] = p_bot
        else:
            t_in = self.amb_temp_dist.calc_temp(self.tubing.bottom_depth).item()
            results = list(
                self.calc_pwh_pin(
                    p_in=p_bot,
                    t_in=t_in,
                    q_liq=q_liq,
                    wct=wct,
                    freq=freq,
                    p_ann=p_ann,
                    hydr_corr_type=hydr_corr_type,
                    head_factor=head_factor,
                    step_length=step_length,
                    output_params=output_params,
                    heat_balance=heat_balance,
                    optimizer=False,
                    finalize=True,
                    c_pump_power=c_pump_power,
                    c_load_i=c_load_i,
                    c_transform_voltage=c_transform_voltage,
                    c_cs_power=c_cs_power,
                )
            )
            del results[1]

            results[0], *_ = self.casing.calc_pt(
                h_start="top",
                p_mes=p_bot,
                flow_direction=1,
                q_liq=q_liq,
                wct=wct,
                phase_ratio_value=None,
                t_mes=t_in,
                hydr_corr_type=hydr_corr_type,
                step_len=step_length,
                extra_output=output_params,
                heat_balance=heat_balance,
            )
            self._output_objects["casing"] = self.casing

        results[-2] = convergence
        results.append(self.p_dis)

        return results

    def calc_pwf_pfl(
        self,
        p_fl: float,
        q_liq: float,
        wct: float,
        freq: float,
        p_wh: Optional[float] = None,
        p_ann: Optional[float] = None,
        hydr_corr_type: Optional[str] = None,
        head_factor: Optional[float] = None,
        c_choke: Optional[Union[float, dict]] = None,
        c_pump_power: float = 1,
        c_load_i: float = 1,
        c_transform_voltage: float = 1,
        c_cs_power: float = 1,
        step_length: float = 10,
        output_params: bool = False,
        heat_balance: bool = False,
    ) -> list:
        """
        Расчет забойного давления по линейному с учетом всех гидравлических элементов

        Parameters
        ----------
        :param p_fl: линейное давление, Па абс
        :param q_liq: дебит жидкости, ст. м3/с
        :param wct: обводненность, доли ед
        :param freq: частота вращения вала ЭЦН, Гц
        :param p_wh: буферное давление, Па абс
        :param p_ann: затрубное давление, Па изб
        :param hydr_corr_type: тип гидравлической корреляции
        :param head_factor: к-т адаптации на напор ЭЦН
        :param c_choke: адаптационный коэффициент штуцера\
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value},\
            где value - постоянный перепад, который будет использоваться как перепад между буферным и линейным давлением
        :param c_pump_power: адаптационный коэффициент для мощности насоса
        :param c_load_i: адаптационный коэффициент для загрузки по току
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг расчета распределений параметров
        :param heat_balance: опция расчета теплопотерь

        :return: забойное давление, Па абс.
        :return: мощность на станции управления, Вт
        :return: динамический уровень, м
        :return: напряжение на трансформаторе, В
        :return: сила тока ПЭД, А
        :return: загрузка ПЭД, доли ед.
        :return: КПД насоса факт., доли ед.
        :return: КПД системы, доли ед.
        :return: коэффициент сепарации общий, доли ед.
        :return: давление на приеме, Па изб.
        :return: температура ПЭД, К
        :return: мощность ЭЦН, Вт
        :return: мощность ПЭД, Вт
        :return: доля газа на приеме насоса, доли ед.
        :return: флаг сходимости расчета: 0 - сошлось; 1 - нет
        :return: дебит ГЖС в условиях приема насоса, м3/с
        :return: давление на выкиде ЭЦН, Па
        """
        # Установим дебит жидкости и обводненность в флюид и переопределим флюид для всех
        # зависимых от него классов
        self._output_objects = {"params": ["p", "t", "depth"]}

        if output_params:
            self._output_objects["params"] += DISTRS

        self._reinit(wct, q_liq)
        self.casing.fluid = deepcopy(self.fluid)

        if self.choke:
            t_fl = self.amb_temp_dist.calc_temp(self.tubing.top_depth).item()
            self.choke.fluid = deepcopy(self.fluid)

            if p_wh is not None and c_choke is None:
                c_choke = adapt_choke(self.choke, p_fl, p_wh, t_fl, q_liq, wct)

        results = self._calc_pwf(
            p_fl=p_fl,
            q_liq=q_liq,
            wct=wct,
            freq=freq,
            p_ann=p_ann,
            hydr_corr_type=hydr_corr_type,
            head_factor=head_factor,
            step_length=step_length,
            output_params=output_params,
            heat_balance=heat_balance,
            c_choke=c_choke,
            c_pump_power=c_pump_power,
            c_load_i=c_load_i,
            c_transform_voltage=c_transform_voltage,
            c_cs_power=c_cs_power,
        )

        # Сохранение дополнительных атрибутов распределений
        self._extra_output = make_unified_distributions(**self._output_objects)

        # Корректная стыковка распределений из оптимизатора
        if ("choke", "tubing") in self._output_objects.keys():
            self._extra_output["p"][1] = self._extra_output["p"][2]

        self._extra_output["nodes"] = self.__make_nodes()

        return results
