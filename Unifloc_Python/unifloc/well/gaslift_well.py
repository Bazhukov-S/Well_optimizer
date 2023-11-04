from copy import deepcopy
from typing import Optional, Tuple, Union

import unifloc.equipment.gl_system as gl_sys
import unifloc.service._constants as const
import unifloc.service._tools as tls
import unifloc.tools.common_calculations as com
import unifloc.tools.exceptions as exc
import unifloc.well._well as abw


class GasLiftWell(abw.AbstractWell):
    """
    Класс для расчета газлифтной скважины.

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
                    * "p" - давление калибровки, Па абс. - float
                    * "t" - температура калибровки объемного коэффициента нефти, К - float
                * "muob" - словарь с калибровочным значением вязкости нефти при давлении насыщения
                - dict, optional
                    * "value" - калибровочное значение вязкости нефти при давлении насыщения,
                    сПз - float
                    * "p" - давление калибровки, Па абс. - float
                    * "t" - температура калибровки вязкости нефти, К - float
                * "table_model_data" - словарь с исходными данными табличной модели - dict, optional
                    * "pvt_dataframes_dict" - словарь с таблицами с исходными данными
                    - dict of DataFrames
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
    * ambient_temperature_data ("ключ" - определение - тип) - словарь с распределением температуры
     породы по MD - dict
        * обязательные ключи MD, T - list
    * equipment_data ("ключ" - определение - тип) - dict, optional
        * "choke" - словарь с исходными данными для создания объекта штуцера - dict, optional
            * "d" - диаметр штуцера, м - float
        * "gl_system" - словарь с исходными данными для создания системы газлифтных клапанов -
         dict, optional
            * "**valves_data" - произвольное количество словарей с газлифтными клапанами
                * Ключем словаря обязательно должен быть ключ начинающийся с "valve", например
                "valve1", "valve2"
                * Аргументы словаря:
                    * "h_mes" - измеренная глубина установки клапана, м - float
                    * "d" - диаметр клапана, м - float, optional
                    * "p_valve" - давление зарядки клапана, Па абс. - float, optional
                    * "valve_type" - тип клапана - string, optional
                * Если непонятно, см. пример ниже
    """

    def __init__(
        self,
        fluid_data: dict,
        pipe_data: dict,
        well_trajectory_data: dict,
        ambient_temperature_data: dict,
        equipment_data: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        :param fluid_data: словарь с исходными данными для создания флюида
        :param pipe_data: словарь с исходными данными для создания колонн труб
        :param well_trajectory_data: словарь с исходными данными для создания инклинометрии скважины
        :param ambient_temperature_data: словарь с распределением температуры породы по MD
        :param equipment_data: словарь с исходными данными для создания различного оборудования,
         optional

        Examples:
        --------
        >>> import pandas as pd
        >>> import unifloc.tools.units_converter as uc
        >>> from unifloc.well.gaslift_well import GasLiftWell
        >>> # Инициализация исходных данных
        >>> df = pd.DataFrame(columns=["MD", "TVD"], data=[[0, 0], [1400, 1400], [1800, 1800]])
        >>> # Возможный способ задания инклинометрии через dict
        >>> # df = {"MD": [0, 1000],
        >>> #       "TVD": [0, 900]}
        >>>
        >>> # В словари с калибровками подается давление и температура калибровки.
        >>> # Зачастую - это давление насыщения и пластовая температура
        >>> fluid_data = {"q_fluid": uc.convert_rate(100, "m3/day", "m3/s"),
        ...               "pvt_model_data": {"black_oil": {"gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.8,
        ...               "wct": 0.1, "phase_ratio": {"type": "GOR", "value": 300},
        ...               "oil_correlations": {"pb": "Standing", "rs": "Standing","rho": "Standing",
        ...               "b": "Standing", "mu": "Beggs", "compr": "Vasquez"},
        ...               "gas_correlations":
        ...               {"ppc": "Standing","tpc": "Standing", "z": "Dranchuk", "mu": "Lee"},
        ...               "water_correlations": {"b": "McCain", "compr": "Kriel","rho": "Standing",
        ...               "mu": "McCain"},
        ...               "rsb": {"value": 300, "p": 10000000, "t": 303.15},
        ...               "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
        ...               "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
        ...               "table_model_data": None, "use_table_model": False}}}
        >>> # Диаметр можно задавать как числом так и таблицей с распределением по глубине
        >>> d = pd.DataFrame(columns=["MD", "d"], data=[[0, 0.062], [1000, 0.082]])
        >>> # Так тоже возможно: d = {"MD": [0, 1000], "d": [0.06, 0.08]}
        >>> pipe_data = {"casing": {"bottom_depth": 1800, "d": 0.146, "roughness": 0.0001},
        ...              "tubing": {"bottom_depth": 1400, "d": d, "roughness": 0.0001}}
        >>> well_trajectory_data = {"inclinometry": df}
        >>> equipment_data = {"gl_system": {
        ...                   "valve1": {"h_mes": 800, "d": 0.006,
        ...                               "p_valve": uc.convert_pressure(3, "atm", "Pa"),
        ...                              "valve_type": "ЦКсОК"},
        ...                   "valve2": {"h_mes": 850, "d": 0.006,
        ...                               "p_valve": uc.convert_pressure(3, "atm", "Pa"),
        ...                              "valve_type": "ЦКсОК"},
        ...                   "valve3": {"h_mes": 900, "d": 0.006,
        ...                               "p_valve": uc.convert_pressure(3, "atm", "Pa"),
        ...                              "valve_type": "ЦКсОК"},
        ...                   "valve4": {"h_mes": 1000, "d": 0.006,
        ...                               "p_valve": uc.convert_pressure(3, "atm", "Pa"),
        ...                              "valve_type": "ЦКсОК"}
        ...                                               }
        ...                   }
        >>> ambient_temperature_data = {"MD": [0, 1800], "T": [293.15, 303.15]}
        >>>
        >>> # Инициализация объекта скважины
        >>> well = GasLiftWell(fluid_data, pipe_data, well_trajectory_data,
        ...                    ambient_temperature_data, equipment_data)

        >>> # Расчет линейного давления
        >>> parameters = well.calc_pfl_pwf(uc.convert_pressure(20, "MPa", "Pa"),
        ...                                uc.convert_rate(100, "m3/day", "m3/s"), 0.1,
        ...                                q_gas_inj=uc.convert_rate(10000, "m3/day", "m3/s"))

        >>> # Расчет забойного давления
        >>> p_fl = parameters[0]
        >>> # Расчет с сохранением доп. атрибутов распределений свойств
        >>> p_wf = well.calc_pwf_pfl(p_fl, uc.convert_rate(100, "m3/day", "m3/s"), 0.1,
        ...                          q_gas_inj=uc.convert_rate(100000, "m3/day", "m3/s"),
        ...                          output_params=True)
        >>> # Запрос всех значений доп. свойств в виде словаря
        >>> result = well.extra_output
        """
        super().__init__(
            fluid_data,
            pipe_data,
            equipment_data,
            well_trajectory_data,
            ambient_temperature_data,
        )
        if equipment_data and "gl_system" in equipment_data and equipment_data["gl_system"]:
            self.gl_system = gl_sys.GlSystem(equipment_data["gl_system"])
        else:
            # Считаем, что клапанов нет
            self.gl_system = gl_sys.GlSystem({"valve": {"h_mes": self.tubing.bottom_depth, "d": 0.006}})

        self.tubing_upper_point = None
        self.tubing_below_point = None

    def _reinit(self, wct, q_liq):
        """
        Сброс флюида и пересохранение объекта в других объектах

        :param wct: обводненность, д.ед.
        :param q_liq: дебит жидкости, м/c
        :return:
        """
        super()._reinit(wct, q_liq)
        self.tubing_upper_point = None
        self.tubing_below_point = None

    def __make_nodes(self) -> list:
        """
        Создание распределений ключевых узлов

        :return: распределение ключевых узлов
        """
        nodes_cas = [None for _ in self.casing.distributions["depth"]]
        nodes_cas[-1] = "Верхние дыры перфорации"

        if self.tubing_upper_point is None:
            nodes_tub = [None for _ in self.tubing.distributions["depth"]]
            nodes_tub[0] = "Буфер"
            nodes_tub[-1] = "Башмак НКТ"
            nodes_cas = nodes_tub + nodes_cas
        else:
            nodes_tub_upper = [None for _ in self.tubing_upper_point.distributions["depth"]]
            nodes_tub_upper[0] = "Буфер"
            nodes_tub_upper[-1] = "Точка ввода газа"
            nodes_tub_below = [None for _ in self.tubing_below_point.distributions["depth"]]
            nodes_tub_below[-1] = "Башмак НКТ"
            nodes_cas = nodes_tub_upper + nodes_tub_below + nodes_cas

        if self.choke and "depth" in self.choke.distributions:
            nodes_ch = [None for _ in self.choke.distributions["depth"]]
            nodes_ch[0] = "Линия"
            nodes_cas = nodes_ch + nodes_cas

        return nodes_cas

    def calc_pwh_pwf(
        self,
        p_wf: float,
        q_liq: float,
        wct: float,
        hydr_corr_type: Optional[str] = None,
        q_gas_inj: Optional[float] = None,
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        step_length: Optional[float] = None,
        heat_balance: bool = False,
        finalize: bool = False,
        output_params: bool = False,
    ):
        """
        Расчет устьевого давления по забойному с учетом всех гидравлических элементов

        Parameters
        ----------
        :param p_wf: забойное давление, Па абс.
        :param q_liq: дебит жидкости, ст. м3/с
        :param wct: обводненность, доли ед.
        :param hydr_corr_type: тип гидравлической корреляции
        :param q_gas_inj: закачка газлифтного газа, ст. м3/с
        :param friction_factor: к-т адаптации КРД на трение,\
            если не задан берется из атрибутов трубы
        :param grav_holdup_factor: к-т адаптации КРД на гравитацию / holdup,\
            если не задан берется из атрибутов трубы
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг для расчета дополнительных распределений параметров
        :param finalize: рассчитать общие распределения для скважины
        :param heat_balance: опция учета теплопотерь

        :return: линейное давление, Па абс.;
        :return: статус расчёта:
                 0 - расчет успешный;
                 1 - достигнуто минимальное давление, невозможно рассчитать линейное давление;
                 -1 - ошибка интегрирования
        """
        self._reinit(wct, q_liq)

        # Фонтан в случае отсутствия расхода
        if q_gas_inj:
            self.gl_system.q_inj = q_gas_inj
        else:
            self.gl_system.q_inj = 0

        # 1. Расчет давления, температуры на приеме по ЭК вверх
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
            friction_factor=friction_factor,
            grav_holdup_factor=grav_holdup_factor,
            extra_output=output_params,
            heat_balance=heat_balance,
        )
        if finalize:
            self._output_objects["casing"] = self.casing
        if status != 0:
            # Случай неуспешного расчета до приема
            self._extra_output = tls.make_unified_distributions(**self._output_objects)
            return p_in, t_in, status

        # 2. Расчет естественной сепарации и модификация свойств флюида после сепарации
        self.fluid.calc_flow(p_in, t_in)

        # Разобьем НКТ на участки до и после клапана в зависимости от его расположения
        if self.gl_system.h_mes_work > self.tubing.bottom_depth:
            raise exc.GlValveError(
                f"Глубина рабочего клапана = {self.gl_system.h_mes_work} м больше "
                f"глубины НКТ = {self.tubing.bottom_depth} м.",
                self.gl_system.h_mes_work,
            )
        if self.gl_system.h_mes_work == self.tubing.bottom_depth:
            fluid_upper_point = deepcopy(self.fluid)
            fluid_upper_point.q_gas_free = self.gl_system.q_inj
            fluid_upper_point.reinit_q_gas_free(self.gl_system.q_inj)
            self.tubing.fluid = fluid_upper_point

            # 3. Расчет распределения давления по НКТ вверх
            p_wh, t_wh, status = self.tubing.calc_pt(
                h_start="bottom",
                p_mes=p_in,
                flow_direction=-1,
                q_liq=q_liq,
                wct=wct,
                phase_ratio_value=None,
                t_mes=t_in,
                hydr_corr_type=hydr_corr_type,
                step_len=step_length,
                friction_factor=friction_factor,
                grav_holdup_factor=grav_holdup_factor,
                extra_output=output_params,
                heat_balance=heat_balance,
            )
            if finalize:
                self._output_objects["tubings"] = [self.tubing]
            if status != 0:
                # Случай неуспешного расчета до буфера
                self._extra_output = tls.make_unified_distributions(**self._output_objects)
                return p_wh, t_wh, status
        else:
            self.tubing_below_point = deepcopy(self.tubing)
            self.tubing_below_point.top_depth = self.gl_system.h_mes_work

            self.tubing_upper_point = deepcopy(self.tubing)
            self.fluid.q_gas_free = self.gl_system.q_inj
            self.fluid.reinit_q_gas_free(self.gl_system.q_inj)
            self.tubing_upper_point.fluid = deepcopy(self.fluid)
            self.tubing_upper_point.bottom_depth = self.gl_system.h_mes_work

            # 3.1 Расчет распределения давления по НКТ ниже клапана
            p_inj, t_inj, status = self.tubing_below_point.calc_pt(
                h_start="bottom",
                p_mes=p_in,
                flow_direction=-1,
                q_liq=q_liq,
                wct=wct,
                phase_ratio_value=None,
                t_mes=t_in,
                hydr_corr_type=hydr_corr_type,
                step_len=step_length,
                friction_factor=friction_factor,
                grav_holdup_factor=grav_holdup_factor,
                extra_output=output_params,
                heat_balance=heat_balance,
            )
            if finalize:
                self._output_objects["tubings"] = [self.tubing_below_point]
            if status != 0:
                # Случай неуспешного расчета до клапана
                self._extra_output = tls.make_unified_distributions(**self._output_objects)
                return p_inj, t_inj, status

            # 3.2 Расчет распределения давления по НКТ выше клапана
            p_wh, t_wh, status = self.tubing_upper_point.calc_pt(
                h_start="bottom",
                p_mes=p_inj,
                flow_direction=-1,
                q_liq=q_liq,
                wct=wct,
                phase_ratio_value=None,
                t_mes=t_inj,
                hydr_corr_type=hydr_corr_type,
                step_len=step_length,
                friction_factor=friction_factor,
                extra_output=output_params,
                heat_balance=heat_balance,
            )
            if finalize:
                self._output_objects["tubings"].insert(0, self.tubing_upper_point)
            if status != 0:
                # Случай неуспешного расчета до буфера
                self._extra_output = tls.make_unified_distributions(**self._output_objects)

        return p_wh, t_wh, status

    def calc_pfl_pwf(
        self,
        p_wf: float,
        q_liq: float,
        wct: float,
        hydr_corr_type: Optional[str] = None,
        q_gas_inj: Optional[float] = None,
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        step_length: Optional[float] = None,
        output_params: bool = False,
        heat_balance: bool = False,
        c_choke: Optional[Union[float, dict]] = None,
    ) -> Tuple[float, float]:
        """
        Расчет линейного давления по забойному с учетом всех гидравлических элементов

        Parameters
        ----------
        :param p_wf: забойное давление, Па абс.
        :param q_liq: дебит жидкости, ст. м3/с
        :param wct: обводненность, доли ед.
        :param hydr_corr_type: тип гидравлической корреляции
        :param q_gas_inj: закачка газлифтного газа, ст. м3/с
        :param friction_factor: к-т адаптации КРД на трение,\
            если не задан берется из атрибутов трубы
        :param grav_holdup_factor: к-т адаптации КРД на гравитацию / holdup,\
            если не задан берется из атрибутов трубы
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг для расчета дополнительных распределений параметров
        :param heat_balance: опция учета теплопотерь
        :param c_choke: адаптационный коэффициент штуцера\
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value},\
            где value - постоянный перепад, который будет использоваться как перепад между буферным и линейным давлением

        :return: линейное давление, Па абс.;
        :return: статус расчёта:
                 0 - расчет успешный;
                 1 - достигнуто минимальное давление, невозможно рассчитать линейное давление;
                 -1 - ошибка интегрирования
        """

        self._output_objects = {"params": ["p", "t", "depth"]}

        if output_params:
            self._output_objects["params"] += const.DISTRS

        p_wh, t_wh, status = self.calc_pwh_pwf(
            p_wf=p_wf,
            q_liq=q_liq,
            wct=wct,
            hydr_corr_type=hydr_corr_type,
            q_gas_inj=q_gas_inj,
            friction_factor=friction_factor,
            grav_holdup_factor=grav_holdup_factor,
            step_length=step_length,
            output_params=output_params,
            finalize=True,
            heat_balance=heat_balance,
        )

        p_fl, t_fl, node_status = self.calc_p_choke(
            choke=self.choke,
            p_received=p_wh,
            t_received=t_wh,
            q_liq=q_liq,
            wct=wct,
            flow_direction=-1,
            output_params=output_params,
            c_choke=c_choke,
        )
        if node_status:
            self._output_objects["choke"] = self.choke

        self._extra_output = tls.make_unified_distributions(**self._output_objects)

        return p_fl, status

    def _calc_pwf_hb(
        self,
        p_fl: float,
        q_liq: float,
        wct: float,
        hydr_corr_type: str,
        q_gas_inj: float,
        friction_factor: float,
        grav_holdup_factor: float,
        c_choke: Union[float, dict],
        step_length: float,
        heat_balance: bool,
        output_params: bool,
    ):
        """
        Функция расчета давления на забое итеративно по давлению в линии
        """
        p_wf, convergence = self.calc_p_iter(
            error_func=self._calc_pwh_error,
            error_func_abs=self._calc_pwh_error_abs,
            variables=(
                p_fl,
                q_liq,
                wct,
                c_choke,
                hydr_corr_type,
                q_gas_inj,
                friction_factor,
                grav_holdup_factor,
                step_length,
                heat_balance,
            ),
        )

        # Сохранение параметров распределений
        self.calc_pwh_pwf(
            p_wf=p_wf,
            q_liq=q_liq,
            wct=wct,
            hydr_corr_type=hydr_corr_type,
            q_gas_inj=q_gas_inj,
            friction_factor=friction_factor,
            grav_holdup_factor=grav_holdup_factor,
            step_length=step_length,
            output_params=output_params,
            finalize=True,
            heat_balance=heat_balance,
        )

        return p_wf, convergence

    def calc_pwf_pfl(
        self,
        p_fl: float,
        q_liq: float,
        wct: float,
        p_wh: Optional[float] = None,
        hydr_corr_type: Optional[str] = None,
        q_gas_inj: Optional[float] = None,
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        c_choke: Optional[Union[float, dict]] = None,
        step_length: Optional[float] = None,
        output_params: bool = False,
        heat_balance: bool = False,
    ) -> float:
        """
        Расчет забойного давления по линейному с учетом всех гидравлических элементов

        Parameters
        ----------
        :param p_fl: линейное давление, Па абс.
        :param q_liq: дебит жидкости, ст. м3/с
        :param wct: обводненность, доли ед.
        :param p_wh: буферное давление, Па абс.
        :param hydr_corr_type: тип гидравлической корреляции
        :param q_gas_inj: закачка газлифтного газа, ст. м3/с
        :param friction_factor: к-т адаптации КРД на трение, если не задан берется из атрибутов
            трубы
        :param grav_holdup_factor: к-т адаптации КРД на гравитацию/holdup, если не задан берется
            из атрибутов трубы
        :param c_choke: адаптационный коэффициент штуцера \
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value}, \
            где value - постоянный перепад, который будет использоваться как перепад между буферным и
            линейным давлением
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг для расчета дополнительных распределений параметров
        :param heat_balance: опция учета теплопотерь

        :return: забойное давление, Па абс.
        """

        t_fl = self.amb_temp_dist.calc_temp(self.tubing.top_depth).item()

        self._output_objects = {"params": ["p", "t", "depth"]}

        if output_params:
            self._output_objects["params"] += const.DISTRS

        # Сброс флюида и пересохранение объекта в других объектах
        self._reinit(wct, q_liq)
        self.tubing.fluid = deepcopy(self.fluid)
        self.casing.fluid = deepcopy(self.fluid)

        # Фонтан в случае отсутствия расхода
        if q_gas_inj:
            self.gl_system.q_inj = q_gas_inj
        else:
            self.gl_system.q_inj = 0

        # 1. Расчет в штуцере, если он есть
        if self.choke:
            self.choke.fluid = deepcopy(self.fluid)

            # 1.1 Адаптация штуцера в случае наличия буферного давления и отсутствия коэффициента
            # штуцера
            if p_wh is not None and q_gas_inj is not None and c_choke is None:
                self.choke.fluid.q_gas_free = q_gas_inj
                self.choke.fluid.reinit_q_gas_free(self.gl_system.q_inj)
                c_choke = com.adapt_choke(
                    choke=self.choke,
                    p_out=p_fl,
                    p_in=p_wh,
                    t_out=t_fl,
                    q_liq=q_liq,
                    wct=wct,
                )

        if heat_balance:
            p_wf, *_ = self._calc_pwf_hb(
                p_fl=p_fl,
                q_liq=q_liq,
                wct=wct,
                hydr_corr_type=hydr_corr_type,
                q_gas_inj=q_gas_inj,
                friction_factor=friction_factor,
                grav_holdup_factor=grav_holdup_factor,
                c_choke=c_choke,
                step_length=step_length,
                heat_balance=heat_balance,
                output_params=output_params,
            )
        else:
            p_wh, t_wh, node_status = self.calc_p_choke(
                choke=self.choke,
                p_received=p_fl,
                t_received=t_fl,
                q_liq=q_liq,
                wct=wct,
                flow_direction=1,
                output_params=output_params,
                c_choke=c_choke,
            )
            if node_status:
                self._output_objects["choke"] = self.choke

            if self.gl_system.h_mes_work > self.tubing.bottom_depth:
                raise exc.GlValveError(
                    f"Глубина рабочего клапана = {self.gl_system.h_mes_work} м больше "
                    f"глубины НКТ = {self.tubing.bottom_depth} м.",
                    self.gl_system.h_mes_work,
                )
            if self.gl_system.h_mes_work == self.tubing.bottom_depth:
                fluid_upper_point = deepcopy(self.fluid)
                fluid_upper_point.q_gas_free = self.gl_system.q_inj
                fluid_upper_point.reinit_q_gas_free(self.gl_system.q_inj)
                self.tubing.fluid = fluid_upper_point

                # 2. Расчет распределения давления по НКТ вниз
                self._output_objects["tubings"] = [self.tubing]
                p_in, t_in, _ = self.tubing.calc_pt(
                    h_start="top",
                    p_mes=p_wh,
                    flow_direction=1,
                    q_liq=q_liq,
                    wct=wct,
                    phase_ratio_value=None,
                    t_mes=t_wh,
                    hydr_corr_type=hydr_corr_type,
                    step_len=step_length,
                    friction_factor=friction_factor,
                    extra_output=output_params,
                    grav_holdup_factor=grav_holdup_factor,
                    heat_balance=heat_balance,
                )
            else:
                self.tubing_upper_point = deepcopy(self.tubing)
                fluid_upper_point = deepcopy(self.fluid)
                fluid_upper_point.q_gas_free = self.gl_system.q_inj
                fluid_upper_point.reinit_q_gas_free(self.gl_system.q_inj)
                self.tubing_upper_point.fluid = fluid_upper_point
                self.tubing_upper_point.bottom_depth = self.gl_system.h_mes_work
                self.tubing_below_point = deepcopy(self.tubing)
                self.tubing_below_point.fluid = deepcopy(self.fluid)
                self.tubing_below_point.top_depth = self.gl_system.h_mes_work

                # 2.1 Расчет распределения давления по НКТ выше клапана
                self._output_objects["tubings"] = [self.tubing_upper_point]
                p_inj, t_inj, _ = self.tubing_upper_point.calc_pt(
                    h_start="top",
                    p_mes=p_wh,
                    flow_direction=1,
                    q_liq=q_liq,
                    wct=wct,
                    phase_ratio_value=None,
                    t_mes=t_wh,
                    hydr_corr_type=hydr_corr_type,
                    step_len=step_length,
                    friction_factor=friction_factor,
                    extra_output=output_params,
                    heat_balance=heat_balance,
                    grav_holdup_factor=grav_holdup_factor,
                )

                # 2.2 Расчет распределения давления по НКТ ниже клапана
                self._output_objects["tubings"].append(self.tubing_below_point)
                p_in, t_in, _ = self.tubing_below_point.calc_pt(
                    h_start="top",
                    p_mes=p_inj,
                    flow_direction=1,
                    q_liq=q_liq,
                    wct=wct,
                    phase_ratio_value=None,
                    t_mes=t_inj,
                    hydr_corr_type=hydr_corr_type,
                    step_len=step_length,
                    friction_factor=friction_factor,
                    extra_output=output_params,
                    heat_balance=heat_balance,
                    grav_holdup_factor=grav_holdup_factor,
                )

            # 3. Расчет давления на забое по ЭК вниз
            self._output_objects["casing"] = self.casing
            p_wf, t_wf, _ = self.casing.calc_pt(
                h_start="top",
                p_mes=p_in,
                flow_direction=1,
                q_liq=q_liq,
                wct=wct,
                phase_ratio_value=None,
                t_mes=t_in,
                hydr_corr_type=hydr_corr_type,
                step_len=step_length,
                friction_factor=friction_factor,
                extra_output=output_params,
                heat_balance=heat_balance,
                grav_holdup_factor=grav_holdup_factor,
            )

        self._extra_output = tls.make_unified_distributions(**self._output_objects)
        self._extra_output["nodes"] = self.__make_nodes()
        return p_wf
