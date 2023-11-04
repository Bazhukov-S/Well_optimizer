from copy import deepcopy
from typing import Optional, Union

import scipy.optimize as opt

import unifloc.service._constants as const
import unifloc.service._tools as tls
from unifloc.equipment.choke import Choke
from unifloc.equipment.gl_system import GlSystem
from unifloc.pipe.annulus import Annulus
from unifloc.tools.common_calculations import adapt_choke
from unifloc.tools.exceptions import GlValveError

from ._well import AbstractWell


class GasLiftWellOneValve(AbstractWell):
    """
    Класс для расчета газлифтной скважины с одним рабочим клапаном.

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
        * "ann_choke" - словарь с исходными данными для создания объекта штуцера на газовой линии - dict, optional
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
                    * "s_bellow" - площадь поперечного сечения сильфона клапана, м2 - float, optional
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
        >>> from unifloc.well.gaslift_well_one_valve import GasLiftWellOneValve
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
        >>> pipe_data = {"casing": {"bottom_depth": 1800, "d": 0.146, "roughness": 0.0001, "s_wall": 0.005},
        ...              "tubing": {"bottom_depth": 1400, "d": d, "roughness": 0.0001, "s_wall": 0.005}}
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
        >>> well = GasLiftWellOneValve(fluid_data, pipe_data, well_trajectory_data,
        ...                    ambient_temperature_data, equipment_data)
        >>> # Расчет линейного давления
        >>> parameters = well.calc_pwh_pwf(p_wf=uc.convert_pressure(20, "MPa", "Pa"),
        ...                                q_liq=uc.convert_rate(100, "m3/day", "m3/s"), wct=0.1,
        ...                                p_gas_inj=uc.convert_pressure(15, "MPa", "Pa"))
        >>> # Расчет забойного давления
        >>> p_fl = parameters[0]
        >>> q_inj = well.gl_system.q_inj
        >>> # Расчет с сохранением доп. атрибутов распределений свойств
        >>> p_wf = well.calc_pwf_pfl(p_fl=p_fl, q_liq=uc.convert_rate(100, "m3/day", "m3/s"), wct=0.1,
        ...                          q_gas_inj=q_inj,
        ...                          output_params=True)
        >>> # Запрос всех значений доп. свойств в виде словаря
        >>> result = well.extra_output
        >>> result_ann = well.extra_output_annulus
        """
        super().__init__(
            fluid_data,
            pipe_data,
            equipment_data,
            well_trajectory_data,
            ambient_temperature_data,
        )
        if equipment_data is not None and "gl_system" in equipment_data and equipment_data["gl_system"] is not None:
            self.gl_system = GlSystem(equipment_data["gl_system"])
        else:
            # Считаем, что клапанов нет
            self.gl_system = GlSystem({"valve": {"h_mes": self.tubing.bottom_depth, "d": 0.006}})

        # Флюид для расчета затрубного пространства
        self.annulus_fluid = deepcopy(self.fluid)
        self.annulus_fluid.fluid_type = "gas"
        self.annulus_fluid.reinit_fluid_type("gas")

        # Инициализация класса для расчета затрубного пространства
        if "casing" in pipe_data and "tubing" in pipe_data:
            self.annulus = Annulus(
                fluid=self.annulus_fluid,
                bottom_depth=self.gl_system.h_mes_work,
                ambient_temperature_distribution=self.amb_temp_dist,
                d_casing=pipe_data["casing"]["d"],
                d_tubing=pipe_data["tubing"]["d"],
                s_wall=self.tubing.s_wall,
                roughness=pipe_data["casing"]["roughness"],
                trajectory=self.well_trajectory,
            )
        else:
            self.annulus = None

        # Инициализация класса для расчета штуцера на газовой линии
        if equipment_data is not None and "ann_choke" in equipment_data:
            equipment_data["ann_choke"].update({"h_mes": 0, "fluid": self.annulus_fluid, "d_up": self.casing.d0})
            self.ann_choke = Choke(**equipment_data["ann_choke"])
        else:
            self.ann_choke = None

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

        :return:
        """
        nodes_cas = [None for _ in self.casing.distributions["depth"]]
        nodes_cas[-1] = "Верхние дыры перфорации"

        if self.tubing_below_point is None and self.tubing_upper_point.distributions:
            nodes_tub = [None for _ in self.tubing_upper_point.distributions["depth"]]
            nodes_tub[0] = "Буфер"
            nodes_tub[-1] = "Башмак НКТ"
            nodes_cas = nodes_tub + nodes_cas
        else:
            if self.tubing_upper_point.distributions:
                nodes_tub_upper = [None for _ in self.tubing_upper_point.distributions["depth"]]
                nodes_tub_upper[0] = "Буфер"
                nodes_tub_upper[-1] = "Рабочий газлифтный клапан"
            else:
                nodes_tub_upper = []

            if self.tubing_below_point.distributions:
                nodes_tub_below = [None for _ in self.tubing_below_point.distributions["depth"]]
                nodes_tub_below[-1] = "Башмак НКТ"
            else:
                nodes_tub_below = []

            nodes_cas = nodes_tub_upper + nodes_tub_below + nodes_cas

        if self.choke and "depth" in self.choke.distributions:
            nodes_ch = [None for _ in self.choke.distributions["depth"]]
            nodes_ch[0] = "Линия"
            nodes_cas = nodes_ch + nodes_cas

        return nodes_cas

    def __make_nodes_annulus(self) -> list:
        """
        Создание распределений ключевых узлов

        :return:
        """
        nodes_ann = [None for _ in self.annulus.distributions["depth"]]
        nodes_ann[-1] = "Рабочий газлифтный клапан"
        nodes_ann[0] = "Затруб"

        if self.ann_choke and "depth" in self.ann_choke.distributions:
            nodes_ch = [None for _ in self.ann_choke.distributions["depth"]]
            nodes_ch[0] = "Газовая линия"
            nodes_ann = nodes_ch + nodes_ann

        return nodes_ann

    def calc_pwf_pfl(
        self,
        p_fl: float,
        q_liq: float,
        wct: float,
        p_wh: Optional[float] = None,
        p_ann: Optional[float] = None,
        hydr_corr_type: Optional[str] = None,
        q_gas_inj: Optional[float] = None,
        p_gas_inj: Optional[float] = None,
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        c_choke: Optional[Union[float, dict]] = None,
        c_ann_choke: Optional[Union[float, dict]] = None,
        c_gl_valve: Optional[float] = None,
        step_length: float = 10,
        output_params: Optional[bool] = False,
        heat_balance: Optional[bool] = False,
        recalc_pb: Optional[bool] = False,
    ) -> list:
        """
        Расчет забойного давления по линейному с учетом всех гидравлических элементов

        Parameters
        ----------
        :param p_fl: линейное давление, Па абс.
        :param q_liq: дебит жидкости, ст. м3/с
        :param wct: обводненность, доли ед.
        :param p_wh: буферное давление, Па абс.
        :param p_ann: затрубное давление, Па абс.
        :param hydr_corr_type: тип гидравлической корреляции
        :param q_gas_inj: закачка газлифтного газа, ст. м3/с
        :param p_gas_inj: давление закачки газлифтного газа, Па
        :param friction_factor: к-т адаптации КРД на трение, если не задан берется
            из атрибутов трубы
        :param grav_holdup_factor: к-т адаптации КРД на гравитацию/holdup, если не задан берется
            из атрибутов трубы
        :param c_choke: адаптационный коэффициент штуцера \
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value}, \
            где value - постоянный перепад, который будет использоваться как перепад между буферным и
            линейным давлением
        :param c_ann_choke: адаптационный коэффициент штуцера на газовой линии \
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value}, \
            где value - постоянный перепад, который будет использоваться как перепад между затрубным
            и давлением закачки гл.г.
        :param c_gl_valve: к-т адаптации газлифтного клапана
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг для расчета дополнительных распределений параметров
        :param heat_balance: опция учета теплопотерь
        :param recalc_pb: опция пересчета давления насыщения выше точки ввода газа

        :return: 0 - давление закачки газлифтного газа, Па
        :return: 1 - температура закачки газлифтного газа, К
        :return: 2 - забойное давление, Па
        :return: 3 - давление на выходе из клапана (давление в НКТ на глубине спуска клапана), Па
        :return: 4 - температура на выходе из клапана (температура в НКТ на глубине спуска клапана), К
        :return: 5 - статус расчёта:
                     0 - расчет успешный;
                     1 - достигнуто минимальное давление, невозможно рассчитать линейное давление;
                     -1 - ошибка интегрирования
        :return: 6 - давление на входе в клапан (давление в затрубе на глубине спуска клапана), Па
        :return: 7 - температура на входе в клапана (температура в затрубе на глубине спуска клапана), К
        :return: 8 - закачка газлифтного газа, ст. м3/с
        :return: 9 - затрубное давление, Па
        :return: 10 - затрубная температура, К
        :return: 11 - расчетное буферное давление, Па
        """
        if self.choke:
            self.choke.fluid = deepcopy(self.fluid)
            # 1.1 Адаптация штуцера в случае наличия буферного давления и отсутствия коэффициента
            # штуцера
            if p_wh is not None and q_gas_inj is not None and c_choke is None:
                self.choke.fluid.q_gas_free = q_gas_inj
                self.choke.fluid.reinit_q_gas_free(q_gas_inj)
                t_fl = self.amb_temp_dist.calc_temp(self.tubing.top_depth).item()
                c_choke = adapt_choke(self.choke, p_fl, p_wh, t_fl, q_liq, wct)

        if self.ann_choke:
            self.ann_choke.fluid = deepcopy(self.annulus_fluid)
            # 1.2 Адаптация штуцера на газовой линии в случае наличия затрубного давления,
            # давления закачки гл/г, расхода гл/г и отсутствия коэффициента штуцера
            if p_ann is not None and p_gas_inj is not None and q_gas_inj is not None and c_ann_choke is None:
                self.ann_choke.fluid.q_fluid = q_gas_inj
                self.ann_choke.fluid.reinit_q_fluid(q_gas_inj)
                t_ann = self.amb_temp_dist.calc_temp(self.casing.top_depth).item()
                c_ann_choke = adapt_choke(self.ann_choke, p_ann, p_gas_inj, t_ann, q_gas_inj, 0)

        # 2. Итерационный расчет Рзаб путем минимизации ошибки Рбуф
        results = self.__calc_pwf(
            p_fl=p_fl,
            q_liq=q_liq,
            wct=wct,
            hydr_corr_type=hydr_corr_type,
            q_gas_inj=q_gas_inj,
            p_gas_inj=p_gas_inj,
            friction_factor=friction_factor,
            grav_holdup_factor=grav_holdup_factor,
            c_choke=c_choke,
            c_ann_choke=c_ann_choke,
            c_gl_valve=c_gl_valve,
            step_length=step_length,
            output_params=output_params,
            heat_balance=heat_balance,
            recalc_pb=recalc_pb,
        )

        # 3. Расчет давления на входе в клапан (в ЭК на глубине спуска клапана)
        if results[5] is not None:
            p_cas_valve, t_cas_valve = results[5], results[6]
        elif results[3] is not None and results[4] is not None:
            t_cas_valve = self.amb_temp_dist.calc_temp(self.annulus.bottom_depth).item()
            p_cas_valve = self.gl_system.valve_working.calc_pt(
                p_mes=results[3],
                t_mes=results[4],
                flow_direction=1,
                q_gas=self.gl_system.q_inj,
                gamma_gas=self.fluid.gamma_gas,
                cd=c_gl_valve,
            )
        else:
            p_cas_valve, t_cas_valve = None, None

        # 4. Расчет затрубного давления
        self.annulus.fluid = deepcopy(self.annulus_fluid)
        self.annulus.fluid.q_fluid = self.gl_system.q_inj
        self.annulus.fluid.reinit_q_fluid(self.gl_system.q_inj)
        self.annulus.bottom_depth = self.gl_system.h_mes_work

        if results[7] is not None:
            p_ann, t_ann = results[7], results[8]
        elif p_cas_valve is not None and t_cas_valve is not None:
            p_ann, t_ann, status = self.annulus.calc_pt(
                h_start="bottom",
                p_mes=p_cas_valve,
                flow_direction=1,
                q_liq=self.gl_system.q_inj,
                wct=0,
                t_mes=t_cas_valve,
                hydr_corr_type="gray",
                step_len=step_length,
                friction_factor=friction_factor,
                extra_output=output_params,
                heat_balance=False,
                grav_holdup_factor=grav_holdup_factor,
            )
        else:
            p_ann, t_ann, status = None, None, 1

            self._ann_output_objects["annulus"] = self.annulus
            if status != 0:
                # Случай неуспешного расчета затрубного давления
                self._ann_extra_output = tls.make_unified_distributions(**self._ann_output_objects, flag_ann=True)
                return [
                    p_ann,
                    t_ann,
                    results[2],
                    results[3],
                    results[4],
                    status,
                    p_cas_valve,
                    t_cas_valve,
                    self.gl_system.q_inj,
                    p_ann,
                    t_ann,
                    results[0],
                ]

        # 5. Расчет давления закачки газа
        if results[9] is not None:
            p_gas_inj_, t_gas_inj_ = results[9], results[10]
        else:
            p_gas_inj_, t_gas_inj_, *_ = self.calc_p_choke(
                choke=self.ann_choke,
                p_received=p_ann,
                t_received=t_ann,
                flow_direction=1,
                q_liq=self.gl_system.q_inj,
                wct=0,
                output_params=output_params,
                c_choke=c_ann_choke,
            )

        self._extra_output = tls.make_unified_distributions(**self._output_objects)
        self._extra_output["nodes"] = self.__make_nodes()

        self._extra_output_annulus = tls.make_unified_distributions(**self._ann_output_objects, flag_ann=True)
        self._extra_output_annulus["nodes"] = self.__make_nodes_annulus()

        return [
            p_gas_inj_,
            t_gas_inj_,
            results[2],
            results[3],
            results[4],
            results[11],
            p_cas_valve,
            t_cas_valve,
            self.gl_system.q_inj,
            p_ann,
            t_ann,
            results[0],
        ]

    def calc_pwh_pwf(
        self,
        p_wf: float,
        q_liq: float,
        wct: float,
        hydr_corr_type: Optional[str] = None,
        q_gas_inj: Optional[float] = None,
        p_gas_inj: Optional[float] = None,
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        c_ann_choke: Optional[Union[float, dict]] = None,
        c_gl_valve: Optional[float] = None,
        step_length: float = 10,
        heat_balance: Optional[bool] = False,
        output_params: Optional[bool] = False,
        recalc_pb: Optional[bool] = False,
        finalize: Optional[bool] = False,
    ) -> tuple:
        """
        Расчет буферного давления по забойному с учетом всех гидравлических элементов

        Parameters
        ----------
        :param p_wf: забойное давление, Па абс.
        :param q_liq: дебит жидкости, ст. м3/с
        :param wct: обводненность, доли ед.
        :param hydr_corr_type: тип гидравлической корреляции
        :param q_gas_inj: закачка газлифтного газа, ст. м3/с
        :param p_gas_inj: давление закачки газлифтного газа, Па
        :param friction_factor: к-т адаптации КРД на трение,
            если не задан берется из атрибутов трубы
        :param grav_holdup_factor: к-т адаптации КРД на гравитацию / holdup,
            если не задан берется из атрибутов трубы
        :param c_ann_choke: адаптационный коэффициент штуцера на газовой линии \
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value}, \
            где value - постоянный перепад, который будет использоваться как перепад между затрубным
            и давлением закачки гл.г.
        :param c_gl_valve: к-т адаптации газлифтного клапана
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг для расчета дополнительных распределений параметров
        :param heat_balance: опция учета теплопотерь
        :param recalc_pb: опция пересчета давления насыщения выше точки ввода газа
        :param finalize: рассчитать общие распределения для скважины

        :return: 0 - буферное давление, Па
        :return: 1 - температура на буфере, К
        :return: 2 - забойное давление, Па
        :return: 3 - давление на выходе из клапана (давление в НКТ на глубине спуска клапана), Па
        :return: 4 - температура на выходе из клапана (температура в НКТ на глубине спуска клапана), К
        :return: 5 - давление на входе в клапан (давление в затрубе на глубине спуска клапана), Па
        :return: 6 - температура на входе в клапана (температура в затрубе на глубине спуска клапана), К
        :return: 7 - затрубное давление, Па
        :return: 8 - затрубная температура, К
        :return: 9 - давление закачки газлифтного газа, Па
        :return: 10 - температура закачки газлифтного газа, К
        :return: 11 - статус расчёта:
                     0 - расчет успешный;
                     1 - достигнуто минимальное давление, невозможно рассчитать линейное давление;
                     -1 - ошибка интегрирования

        """
        if not self._output_objects:
            self._output_objects = {"params": ["p", "t", "depth"]}

        if not self._ann_output_objects:
            self._ann_output_objects = {"params": ["p", "t", "depth"]}

        if output_params:
            self._output_objects["params"] += const.DISTRS
            self._ann_output_objects["params"] += const.DISTRS

        # Сброс флюида и пересохранение объекта в других объектах
        self._reinit(wct, q_liq)
        self.tubing.fluid = deepcopy(self.fluid)
        self.casing.fluid = deepcopy(self.fluid)

        # Фонтан в случае отсутствия расхода
        if q_gas_inj is not None:
            self.gl_system.q_inj = q_gas_inj
        else:
            self.gl_system.q_inj = 0

        # 1.  Расчет распределения давления по ЭК от забоя до глубины спуска НКТ
        t_wf = self.amb_temp_dist.calc_temp(self.casing.bottom_depth).item()
        p_in, t_in, status = self.casing.calc_pt(
            h_start="bottom",
            p_mes=p_wf,
            flow_direction=-1,
            q_liq=q_liq,
            wct=wct,
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
            return (
                p_in,
                t_in,
                p_wf,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                status,
            )

        # 2. Расчет естественной сепарации и модификация свойств флюида после сепарации
        self.fluid.calc_flow(p_in, t_in)

        if self.gl_system.h_mes_work > self.tubing.bottom_depth:
            raise GlValveError(
                f"Глубина рабочего клапана = {self.gl_system.h_mes_work} м больше "
                f"глубины НКТ = {self.tubing.bottom_depth} м.",
                self.gl_system.h_mes_work,
            )

        self.tubing_upper_point = deepcopy(self.tubing)

        if self.gl_system.h_mes_work == self.tubing.bottom_depth:
            self._output_objects["tubings"] = []
            p_tub_valve, t_tub_valve = p_in, t_in
        else:
            self.tubing_below_point = deepcopy(self.tubing)
            self.tubing_below_point.top_depth = self.gl_system.h_mes_work
            self.tubing_upper_point.bottom_depth = self.gl_system.h_mes_work

            # 3. Расчет распределения давления по НКТ ниже клапана
            p_tub_valve, t_tub_valve, status = self.tubing_below_point.calc_pt(
                h_start="bottom",
                p_mes=p_in,
                flow_direction=-1,
                q_liq=q_liq,
                wct=wct,
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
                return (
                    p_tub_valve,
                    t_tub_valve,
                    p_wf,
                    p_tub_valve,
                    t_tub_valve,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    status,
                )

        # 3.1. Определение расхода г/г при известном давлении закачки
        if q_gas_inj is None and p_gas_inj is not None:
            q_gas_inj_result = self.__calc_q_gas_inj(
                p_gas_inj,
                p_tub_valve,
                t_tub_valve,
                friction_factor,
                grav_holdup_factor,
                c_ann_choke,
                c_gl_valve,
                step_length,
                output_params,
            )
            self.gl_system.q_inj = q_gas_inj_result[2]
            p_cas_valve = q_gas_inj_result[3]
            t_cas_valve = q_gas_inj_result[4]
            p_ann = q_gas_inj_result[5]
            t_ann = q_gas_inj_result[6]
            p_inj = q_gas_inj_result[0]
            t_inj = q_gas_inj_result[1]
        else:
            p_cas_valve = None
            t_cas_valve = None
            p_ann = None
            t_ann = None
            p_inj = None
            t_inj = None

        if recalc_pb:
            # Расчет нового газового фактора
            q_oil_st = (1 - self.fluid.wct) * self.fluid.q_fluid

            if self.fluid.phase_ratio["type"].lower() == "gor":
                q_gas_st = self.fluid.rp * q_oil_st
                rp_new = (q_gas_st + self.gl_system.q_inj) / q_oil_st
                self.tubing_upper_point.fluid.reinit_phase_ratio({"type": "GOR", "value": rp_new})
            else:
                q_gas_st = self.fluid.rp / (1 - self.fluid.wct) * q_oil_st
                rp_new = (q_gas_st + self.gl_system.q_inj) / q_oil_st
                self.tubing_upper_point.fluid.reinit_phase_ratio({
                    "type": "GLR",
                    "value": rp_new * (1 - self.fluid.wct),
                })
        else:
            # Расчет с добавлением дебита свободного газа
            self.tubing_upper_point.fluid.q_gas_free = self.gl_system.q_inj
            self.tubing_upper_point.fluid.reinit_q_gas_free(self.gl_system.q_inj)

        if self.choke is not None:
            self.choke.fluid = self.tubing_upper_point.fluid

        # 4. Расчет распределения давления по НКТ выше клапана
        p_wh, t_wh, status = self.tubing_upper_point.calc_pt(
            h_start="bottom",
            p_mes=p_tub_valve,
            flow_direction=-1,
            q_liq=q_liq,
            wct=wct,
            t_mes=t_tub_valve,
            hydr_corr_type=hydr_corr_type,
            step_len=step_length,
            friction_factor=friction_factor,
            grav_holdup_factor=grav_holdup_factor,
            extra_output=output_params,
            heat_balance=heat_balance,
        )
        if finalize:
            self._output_objects["tubings"].insert(0, self.tubing_upper_point)
        if status != 0:
            # Случай неуспешного расчета до буфера
            self._extra_output = tls.make_unified_distributions(**self._output_objects)
            return (
                p_wh,
                t_wh,
                p_wf,
                p_tub_valve,
                t_tub_valve,
                p_cas_valve,
                t_cas_valve,
                p_ann,
                t_ann,
                p_inj,
                t_inj,
                status,
            )
        self._extra_output_annulus = tls.make_unified_distributions(**self._ann_output_objects, flag_ann=True)
        return (
            p_wh,
            t_wh,
            p_wf,
            p_tub_valve,
            t_tub_valve,
            p_cas_valve,
            t_cas_valve,
            p_ann,
            t_ann,
            p_inj,
            t_inj,
            status,
        )

    def __calc_pwf(
        self,
        p_fl: float,
        q_liq: float,
        wct: float,
        hydr_corr_type: Optional[str] = None,
        q_gas_inj: Optional[float] = None,
        p_gas_inj: Optional[float] = None,
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        c_choke: Optional[Union[float, dict]] = None,
        c_ann_choke: Optional[Union[float, dict]] = None,
        c_gl_valve: Optional[float] = None,
        step_length: float = 10,
        output_params: Optional[bool] = False,
        heat_balance: Optional[bool] = False,
        recalc_pb: Optional[bool] = False,
    ) -> list:
        """
        Функция расчета забойного давления итеративно по буферному давлению
        """
        p_wf, *_ = self.calc_p_iter(
            error_func=self._calc_pwh_error,
            error_func_abs=self._calc_pwh_error_abs,
            variables=(
                p_fl,
                q_liq,
                wct,
                c_choke,
                hydr_corr_type,
                q_gas_inj,
                p_gas_inj,
                friction_factor,
                grav_holdup_factor,
                c_ann_choke,
                c_gl_valve,
                step_length,
                heat_balance,
            ),
        )
        if self.choke and not c_choke:
            self._output_objects["choke"] = self.choke
        return list(
            self.calc_pwh_pwf(
                p_wf=p_wf,
                q_liq=q_liq,
                wct=wct,
                hydr_corr_type=hydr_corr_type,
                q_gas_inj=q_gas_inj,
                p_gas_inj=p_gas_inj,
                friction_factor=friction_factor,
                grav_holdup_factor=grav_holdup_factor,
                c_ann_choke=c_ann_choke,
                c_gl_valve=c_gl_valve,
                step_length=step_length,
                output_params=output_params,
                heat_balance=heat_balance,
                recalc_pb=recalc_pb,
                finalize=True,
            )
        )

    def __calc_q_gas_inj(
        self,
        p_gas_inj: float,
        p_tub_valve: float,
        t_tub_valve: float,
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        c_ann_choke: Optional[Union[float, dict]] = None,
        c_gl_valve: Optional[float] = None,
        step_length: float = 10,
        output_params: bool = False,
    ) -> list:
        """
        Функция расчета расхода газлифтного газа итеративно по давлению в газовой линии
        """

        try:
            p_cas_valve = opt.brenth(
                self.__calc_p_gas_inj_error,
                a=p_tub_valve,
                b=const.P_UP_LIMIT,
                args=(
                    p_gas_inj,
                    p_tub_valve,
                    t_tub_valve,
                    friction_factor,
                    grav_holdup_factor,
                    c_ann_choke,
                    c_gl_valve,
                    step_length,
                    output_params,
                ),
                xtol=10000,
            )
            convergence = 1
        except ValueError:
            p_cas_valve = opt.minimize_scalar(
                self.__calc_p_gas_inj_error_abs,
                method="bounded",
                bounds=(p_tub_valve, const.P_UP_LIMIT),
                args=(
                    p_gas_inj,
                    p_tub_valve,
                    t_tub_valve,
                    friction_factor,
                    grav_holdup_factor,
                    c_ann_choke,
                    c_gl_valve,
                    step_length,
                    output_params,
                ),
                options={"xatol": 10000},
            )
            p_cas_valve = p_cas_valve.x
            convergence = 0

        results = list(
            self.calc_p_gas_inj_q_gas_inj(
                p_cas_valve=p_cas_valve,
                p_tub_valve=p_tub_valve,
                t_tub_valve=t_tub_valve,
                friction_factor=friction_factor,
                grav_holdup_factor=grav_holdup_factor,
                c_ann_choke=c_ann_choke,
                c_gl_valve=c_gl_valve,
                step_length=step_length,
                output_params=output_params,
            )
        )
        results[-1] = convergence

        return results

    def __calc_p_gas_inj_error_abs(
        self,
        p_cas_valve: float,
        p_gas_inj: float,
        p_tub_valve: float,
        t_tub_valve: float,
        friction_factor: float = None,
        grav_holdup_factor: float = None,
        c_ann_choke: Optional[Union[float, dict]] = None,
        c_gl_valve: float = None,
        step_length: float = None,
        output_params: bool = False,
    ) -> float:
        """
        Расчет модуля ошибки в давлении газовой линии
        """
        return abs(
            self.__calc_p_gas_inj_error(
                p_cas_valve=p_cas_valve,
                p_gas_inj=p_gas_inj,
                p_tub_valve=p_tub_valve,
                t_tub_valve=t_tub_valve,
                friction_factor=friction_factor,
                grav_holdup_factor=grav_holdup_factor,
                c_ann_choke=c_ann_choke,
                c_gl_valve=c_gl_valve,
                step_length=step_length,
                output_params=output_params,
            )
        )

    def __calc_p_gas_inj_error(
        self,
        p_cas_valve: float,
        p_gas_inj: float,
        p_tub_valve: float,
        t_tub_valve: float,
        friction_factor: float = None,
        grav_holdup_factor: float = None,
        c_ann_choke: Optional[Union[float, dict]] = None,
        c_gl_valve: float = None,
        step_length: float = None,
        output_params: bool = False,
    ) -> float:
        """
        Расчет ошибки в давлении газовой линии
        """
        calc_pgas = self.calc_p_gas_inj_q_gas_inj(
            p_cas_valve=p_cas_valve,
            p_tub_valve=p_tub_valve,
            t_tub_valve=t_tub_valve,
            friction_factor=friction_factor,
            grav_holdup_factor=grav_holdup_factor,
            c_ann_choke=c_ann_choke,
            c_gl_valve=c_gl_valve,
            step_length=step_length,
            output_params=output_params,
        )[0]

        return p_gas_inj - calc_pgas

    def calc_p_gas_inj_q_gas_inj(
        self,
        p_cas_valve: float,
        p_tub_valve: float,
        t_tub_valve: float,
        friction_factor: float = None,
        grav_holdup_factor: float = None,
        c_ann_choke: Optional[Union[float, dict]] = None,
        c_gl_valve: Optional[float] = None,
        step_length: float = None,
        output_params: bool = False,
    ) -> tuple:
        """
        Метод расчета давления в газовой линии по известному расходу газлифтного газа

        Parameters
        ----------
        :param p_cas_valve: давление на входе в клапан, Па
        :param p_tub_valve: давление на выходе из клапана, Па
        :param t_tub_valve: температура на выходе из клапана, К
        :param friction_factor: к-т адаптации КРД на трение, если не задан берется из атрибутов трубы
        :param grav_holdup_factor: к-т адаптации КРД на гравитацию / holdup, если не задан берется из атрибутов трубы
        :param c_ann_choke: адаптационный коэффициент штуцера на газовой линии \
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value}, \
            где value - постоянный перепад, который будет использоваться как перепад между затрубным
            и давлением закачки гл.г.
        :param c_gl_valve: к-т адаптации газлифтного клапана
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг для расчета дополнительных распределений параметров

        :return: 0 - давление закачки г/г, Па
        :return: 1 - температура закачки г/г, К
        :return: 2 - расход г/г, м3/с
        :return: 3 - давление на входе в клапан (давление в затрубе на глубине спуска клапана), Па
        :return: 4 - температура на входе в клапан (температура в затрубе на глубине спуска клапана), К
        :return: 5 - затрубное давление, Па
        :return: 6 - затрубная температура, К
        :return: 7 - статус расчёта:
                     0 - расчет успешный;
                     1 - достигнуто минимальное давление, невозможно рассчитать давление закачки г/г;
                     -1 - ошибка интегрирования
        -------
        """
        # Расчет расхода газлифтного газа через клапан
        q_gas_inj = self.gl_system.valve_working.calc_qgas(
            p_in=p_cas_valve,
            p_out=p_tub_valve,
            t=t_tub_valve,
            gamma_gas=self.fluid.gamma_gas,
            cd=c_gl_valve,
        )[0]

        self.gl_system.valve_working.make_dist(
            self.gl_system.h_mes_work,
            p_tub_valve,
            t_tub_valve,
            p_cas_valve,
            t_tub_valve,
        )

        if output_params:
            fluid_valve = deepcopy(self.fluid)
            self.gl_system.valve_working.distributions.update(
                tls.make_output_attrs(
                    fluid_valve,
                    self.gl_system.valve_working.distributions["p"],
                    self.gl_system.valve_working.distributions["t"],
                )
            )

            ann_fluid_valve = deepcopy(self.annulus_fluid)
            self.gl_system.valve_working.distributions_annulus.update(
                tls.make_output_attrs(
                    ann_fluid_valve,
                    self.gl_system.valve_working.distributions["p"],
                    self.gl_system.valve_working.distributions["t"],
                )
            )

        self._ann_output_objects["gl_valves"] = self.gl_system.valve_working
        self._output_objects["gl_valves"] = self.gl_system.valve_working

        self.annulus.fluid = deepcopy(self.annulus_fluid)
        self.annulus.fluid.q_fluid = q_gas_inj
        self.annulus.fluid.reinit_q_fluid(q_gas_inj)

        t_cas_valve = self.amb_temp_dist.calc_temp(self.annulus.bottom_depth).item()
        self.annulus.bottom_depth = self.gl_system.h_mes_work

        # Расчет затрубного давления
        p_ann, t_ann, status = self.annulus.calc_pt(
            h_start="bottom",
            p_mes=p_cas_valve,
            flow_direction=1,
            q_liq=q_gas_inj,
            wct=0,
            t_mes=t_cas_valve,
            hydr_corr_type="gray",
            step_len=step_length,
            friction_factor=friction_factor,
            extra_output=output_params,
            heat_balance=False,
            grav_holdup_factor=grav_holdup_factor,
        )

        self._ann_output_objects["annulus"] = self.annulus
        if status != 0:
            # Случай неуспешного расчета затрубного давления
            return (
                p_ann,
                t_ann,
                q_gas_inj,
                p_cas_valve,
                t_cas_valve,
                p_ann,
                t_ann,
                status,
            )

        # Расчет давления закачки гл газа
        p_gas_inj, t_gas_inj, node_status = self.calc_p_choke(
            choke=self.ann_choke,
            p_received=p_ann if p_ann <= 1e8 else 1e8 - 1,
            t_received=t_ann,
            flow_direction=1,
            q_liq=q_gas_inj,
            wct=0,
            output_params=output_params,
            c_choke=c_ann_choke,
        )
        if node_status:
            self._ann_output_objects["ann_choke"] = self.ann_choke

        return (
            p_gas_inj,
            t_gas_inj,
            q_gas_inj,
            p_cas_valve,
            t_cas_valve,
            p_ann,
            t_ann,
            status,
        )
