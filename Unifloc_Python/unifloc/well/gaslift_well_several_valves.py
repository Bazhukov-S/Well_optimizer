from copy import deepcopy
from typing import Optional, Union

import scipy.interpolate as interp

import unifloc.equipment.choke as chk
import unifloc.equipment.gl_system as gl_sys
import unifloc.equipment.gl_valve as gl_vl
import unifloc.pipe.annulus as ann
import unifloc.service._constants as const
import unifloc.service._tools as tls
import unifloc.well._well as abw
import unifloc.well.gaslift_well_one_valve as gl_well_vl


class GasLiftWellSeveralValves(abw.AbstractWell):
    """
    Класс для расчета газлифтной скважины с несколькими открытыми клапанами.

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
        >>> from unifloc.well.gaslift_well_several_valves import GasLiftWellSeveralValves
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
        ...    "valve1": {"h_mes": 1300, "d": 0.003, "s_bellow": 0.000199677,
        ...               "p_valve": uc.convert_pressure(50, "atm", "Pa"),
        ...               "valve_type": "ЦКсОК"},
        ...    "valve2": {"h_mes": 1100, "d": 0.004, "s_bellow": 0.000195483,
        ...               "p_valve": uc.convert_pressure(60, "atm", "Pa"),
        ...               "valve_type": "ЦКсОК"},
        ...    "valve3": {"h_mes": 800, "d": 0.005, "s_bellow": 0.000199032,
        ...               "p_valve": uc.convert_pressure(40, "atm", "Pa"),
        ...               "valve_type": "ЦКсОК"},
        ...    "valve4": {"h_mes": 900, "d": 0.004, "s_bellow": 0.000199032,
        ...               "p_valve": uc.convert_pressure(50, "atm", "Pa"),
        ...               "valve_type": "ЦКсОК"}},
        ...    }
        >>> ambient_temperature_data = {"MD": [0, 1800], "T": [293.15, 303.15]}
        >>>
        >>> # Инициализация объекта скважины
        >>> well = GasLiftWellSeveralValves(fluid_data, pipe_data, well_trajectory_data,
        ...                    ambient_temperature_data, equipment_data)
        >>> # Расчет забойного давления
        >>> p_fl = 10 * 101325
        >>> p_inj = 100 * 101325
        >>> # Расчет с сохранением доп. атрибутов распределений свойств
        >>> p_wf = well.calc_pwf_pfl(p_fl=p_fl, q_liq=uc.convert_rate(100, "m3/day", "m3/s"), wct=0.1,
        ...                          p_gas_inj=p_inj,
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

        self.glw_one_valve = gl_well_vl.GasLiftWellOneValve(
            fluid_data,
            pipe_data,
            well_trajectory_data,
            ambient_temperature_data,
            equipment_data,
        )

        if equipment_data is not None and "gl_system" in equipment_data and equipment_data["gl_system"] is not None:
            self.gl_system = gl_sys.GlSystem(equipment_data["gl_system"])
        else:
            # Считаем, что клапанов нет
            self.gl_system = gl_sys.GlSystem({"valve": {"h_mes": self.tubing.bottom_depth, "d": 0.006}})

        self.gl_system_init = deepcopy(self.gl_system)

        # Флюид для расчета затрубного пространства
        self.annulus_fluid = deepcopy(self.fluid)
        self.annulus_fluid.fluid_type = "gas"
        self.annulus_fluid.reinit_fluid_type("gas")

        # Инициализация класса для расчета затрубного пространства
        if "casing" in pipe_data and "tubing" in pipe_data:
            self.annulus = ann.Annulus(
                fluid=self.annulus_fluid,
                bottom_depth=self.gl_system.valves[0].h_mes,
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
            self.ann_choke = chk.Choke(**equipment_data["ann_choke"])
        else:
            self.ann_choke = None

    def __make_nodes(self) -> list:
        """
        Создание распределений ключевых узлов

        :return: список ключевых расчетных узлов в НКТ скважины (от верхних дыр перфорации до линии)
                и газлифтных клапанов (с указанием рабочего), распределенных по измеренной глубине
        """
        nodes = [None for _ in self.extra_output["depth"]]
        nodes[-1] = "Верхние дыры перфорации"

        valve_count = len(self.gl_system.valves)
        for vl in range(valve_count):
            for par in range(len(self.extra_output["depth"])):
                if self.gl_system.valves[vl].h_mes * (-1) == self.extra_output["depth"][par]:
                    nodes[par] = "Газлифтный клапан " + str(valve_count - vl)
                    if self.gl_system.valve_working:
                        if self.gl_system.valves[vl].h_mes == self.gl_system.valve_working.h_mes:
                            nodes[par] = "Рабочий газлифтный клапан " + str(valve_count - vl)
                    break

        if self.tubing.bottom_depth * (-1) in self.extra_output["depth"]:
            tub_index = self.extra_output["depth"].index(self.tubing.bottom_depth * (-1))
            if nodes[tub_index] is not None:
                nodes[tub_index] = ''.join([nodes[tub_index], ", Башмак НКТ"])
            else:
                nodes[tub_index] = "Башмак НКТ"

        if self.choke:
            nodes[0] = "Линия"
            nodes[2] = "Буфер"
        else:
            nodes[0] = "Буфер"
        return nodes

    def __make_nodes_annulus(self) -> list:
        """
        Создание распределений ключевых узлов для затрубного пространства

        :return: список ключевых расчетных узлов затрубного пространства скважины
                (от нижнего газлифтного клапана до компрессора) и газлифтных клапанов
                (с указанием рабочего), распределенных по измеренной глубине
        """
        nodes = [None for _ in self.extra_output_annulus["depth"]]

        valve_count = len(self.gl_system.valves)
        for vl in range(valve_count):
            for par in range(len(self.extra_output_annulus["depth"])):
                if self.gl_system.valves[vl].h_mes * (-1) == self.extra_output_annulus["depth"][par]:
                    nodes[par] = "Газлифтный клапан " + str(valve_count - vl)
                    if self.gl_system.valve_working:
                        if self.gl_system.valves[vl].h_mes == self.gl_system.valve_working.h_mes:
                            nodes[par] = "Рабочий газлифтный клапан " + str(valve_count - vl)
                    break

        if self.ann_choke:
            nodes[0] = "Газовая линия"
            nodes[2] = "Затруб"
        else:
            nodes[0] = "Затруб"

        return nodes

    @staticmethod
    def __interpolation(x_curve, y_curve, x_point):
        """
        Метод интерполяционного расчета параметра по заданным зависимостям

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
        return interp_func(x_point).item()

    def calc_pwh_pwf(self, *args, **kwargs):
        """
        Расчет буферного давления по забойному с учетом всех гидравлических элементов
        """
        pass

    def calc_pwf_pfl(
        self,
        p_fl: float,
        q_liq: float,
        wct: float,
        p_gas_inj: float,
        p_wh: Optional[float] = None,
        p_ann: Optional[float] = None,
        hydr_corr_type: Optional[str] = None,
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        c_choke: Optional[Union[float, dict]] = None,
        c_ann_choke: Optional[Union[float, dict]] = None,
        c_gl_valve: Optional[float] = None,
        step_length: float = 10,
        output_params: bool = False,
        heat_balance: bool = False,
        recalc_pb: bool = False,
    ) -> list:
        """
        * Расчет забойного давления по линейному с учетом всех гидравлических элементов
        * Выбор в качестве рабочего клапана наиболее верхнего из открытых

        Parameters
        ----------
        :param p_fl: линейное давление, Па абс.
        :param q_liq: дебит жидкости, ст. м3/с
        :param wct: обводненность, доли ед.
        :param p_gas_inj: давление закачки газлифтного газа, Па
        :param p_wh: буферное давление, Па абс.
        :param p_ann: затрубное давление, Па абс.
        :param hydr_corr_type: тип гидравлической корреляции
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

        :return: 0 - расчетное давление закачки газлифтного газа, Па
        :return: 1 - температура закачки газлифтного газа, К
        :return: 2 - забойное давление, Па
        :return: 3 - давление на выходе из рабочего клапана (давление в НКТ на глубине спуска клапана), Па
        :return: 4 - температура на выходе из рабочего клапана (температура в НКТ на глубине спуска клапана), К
        :return: 5 - статус расчёта:
                     0 - расчет успешный;
                     1 - достигнуто минимальное давление, невозможно рассчитать линейное давление;
                     -1 - ошибка интегрирования
        :return: 6 - давление на входе в рабочий клапан (давление в затрубе на глубине спуска клапана), Па
        :return: 7 - температура на входе в рабочий клапан (температура в затрубе на глубине спуска клапана), К
        :return: 8 - закачка газлифтного газа, ст. м3/с
        :return: 9 - затрубное давление, Па
        :return: 10 - затрубная температура, К
        :return: 11 - расчетное буферное давление, Па
        -------
        """

        self._output_objects = {"params": ["p", "t", "depth"]}
        self._ann_output_objects = {"params": ["p", "t", "depth"]}

        if output_params:
            self._output_objects["params"] += const.DISTRS
            self._ann_output_objects["params"] += const.DISTRS

        # Расчет забойного давления и расхода газлифтного газа для каждого клапана
        results = {
            "result": [],
            "extra_output": [],
            "extra_output_annulus": [],
            "valves": [],
            "status": [],
        }

        h_valve_min = self.gl_system_init.h_mes_work
        i, value = 0, 0
        for valve in self.gl_system.valves:
            if valve.d != 0:
                one_vl_well = deepcopy(self.glw_one_valve)

                output_obj = deepcopy(self._output_objects)
                ann_output_obj = deepcopy(self._ann_output_objects)

                one_vl_well.gl_system.valve_working = valve
                self.gl_system.valve_working = valve

                one_vl_res = one_vl_well.calc_pwf_pfl(
                    p_fl=p_fl,
                    q_liq=q_liq,
                    wct=wct,
                    p_wh=p_wh,
                    p_ann=p_ann,
                    hydr_corr_type=hydr_corr_type,
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

                if one_vl_res[5] != 0:
                    continue

                results["result"].append(one_vl_res)

                # Расчет распределения давления в затрубе по рассчитанному затрубному давлению и расходу гл. газа
                self.annulus.fluid = deepcopy(self.annulus_fluid)
                self.annulus.fluid.q_fluid = one_vl_res[8]
                self.annulus.fluid.reinit_q_fluid(one_vl_res[8])
                self.annulus.calc_pt(
                    h_start="top",
                    p_mes=one_vl_res[9],
                    flow_direction=-1,
                    q_liq=one_vl_res[8],
                    wct=0,
                    t_mes=one_vl_res[10],
                    hydr_corr_type="gray",
                    step_len=step_length,
                    friction_factor=friction_factor,
                    extra_output=output_params,
                    heat_balance=heat_balance,
                    grav_holdup_factor=grav_holdup_factor,
                )
                ann_output_obj["annulus"] = self.annulus

                valves_arr = []
                ann_output_obj["gl_valves"] = []
                output_obj["gl_valves"] = []
                correct_ans = False
                for k, vl in enumerate(self.gl_system.valves):
                    valves_arr.append(
                        self.__make_valves_attrs(
                            valve=vl,
                            tub_dist=one_vl_well.extra_output,
                            cas_dist=self.annulus.distributions,
                            output_params=output_params,
                        )
                    )

                    ann_output_obj["gl_valves"].insert(0, vl)
                    output_obj["gl_valves"].insert(0, vl)

                    if vl == valve and vl.status == gl_vl.GlValve.RETURN_STATUS.OPEN.name:
                        valves_arr[k]["status"] += "_work"
                        correct_ans = True

                results["status"].append(correct_ans)

                output_obj["casing"] = one_vl_well.casing

                if one_vl_well.tubing_below_point:
                    output_obj["tubings"] = [one_vl_well.tubing_below_point]
                else:
                    output_obj["tubings"] = []
                output_obj["tubings"].insert(0, one_vl_well.tubing_upper_point)

                if isinstance(c_choke, dict) is False and self.choke:
                    output_obj["choke"] = one_vl_well.choke

                if isinstance(c_ann_choke, dict) is False and self.ann_choke:
                    ann_output_obj["ann_choke"] = one_vl_well.ann_choke

                results["valves"].append(valves_arr)
                results["extra_output"].append(output_obj)
                results["extra_output_annulus"].append(ann_output_obj)

                if valve.h_mes < h_valve_min and correct_ans:
                    h_valve_min = valve.h_mes
                    value = i

                i += 1

        if not results["status"] or (value == 0 and not results["status"][0]):
            self.gl_system.valve_working = None
            output_obj = deepcopy(self._output_objects)
            ann_output_obj = deepcopy(self._ann_output_objects)

            result = self.glw_one_valve.calc_pwf_pfl(
                p_fl=p_fl,
                q_liq=q_liq,
                wct=wct,
                p_wh=p_wh,
                p_ann=p_ann,
                hydr_corr_type=hydr_corr_type,
                q_gas_inj=0,
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

            vls = []
            ann_output_obj["gl_valves"] = []
            output_obj["gl_valves"] = []
            for k, vl in enumerate(self.gl_system.valves):
                vls.append(
                    self.__make_valves_attrs(
                        valve=vl,
                        tub_dist=self.glw_one_valve.extra_output,
                        output_params=output_params,
                    )
                )
                ann_output_obj["gl_valves"].insert(0, vl)
                output_obj["gl_valves"].insert(0, vl)

            output_obj["casing"] = self.glw_one_valve.casing

            if self.glw_one_valve.tubing_below_point:
                output_obj["tubings"] = [self.glw_one_valve.tubing_below_point]
            else:
                output_obj["tubings"] = []

            output_obj["tubings"].insert(0, self.glw_one_valve.tubing_upper_point)

            if isinstance(c_choke, dict) is False and self.choke:
                output_obj["choke"] = self.glw_one_valve.choke

            if isinstance(c_ann_choke, dict) is False and self.ann_choke:
                ann_output_obj["ann_choke"] = self.glw_one_valve.ann_choke

        else:
            self.gl_system.valve_working = self.gl_system.valves[value]
            result = results["result"][value]
            output_obj = results["extra_output"][value]
            ann_output_obj = results["extra_output_annulus"][value]
            vls = results["valves"][value]

            self._ann_output_objects = ann_output_obj
            self._extra_output_annulus = tls.make_unified_distributions(**self._ann_output_objects, flag_ann=True)
            self._extra_output_annulus["nodes"] = self.__make_nodes_annulus()

        self._output_objects = output_obj
        self._extra_output = tls.make_unified_distributions(**self._output_objects)
        self._extra_output["nodes"] = self.__make_nodes()
        self._extra_output["valves"] = vls

        return result

    def __make_valves_attrs(
        self,
        valve: gl_vl.GlValve,
        tub_dist: dict = None,
        cas_dist: dict = None,
        output_params: bool = False,
    ):
        """
        Метод внесения давлений на входе и выходе клапана в атрибуты соответствующего класса GlValve,
        а также пересчет давлений зарядки сильфона в рабочих условиях, открытия/закрытия, статуса
        и сохранение распределений для клапана

        Parameters
        ----------
        :param valve: объект класса GlValve
        :param tub_dist: словарь с распределением давления и температуры в НКТ
        :param cas_dist: словарь с распределением давления и температуры в затрубе
        :param output_params: флаг для расчета дополнительных распределений параметров

        :return: словарь, содержащий данные по клапану:
            - "h_mes" - глубина спуска клапана, м
            - "status" - статус клапана
            - "p_cas" - давление на входе в клапан (в затрубном пространстве), Па
            - "t_cas" - температура на входе в клапан (в затрубном пространстве), К
            - "p_tub" - давление на выходе из клапана (в НКТ), Па
            - "t_tub" - температура на выходе из клапана (в НКТ), К
            - "p_open" - давление открытия клапна, Па
            - "p_close" - давление закрытия клапана, Па
        -------
        """
        valve.reinit()
        crit_regime = False

        p_d = valve.dome_charge_pressure()
        p_c = valve.close_pressure(p_d)

        if tub_dist:
            # Расчет давления в НКТ напротив газлифтного клапана
            valve.p_tub = self.__interpolation(
                tub_dist["depth"],
                tub_dist["p"],
                valve.h_mes * (-1),
            )
            # Расчет температуры в НКТ напротив газлифтного клапана
            valve.t_tub = self.__interpolation(
                tub_dist["depth"],
                tub_dist["t"],
                valve.h_mes * (-1),
            )
            # Расчет давления открытия газлифтного клапана
            p_o = valve.open_pressure(
                p_c,
                valve.p_tub,
                valve.t_tub,
            )

        if cas_dist:
            # Расчет давления в затрубном пространстве напротив газлифтного клапана
            valve.p_cas = self.__interpolation(
                cas_dist["depth"],
                cas_dist["p"],
                valve.h_mes,
            )
            # Расчет температуры в затрубном пространстве напротив газлифтного клапана
            valve.t_cas = self.__interpolation(
                cas_dist["depth"],
                cas_dist["t"],
                valve.h_mes,
            )

        if tub_dist and cas_dist:
            valve.valve_status(valve.p_cas, valve.t_cas, valve.p_tub, p_o)
            if valve.p_tub / valve.p_cas <= const.CPR:
                crit_regime = True

        # Формирование распределения давлений для газлифтного клапана
        valve.make_dist(valve.h_mes, valve.p_tub, valve.t_tub, valve.p_cas, valve.t_cas)

        if output_params:
            if tub_dist:
                fluid_valve = deepcopy(self.fluid)
                valve.distributions.update(
                    tls.make_output_attrs(fluid_valve, valve.distributions["p"], valve.distributions["t"])
                )
            if cas_dist:
                ann_fluid_valve = deepcopy(self.annulus_fluid)
                valve.distributions_annulus.update(
                    tls.make_output_attrs(
                        ann_fluid_valve,
                        valve.distributions_annulus["p"],
                        valve.distributions_annulus["t"],
                    )
                )
        return {
            "h_mes": valve.h_mes,
            "status": valve.status,
            "p_cas": valve.p_cas,
            "t_cas": valve.t_cas,
            "p_tub": valve.p_tub,
            "t_tub": valve.t_tub,
            "p_open": valve.p_open,
            "p_close": valve.p_close,
            "regime": crit_regime,
        }
