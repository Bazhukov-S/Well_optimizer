"""
Модуль, описывающий поведение абстрактной скважины
"""
import abc
from typing import Any, Callable, Union

import pandas as pd
import scipy.optimize as opt

import ffmt.pvt.adapter as fl
import unifloc.common.ambient_temperature_distribution as amb
import unifloc.common.trajectory as traj
import unifloc.pipe.pipeline as pipe
import unifloc.service._constants as const
import unifloc.tools.exceptions as exc
from unifloc.equipment.choke import Choke


class AbstractWell(abc.ABC):
    """
    Класс абстрактной скважины

    Принимает на вход словари с исходными данными.

    Структура словарей исходных данных, ключи словарей менять нельзя:

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
                    * "p" - давление насыщения, Па абс. - float
                    * "t" - температура калибровки газосодержания при давлении насыщения, К
                            - float
                * "bob" - словарь с калибровочным значением объемного коэффициента нефти при
                давлении насыщения - dict, optional
                    * "value" - калибровочное значение объемного коэффициента нефти
                                при давлении насыщения, ст. м3/ст. м3 - float
                    * "p" - давление насыщения, Па абс. - float
                    * "t" - температура калибровки объемного коэффициента нефти
                            при давлении насыщения, К - float
                * "muob" - словарь с калибровочным значением вязкости нефти
                           при давлении насыщения - dict, optional
                    * "value" - калибровочное значение вязкости нефти при давлении насыщения, сПз - float
                    * "p" - давление насыщения, Па абс. - float
                    * "t" - температура калибровки вязкости нефти при давлении насыщения, К - float
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
    * equipment_data ("ключ" - определение - тип) - dict
        * "packer" - флаг наличия пакера в скважине, определяет наличие естественной сепарации
            - boolean, optional
        * "choke" - словарь с исходными данными для создания объекта штуцера - dict, optional
            * "d" - диаметр штуцера, м - float
            * "correlation" - тип корреляции - string, optional
    * well_trajectory_data ("ключ" - определение - тип) - dict
        * "inclinometry" - таблица с инклинометрией, две колонки: "MD","TVD", индекс по умолчанию,
            см.пример - DataFrame
        * или возможно с помощью dict с ключами "MD", "TVD"
        * Важно!: физичность вводимых данных пока не проверяется, поэтому нужно смотреть
            чтобы TVD <= MD, dTVD <= dMD
    * ambient_temperature_data ("ключ" - определение - тип) - словарь с распределением температуры
     породы по MD - dict
        * обязательные ключи MD, T - list
    """

    def __init__(
        self,
        fluid_data: dict,
        pipe_data: dict,
        equipment_data: dict,
        well_trajectory_data: dict,
        ambient_temperature_data: dict,
    ):
        """
        Parameters
        ----------
        :param fluid_data: словарь с исходными данными для создания флюида
        :param pipe_data: словарь с исходными данными для создания колонн труб
        :param equipment_data: словарь с исходными данными для создания различного оборудования
        :param well_trajectory_data: словарь с исходными данными для создания инклинометрии скважины
        :param ambient_temperature_data: словарь с распределением температуры породы по MD

        """
        # Инициализация класса для расчета флюида
        self.fluid = fl.FluidFlow(**fluid_data)
        self.fluid_data = fluid_data

        # Проверка на пропущенные нули в инклинометрии
        if (
            isinstance(well_trajectory_data["inclinometry"], pd.DataFrame)
            and not (well_trajectory_data["inclinometry"].iloc[0] == [0, 0]).all()
        ):
            well_trajectory_data["inclinometry"] = pd.concat(
                [
                    pd.DataFrame({"MD": 0, "TVD": 0}, index=[0]),
                    well_trajectory_data["inclinometry"],
                ]
            ).reset_index(drop=True)
        elif isinstance(well_trajectory_data["inclinometry"], dict) and not (
            well_trajectory_data["inclinometry"]["MD"][0] == 0 and well_trajectory_data["inclinometry"]["TVD"][0] == 0
        ):
            well_trajectory_data["inclinometry"]["MD"].insert(0, 0)
            well_trajectory_data["inclinometry"]["TVD"].insert(0, 0)

        # Инициализация класса для расчета инклинометрии скважины
        self.well_trajectory_data = well_trajectory_data
        self.well_trajectory = traj.Trajectory(**self.well_trajectory_data)

        # Инициализация класса для расчета температуры окружающей породы
        self.ambient_temperature_data = ambient_temperature_data
        self.amb_temp_dist = amb.AmbientTemperatureDistribution(self.ambient_temperature_data)

        # Инициализация класса для расчета колонны Насосно-Компрессорных Труб
        if "tubing" in pipe_data:
            pipe_data["tubing"].update(
                {
                    "fluid": self.fluid,
                    "trajectory": self.well_trajectory,
                    "top_depth": 0,
                    "ambient_temperature_distribution": self.amb_temp_dist,
                }
            )
            self.tubing = pipe.Pipeline(**pipe_data["tubing"])
        else:
            raise exc.UniflocPyError("НКТ не задано. Расчет невозможен")

        # Инициализация класса для расчета штуцера
        if equipment_data and "choke" in equipment_data:
            equipment_data["choke"].update({"h_mes": 0, "fluid": self.fluid, "d_up": self.tubing.d0})
            self.choke = Choke(**equipment_data["choke"])
        else:
            self.choke = None

        self.equipment_data = equipment_data

        # Инициализация класса для расчета Эксплуатационной Колонны
        if "casing" in pipe_data:
            pipe_data["casing"].update(
                {
                    "fluid": self.fluid,
                    "trajectory": self.well_trajectory,
                    "top_depth": self.tubing.bottom_depth,
                    "ambient_temperature_distribution": self.amb_temp_dist,
                }
            )
            self.casing = pipe.Pipeline(**pipe_data["casing"])
        else:
            raise exc.UniflocPyError("ЭК не задано. Расчет невозможен")

        self.pipe_data = pipe_data

        # Инициализация класса для расчета естественной сепарации
        if equipment_data and "packer" in self.equipment_data:
            if not self.equipment_data["packer"]:
                self.natural_sep = True
            else:
                self.natural_sep = None
        else:
            self.natural_sep = None

        # Атрибут для всех запрашиваемых значений
        self._output_objects = dict()
        self._ann_output_objects = dict()

        self._extra_output = dict()
        self._extra_output_annulus = dict()

    @property
    def extra_output(self):
        """
        Сборный атрибут со всеми требуемыми распределениями

        Словарь с распределениями свойств:

        * "p" - давление, Па
        * "t" - температура, К
        * "depth" - измеренная глубина, м
        * "rs" - газосодержание, м3/м3
        * "pb" - давление насыщения, Па
        * "muo" - вязкость нефти, сПз
        * "mug" - вязкость газа, сПз
        * "muw" - вязкость воды, сПз
        * "mu_liq" - вязкость жидкости, сПз
        * "mu_mix" - вязкость смеси, сПз
        * "z" - z - фактор
        * "bo" - объемный коэффициент нефти, м3/м3
        * "bg" - объемный коэффициент газа, м3/м3
        * "bw" - объемный коэффициент воды, м3/м3
        * "rho_oil_rc" - плотность нефти, кг/м3
        * "rho_gas_rc" - плотность газа, кг/м3
        * "rho_wat_rc" - плотность воды, кг/м3
        * "rho_liq_rc" - плотность жидкости, кг/м3
        * "rho_mix_rc" - плотность смеси, кг/м3
        * "compro" - коэффициент сжимаемости нефти, 1/Па
        * "q_oil_rc" - дебит нефти, м3/с
        * "q_gas_rc" - дебит газа, м3/с
        * "q_wat_rc" - дебит воды, м3/с
        * "q_liq_rc" - дебит жидкости, м3/с
        * "q_mix_rc" - дебит смеси, м3/с
        * "gas_fraction" - доля газа, д.ед.
        * "st_oil_gas" - поверхностное натяжение на границе нефть-газ, Н/м
        * "st_wat_gas" - гповерхностное натяжение на границе вода-газ, Н/м
        * "st_liq_gas" - поверхностное натяжение на границе жидкость-газ, Н/м
        * "dp_dl" - градиент давления, Па/м
        * "dp_dl_fric" - радиент давления с учетом трения, Па/м
        * "dp_dl_grav" - радиент давления с учетом гравитации, Па/м
        * "dp_dl_acc" - радиент давления с учетом инерции, Па/м
        * "liquid_holdup" - истинное содержание жидкости
        * "friction_factor" - коэффициент трения
        * "vsl" - приведенная скорость жидкости, м/с
        * "vsg" - приведенная скорость газа, м/с
        * "vsm" - приведенная скорость смеси, м/с
        * "flow_pattern" - режим потока (0,1,2 или 3)
        * "lambda_l" - объемное содержание жидкости
        * "n_re" - число Рейнольдса
        * "angel" - угол наклона трубы, градусы
        * "vl" - скорость жидкости, м/с
        * "vg" - скорость газа, м/с
        * "nodes" - ключевые узлы

        Для скважины, оборудованной УЭЦН, скорректированные характеристики НРХ:
        * "rate_points_corr" - дебит, м3/с
        * "head_points_corr" - напор, м
        * "power_points_corr" - мощность, кВт
        * "eff_points_corr" - КПД насоса, д.ед.

        Для газлифтной скважины с несколькими клапанами:
        * "valves" - список словарей с информацией по давлениям для каждого газлифтного клапана

        """
        return self._extra_output

    @property
    def extra_output_annulus(self):
        """
        Сборный атрибут со всеми требуемыми распределениями для затруба
        Для газлифтной скважины.

        Словарь с распределениями свойств:

        * "p" - давление, Па
        * "t" - температура, К
        * "depth" - измеренная глубина, м
        * "rs" - газосодержание, м3/м3
        * "pb" - давление насыщения, Па
        * "muo" - вязкость нефти, сПз
        * "mug" - вязкость газа, сПз
        * "muw" - вязкость воды, сПз
        * "mu_liq" - вязкость жидкости, сПз
        * "mu_mix" - вязкость смеси, сПз
        * "z" - z - фактор
        * "bo" - объемный коэффициент нефти, м3/м3
        * "bg" - объемный коэффициент газа, м3/м3
        * "bw" - объемный коэффициент воды, м3/м3
        * "rho_oil_rc" - плотность нефти, кг/м3
        * "rho_gas_rc" - плотность газа, кг/м3
        * "rho_wat_rc" - плотность воды, кг/м3
        * "rho_liq_rc" - плотность жидкости, кг/м3
        * "rho_mix_rc" - плотность смеси, кг/м3
        * "compro" - коэффициент сжимаемости нефти, 1/Па
        * "q_oil_rc" - дебит нефти, м3/с
        * "q_gas_rc" - дебит газа, м3/с
        * "q_wat_rc" - дебит воды, м3/с
        * "q_liq_rc" - дебит жидкости, м3/с
        * "q_mix_rc" - дебит смеси, м3/с
        * "gas_fraction" - доля газа, д.ед.
        * "st_oil_gas" - поверхностное натяжение на границе нефть-газ, Н/м
        * "st_wat_gas" - гповерхностное натяжение на границе вода-газ, Н/м
        * "st_liq_gas" - поверхностное натяжение на границе жидкость-газ, Н/м
        * "dp_dl" - градиент давления, Па/м
        * "dp_dl_fric" - радиент давления с учетом трения, Па/м
        * "dp_dl_grav" - радиент давления с учетом гравитации, Па/м
        * "dp_dl_acc" - радиент давления с учетом инерции, Па/м
        * "liquid_holdup" - истинное содержание жидкости
        * "friction_factor" - коэффициент трения
        * "vsl" - приведенная скорость жидкости, м/с
        * "vsg" - приведенная скорость газа, м/с
        * "vsm" - приведенная скорость смеси, м/с
        * "flow_pattern" - режим потока (0,1,2 или 3)
        * "lambda_l" - объемное содержание жидкости
        * "n_re" - число Рейнольдса
        * "angel" - угол наклона трубы, градусы
        * "vl" - скорость жидкости, м/с
        * "vg" - скорость газа, м/с
        * "nodes" - ключевые узлы
        """
        return self._extra_output_annulus

    def _reinit(self, wct, q_liq):
        """
        Сброс флюида и пересохранение объекта в других объектах

        :param wct: обводненность, д.ед.
        :param q_liq: дебит жидкости, м/c
        """
        self._extra_output = dict()
        self.fluid.reinit()
        self.fluid.reinit_wct(wct)
        self.fluid.reinit_q_fluid(q_liq)
        self.fluid.wct = wct
        self.fluid.q_fluid = q_liq

    @staticmethod
    def calc_p_choke(
        choke: Choke,
        p_received: float,
        t_received: float,
        q_liq: float,
        wct: float,
        flow_direction: float,
        output_params: bool,
        c_choke: float,
    ) -> tuple:
        """
        Расчет давления и температуры до/после штуцера.
        :param choke: объект штуцера
        :param p_received: заданное давление до/после штуцера, Па
        :param t_received: заданная температура до/после штуцера, К
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param flow_direction: направление потока (1 - к p_received; -1 - от p_received)
        :param output_params: флаг расчета дополнительных распределений
        :param c_choke: коэффициент калибровки штуцера

        :return: давление до/после штуцера, Па
        :return: температура до/после штуцера, К
        :return: флаг сохранения атрибута
        """
        # Флаг для сохранения атрибута
        node_status = False

        if choke:
            if isinstance(c_choke, dict) and "const" in c_choke:
                p_calc, t_calc = p_received + (c_choke["const"] * flow_direction), t_received
            else:
                try:
                    p_calc, t_calc, *_ = choke.calc_pt(
                        p_mes=p_received,
                        t_mes=t_received,
                        flow_direction=flow_direction,
                        q_liq=q_liq,
                        wct=wct,
                        phase_ratio_value=None,
                        c_choke=c_choke,
                        extra_output=output_params,
                        catch_supercritical=False,
                    )
                    node_status = True
                except exc.UniflocPyError:
                    p_calc, t_calc = p_received, t_received
        else:
            p_calc, t_calc = p_received, t_received

        return p_calc, t_calc, node_status

    @abc.abstractmethod
    def calc_pwh_pwf(self, *args, **kwargs):
        """
        Расчет буферного давления через давление на забое с учетом всех гидравлических элементов.
        """

    def _calc_pwh_error(
        self, p_wf: float, p_fl: float, q_liq: float, wct: float, c_choke: Union[float, dict], *args
    ) -> float:
        """
        Расчет ошибки давления на буфере.

        :param p_wf: давление на забое скважины, Па
        :param p_fl: давление в линии, Па
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param c_choke: коэффициент калибровки штуцера

        :return: разница рассчитанного давления с заданным
        """
        try:
            p_wh_down, t_wh_down, *_ = self.calc_pwh_pwf(p_wf, q_liq, wct, *args)
            p_wh_top, _, node_status = self.calc_p_choke(
                choke=self.choke,
                p_received=p_fl,
                t_received=t_wh_down,
                q_liq=q_liq,
                wct=wct,
                flow_direction=1,
                output_params=True,
                c_choke=c_choke,
            )
            if node_status:
                self._output_objects["choke"] = self.choke

            return p_wh_top - p_wh_down
        except exc.OptimizationStatusError:
            # Костыль до разработки оптимизатора с динамическими границами
            return -9999999999

    def _calc_pwh_error_abs(
        self, p_wf: float, p_fl: float, q_liq: float, wct: float, c_choke: Union[float, dict], *args
    ) -> float:
        """
        Расчет абсолютной ошибки давления на буфере.

        :param p_wf: давление на забое скважины, Па
        :param p_fl: давление в линии, Па
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param c_choke: коэффициент калибровки штуцера

        :return: разница рассчитанного давления с заданным
        """
        return abs(self._calc_pwh_error(p_wf, p_fl, q_liq, wct, c_choke, *args))

    @staticmethod
    def calc_p_iter(
        error_func: Callable[[Any], float],
        error_func_abs: Callable[[Any], float],
        variables: tuple,
    ) -> tuple:
        """
        Функция для итеративного подбора давления.

        :param error_func: функция расчета ошибки
        :param error_func_abs: функция расчета абсолютной ошибки
        :param variables: переменные, необходимые для расчета

        :return: давление, Па
        """
        try:
            p = opt.brenth(
                error_func,  # noqa
                a=101325,
                b=const.P_UP_LIMIT,  # 700 атм
                args=variables,
                xtol=10000,
            )
            convergence = 1
        except ValueError:
            pres = opt.minimize_scalar(
                error_func_abs,
                method="bounded",
                bounds=(101325, const.P_UP_LIMIT),  # 700 атм
                args=variables,
                options={"xatol": 50000},
            )
            p = pres.x
            convergence = 0

        return p, convergence

    @abc.abstractmethod
    def calc_pwf_pfl(self, *args, **kwargs):
        """
        Расчет забойного давления по линейному с учетом всех гидравлических элементов
        """
