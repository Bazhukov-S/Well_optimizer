"""
Модуль, для описания класса для расчета давления и температуры в трубах
"""
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.misc as misc
import scipy.optimize as opt

import ffmt.pvt.adapter as flow
import unifloc.common.ambient_temperature_distribution as amb
import unifloc.common.trajectory as tr
import unifloc.pipe._pipe as pipe
import unifloc.tools.exceptions as exc


class Pipeline:
    """
    Класс трубопровода. Служит для расчета кривых
    распределения давления и температуры в трубопроводе при помощи различных
    многофазных корреляций.
    """

    __slots__ = [
        "top_depth",
        "bottom_depth",
        "d",
        "trajectory",
        "amb_temp_dist",
        "_fluid",
        "_friction_factor",
        "_grav_holdup_factor",
        "_hydr_corr_type",
        "pipe_object",
        "_last_status",
        "distributions",
        "d0",
        "s_wall",
    ]

    def __init__(
        self,
        top_depth: float,
        bottom_depth: float,
        d: Union[float, pd.DataFrame, dict],
        roughness: float,
        trajectory: tr.Trajectory,
        fluid: flow.FluidFlow,
        ambient_temperature_distribution: amb.AmbientTemperatureDistribution,
        friction_factor: float = 1,
        holdup_factor: float = 1,
        s_wall: float = 0.0055,
    ):
        """

        Parameters
        ----------
        :param top_depth: верхняя измеренная глубина трубопровода, м
        :param bottom_depth: нижняя измеренная глубина трубопровода, м
        :param d: внутренний диаметр, м. Можно задавать в виде таблицы формата
                  pd.DataFrame или dict или одним числом float, int
        :param roughness: шероховатость трубы, м
        :param trajectory: объект с инклинометрией, считается, что MD в одной
                           системе отсчета с top_depth и bottom_depth
        :param fluid: объект PVT модели флюида
        :param ambient_temperature_distribution: объект распределения температуры, считается
                                                 что MD в одной системе отсчета
                                                 с top_depth и bottom_depth
        :param friction_factor: коэффициент калибровки для трения
        :param holdup_factor: коэффициент калибровки для истинного содержания жидкости
        :param s_wall: толщина стенки, м

        Examples:
        --------
        >>> import pandas as pd
        >>> import unifloc.tools.units_converter as uc
        >>> import unifloc.common.trajectory as traj
        >>> import unifloc.common.ambient_temperature_distribution as amb
        >>> import unifloc.pipe.pipeline as pipel
        >>> import unifloc.pvt.fluid_flow as fl
        >>> # Инициализация исходных данных класса Pipeline
        >>> trajectory = traj.Trajectory(pd.DataFrame(columns=["MD", "TVD"],
        ...                                      data=[[float(0), float(0)],
        ...                                            [float(1800), float(1800)]]))
        >>> ambient_temperature_data = {"MD": [0, 1800], "T": [303.15, 363.15]}
        >>> amb_temp = amb.AmbientTemperatureDistribution(ambient_temperature_data)
        >>> fluid_data = {
        ...     "q_fluid": uc.convert_rate(297.1, "m3/day", "m3/s"),
        ...     "pvt_model_data": {
        ...         "black_oil": {
        ...             "gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.8,
        ...             "wct": 0.2, "phase_ratio": {"type": "GOR", "value": 401},
        ...             "rsb": {"value": 90, "p": 15000000, "t": 363.15},
        ...             "muob": {"value": 0.5, "p": 15000000, "t": 363.15},
        ...             "bob": {"value": 1.5, "p": 15000000, "t": 363.15}, }}}
        >>> fluid = fl.FluidFlow(**fluid_data)
        >>> # Так тоже возможно: d = pd.DataFrame(columns=["MD", "d"],
        >>> #                                     data=[[0, 0.062], [1000, 0.082]])
        >>> # Так тоже возможно: d= {"MD": [0, 1000], "d": [0.06, 0.08]}
        >>> d = 0.06
        >>> top_depth = 0
        >>> bottom_depth = 1800
        >>> roughness = 0.0001
        >>> p_wh = 20 * 101325  # Па
        >>> flow_direction = 1
        >>> # Инициализация объекта pvt-модели
        >>> pipe = pipel.Pipeline(top_depth, bottom_depth, d, roughness,
        ...                       trajectory, fluid, amb_temp)
        >>> # Расчет давления и температуры на забое от буфера
        >>> results = pipe.calc_pt("top", p_wh, flow_direction)
        """

        self.top_depth = top_depth
        self.bottom_depth = bottom_depth
        self.s_wall = s_wall

        if isinstance(d, (float, int)):
            self.d = d
            self.d0 = d
        elif isinstance(d, (pd.DataFrame, dict)):
            self.d = interp.interp1d(d["MD"], d["d"], fill_value="extrapolate", kind="previous")
            self.d0 = self.d(top_depth).item()
        else:
            raise TypeError(f"Неподдерживаемый тип данных для диаметра - {type(d)}")

        self.trajectory = trajectory
        self.amb_temp_dist = ambient_temperature_distribution
        self._fluid = fluid
        self._friction_factor = friction_factor
        self._grav_holdup_factor = holdup_factor

        # инициализация трубы
        self._hydr_corr_type = "beggsbrill"
        self.pipe_object = pipe.Pipe(
            fluid=self.fluid,
            d=self.d0,
            roughness=roughness,
            hydr_corr_type=self.hydr_corr_type,
            s_wall=self.s_wall,
        )
        self._last_status = None

        # инициализация словаря распределений
        self.distributions = dict()

    @property
    def friction_factor(self):
        return self._friction_factor

    @property
    def grav_holdup_factor(self):
        return self._grav_holdup_factor

    @friction_factor.setter
    def friction_factor(self, new_value):
        self._friction_factor = new_value

    @grav_holdup_factor.setter
    def grav_holdup_factor(self, new_value):
        self._grav_holdup_factor = new_value

    @property
    def fluid(self):
        return self._fluid

    @fluid.setter
    def fluid(self, new_fluid):
        """
        Изменение флюида в Pipeline и дочернем классе pipe

        Parameters
        ----------
        :param new_fluid: новый флюид, объект класса Fluid_Flow

        """
        self._fluid = new_fluid
        self.pipe_object.fluid = self._fluid

    @property
    def hydr_corr_type(self):
        return self._hydr_corr_type

    @hydr_corr_type.setter
    def hydr_corr_type(self, new_hydr_corr_type):
        """
        Изменение типа корреляции в Pipeline и дочерних классах

        Parameters
        ----------
        :param new_hydr_corr_type: тип гидравлической корреляции

        """
        if new_hydr_corr_type is not None:
            self.pipe_object.hydrcorr = new_hydr_corr_type
            self._hydr_corr_type = new_hydr_corr_type

    def make_extra_distributions(
        self,
        pressure_array: np.ndarray,
        temperature_array: np.ndarray,
        depth_array: np.ndarray,
        well_trajectory: tr.Trajectory,
        amb_temp_dist: amb.AmbientTemperatureDistribution,
        directions: np.ndarray,
        grav_holdup_factor: float,
        friction_factor: float,
        d_func: interp.interp1d,
        d_tub_out_func: interp.interp1d,
        d_cas_in_func: interp.interp1d,
        heat_balance: bool,
    ):
        """
        Расчет и сохранение экстра-выводных параметров в словарь распределений

        Parameters
        ----------
        :param pressure_array: массив давлений
        :param temperature_array: массив температур
        :param depth_array: массив глубин
        :param well_trajectory: объект с траекторией скважины
        :param amb_temp_dist: объект с распределением температуры породы
        :param directions: множитель для направления потока, флаг направления расчета
        :param grav_holdup_factor: коэффициент гравитации/проскальзывания
        :param friction_factor: коэффициент трения
        :param d_tub_out_func: объект расчет диаметра НКТ по глубине скважины
        :param d_cas_in_func: объект расчет диаметра ЭК по глубине скважины
        :param heat_balance: опция учета теплопотерь

        :return: обновляет атрибут со всеми распределениями
        """
        self.distributions.update(
            self.pipe_object.calc_extra_output(
                pressure_array,
                temperature_array,
                depth_array,
                well_trajectory,
                amb_temp_dist,
                directions,
                grav_holdup_factor,
                friction_factor,
                d_func,
                d_tub_out_func,
                d_cas_in_func,
                heat_balance,
            )
        )

    def calc_pt(
        self,
        h_start: str,
        p_mes: float,
        flow_direction: float,
        q_liq: Optional[float] = None,
        wct: Optional[float] = None,
        phase_ratio_value: Optional[float] = None,
        t_mes: Optional[float] = None,
        hydr_corr_type: Optional[str] = None,
        step_len: Optional[float] = None,
        int_method: str = "RK23",
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        extra_output: bool = False,
        steps: Optional[list] = None,
        heat_balance: bool = False,
    ) -> Tuple[float, float, int]:
        """
        Расчет давления и температуры на конце трубы

        Parameters
        ----------

        :param h_start: стартовая точка, "top" или "bottom"
        :param p_mes: измеренное давление на h_start, Па абс.
        :param flow_direction: направление движения потока, флаг
                               1 - к h_start;
                               -1 - от h_start
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, доли ед.
        :param phase_ratio_value: соотношение газа к флюиду для выбранного типа фазового соотношения,
            ст. м3 газа/ст. м3 флюид
        :param t_mes: измеренная температура на глубине h_mes, К
        :param hydr_corr_type: тип гидравлической корреляции ("BeggsBrill", "Gray", "Static")
        :param step_len: шаг расчета кривой распределения давления по стволу, м
        :param int_method: метод интегрирования: ‘RK45’, ‘RK23’, ‘DOP853’, ‘Radau’, ‘BDF’, ‘LSODA’
        :param friction_factor: коэффициент поправки на трение
        :param grav_holdup_factor: коэффициент поправки на истинное содержание жидкости
        :param extra_output: флаг для дополнительного расчета распределений и сохранения в атрибуты
        :param steps: массив глубин, на которых необходимо рассчитать давления
        :param heat_balance: опция учета теплопотерь

        Returns
        -------
        :return: давление на другом конце трубы, Па абс.;
        :return: температура на другом конце трубы, К;
        :return: статус расчета:
                 0 - успешен;
                 1 - во время расчета достигнуто минимальное давление;
                 -1 - завершен с ошибкой интегрирования
        """
        if wct is not None:
            self.fluid.reinit_wct(wct)

        if phase_ratio_value is not None:
            fluid_t = deepcopy(self.fluid)
            pvt_data = fluid_t.pvt_model_data
            phase_ratio = pvt_data["black_oil"]["phase_ratio"]
            phase_ratio.update({"value": phase_ratio_value})
            self.fluid.reinit()
            self.fluid.reinit_phase_ratio(phase_ratio)

        # Присваиваем классу и всем флюидам
        if q_liq is not None:
            self.fluid.reinit_q_fluid(q_liq)

        # Обновляем тип корреляции если задана
        if hydr_corr_type:
            self.hydr_corr_type = hydr_corr_type

        # Обновляем коэффициент трения
        if friction_factor:
            self.friction_factor = friction_factor

        # Обновляем коэффициент на истинное содержание жидкости
        if grav_holdup_factor:
            self.grav_holdup_factor = grav_holdup_factor

        # Проверка корректности направления потока
        if flow_direction not in {1, -1}:
            raise exc.UniflocPyError(f"Неправильно задано направление потока - {flow_direction}")

        directions = np.empty(2)

        # Проверка корректности h_start
        if h_start.lower() == "bottom":
            h_mes = self.bottom_depth
            h1 = self.top_depth
            directions[1] = 1
            flow_direction *= -1

            if step_len is not None:
                # Необходимо сделать отрицательный шаг для правильной
                # генерации выводного массива шагов
                step_len *= -1

        elif h_start.lower() == "top":
            h_mes = self.top_depth
            h1 = self.bottom_depth
            directions[1] = -1
        else:
            raise exc.UniflocPyError(
                f"Некорректное значение для h_start = {h_start}. " f"Корректные значения - 'top' или 'bottom'",
                h_start,
            )

        directions[0] = flow_direction
        # Вывод значений в выбранных с шагом точках
        if steps is None:
            if step_len is not None:
                steps = np.arange(h_mes, h1, step_len)
                steps = np.append(steps, h1)
                if len(steps) <= 1:
                    steps = None
            else:
                steps = None

        # Подготовка аргумента для диаметра
        d = None if not isinstance(self.d, interp.interp1d) else self.d

        if hasattr(self, "d_cas_in") and hasattr(self, "d_tub_out"):
            d_cas_in = None if not isinstance(self.d_cas_in, interp.interp1d) else self.d_cas_in
            d_tub_out = None if not isinstance(self.d_tub_out, interp.interp1d) else self.d_tub_out
        else:
            d_tub_out, d_cas_in = None, None

        # Проверка возможности расчета с учетом теплопотерь
        if heat_balance:
            # Расчет раcпределений давления, температуры по глубине
            if not t_mes:
                t_mes = self.amb_temp_dist.calc_temp(h_mes).item()
            # Расчет теплопотерь возможен только в направлении снизу-вверх
            if h_start.lower() == "top":
                exc.UniflocPyError(
                    "Невозможно провести расчет с учетом теплопотерь сверху-вниз. "
                    "Поменяйте направление расчета на h_start = 'top', "
                    "либо отключите опцию теплопотерь heat_balance = False"
                )
        else:
            t_mes = self.amb_temp_dist.calc_temp(h_mes).item()

        # Расчет раcпределений давления, температуры по глубине
        result = self.pipe_object.integrate_pipe(
            p_mes,
            t_mes,
            h_mes,
            h1,
            self.trajectory,
            self.amb_temp_dist,
            int_method,
            directions,
            self.friction_factor,
            self.grav_holdup_factor,
            steps,
            d,
            d_tub_out,
            d_cas_in,
            heat_balance,
        )

        if len(result.y) > 0:
            self.distributions["depth"] = result.t
            self.distributions["p"] = result.y[0, :]
            self.distributions["t"] = result.y[1, :]

        status = result.status

        if status == 1 and steps is not None:
            self.distributions["depth"] = np.append(self.distributions["depth"], result.t_events)
            self.distributions["p"] = np.append(self.distributions["p"], result.y_events[0][0, 0])
            self.distributions["t"] = np.append(self.distributions["t"], result.y_events[0][0, 1])

        p = self.distributions["p"][-1]
        t = self.distributions["t"][-1]

        sorted_mask = self.distributions["depth"].argsort()
        for par in {"depth", "p", "t"}:
            self.distributions[par] = self.distributions[par][sorted_mask]

        # Если необходим вывод дополнительных параметров
        if extra_output:
            directions[1] = -1
            self.make_extra_distributions(
                self.distributions["p"],
                self.distributions["t"],
                self.distributions["depth"],
                self.trajectory,
                self.amb_temp_dist,
                directions,
                self.grav_holdup_factor,
                self.friction_factor,
                d,
                d_tub_out,
                d_cas_in,
                heat_balance,
            )

        return (
            p,
            t,
            status,
        )

    def calc_qliq(
        self,
        p_top: float,
        p_bottom: float,
        flow_direction: float,
        wct: Optional[float] = None,
        rp: Optional[float] = None,
        hydr_corr_type: Optional[str] = None,
        step_len: Optional[float] = None,
        int_method: str = "RK23",
        friction_factor: Optional[float] = None,
        grav_holdup_factor: Optional[float] = None,
        unstable_rate: bool = False,
        max_rate: Optional[float] = None,
    ) -> Tuple[float, int]:
        """
        Расчет дебита жидкости при известных давлениях на концах трубопровода
        с использованием многофазных корреляций

        :param p_top: давление на top_depth, Па абс.
        :param p_bottom: давление на bottom_depth, Па абс.
        :param flow_direction: направление движения потока, флаг
                               1 - к top_depth;
                               -1 - от top_depth
        :param wct: обводненность, доли ед.
        :param rp: газовый фактор, ст. м3 газа/ст. м3 нефти
        :param hydr_corr_type: тип гидравлической корреляции ("BeggsBrill", "Gray")
        :param step_len: шаг расчета кривой распределения давления по стволу, м
        :param int_method: метод интегрирования: ‘RK45’, ‘RK23’, ‘DOP853’, ‘Radau’, ‘BDF’, ‘LSODA’
        :param friction_factor: коэффициент поправки на трение
        :param grav_holdup_factor: коэффициент поправки на истинное содержание жидкости
        :param unstable_rate: флаг расчета дебита на нестабильной части VLP, по умолчанию False
        :param max_rate: предполагаемый максимально возможный дебит, м/с
         может сильно ускорить расчет
        :return: дебит жидкости, м3/с
        :return: последний статус расчета в трубе:
                 0 - успешен;
                 1 - во время расчета достигнуто минимальное давление;
                 -1 - завершен с ошибкой интегрирования

        Examples:
        --------
        >>> import unifloc.common.trajectory as traj
        >>> import unifloc.common.ambient_temperature_distribution as amb
        >>> import unifloc.pipe.pipeline as pipel
        >>> import unifloc.pvt.fluid_flow as fl
        >>> wct = 0.627
        >>>
        >>> well_trajectory = {"inclinometry": pd.DataFrame(columns=["MD", "TVD"],
        ...                    data=[[float(0), float(0)],[float(85), float(0)]])}
        >>> fluid_data = {"q_fluid": 16.229/86400,
        ...               "pvt_model_data":
        ...                   {"black_oil": {"gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.85,
        ...                                  "wct": wct, "phase_ratio": {"type": "GOR", "value": 69.9992},
        ...                    "rsb": {"value": 70, "p": 5000000, "t": 293.15},
        ...                    "muob": {"value": 0.5, "p": 5000000, "t": 293.15},
        ...                    "bob": {"value": 1.1, "p": 5000000, "t": 293.15}, }}}
        >>>
        >>> traject = traj.Trajectory(well_trajectory["inclinometry"])
        >>> fluid = fl.FluidFlow(**fluid_data)
        >>> ambient_temperature_data = {"MD": [0, 1000], "T": [303.15, 303.15]}
        >>> amb_temp = amb.AmbientTemperatureDistribution(ambient_temperature_data)
        >>> pipe = pipel.Pipeline(0, 85, d=0.01498, roughness=2.54 * 10 ** (-5),
        ...                       trajectory=traject, fluid=fluid,
        ...                       ambient_temperature_distribution=amb_temp,
        ...                       friction_factor=1, holdup_factor=1)
        >>>
        >>> q = pipe.calc_qliq(p_top=17.35524*101325, p_bottom=14.659864823951027*101325,
        ...                    flow_direction=-1, wct=wct, hydr_corr_type="BeggsBrill",
        ...                    max_rate=20/86400)
        """

        # Определение перегиба кривой VLP, если он есть
        if max_rate is None:
            max_rate = 0.011574074074074073
        try:
            solution = opt.brentq(
                self.__dp_dq,
                a=0.00011574074074074075,
                b=max_rate,
                args=(
                    p_top,
                    flow_direction,
                    wct,
                    rp,
                    None,
                    hydr_corr_type,
                    step_len,
                    int_method,
                    friction_factor,
                    grav_holdup_factor,
                ),
                xtol=0.00001,
            )
            if self._last_status != 0:
                raise ValueError
            if unstable_rate:
                lower_bracket = 0
                upper_bracket = solution
            else:
                lower_bracket = solution
                upper_bracket = max_rate
        except ValueError:
            lower_bracket = 0
            upper_bracket = max_rate
        try:
            q_liq = opt.brentq(
                self.__dp_error,
                a=lower_bracket,
                b=upper_bracket,
                args=(
                    p_top,
                    p_bottom,
                    flow_direction,
                    wct,
                    rp,
                    None,
                    hydr_corr_type,
                    step_len,
                    int_method,
                    friction_factor,
                    grav_holdup_factor,
                ),
                xtol=0.00001,
            )
        except ValueError:
            q_liq = opt.minimize_scalar(
                self.__dp_error_abs,
                args=(
                    p_top,
                    p_bottom,
                    flow_direction,
                    wct,
                    rp,
                    None,
                    hydr_corr_type,
                    step_len,
                    int_method,
                    friction_factor,
                    grav_holdup_factor,
                ),
                method="bounded",
                bounds=(lower_bracket, upper_bracket),
            )
            q_liq = q_liq.x
        return q_liq, self._last_status

    def __calc_p_bottom(
        self,
        q_liq,
        p_top,
        flow_direction,
        wct,
        rp,
        t_mes,
        hydr_corr_type,
        step_len,
        int_method,
        friction_factor,
        grav_holdup_factor,
    ):
        """
        Функция расчета p_bottom для текущего дебита
        """
        result = self.calc_pt(
            "top",
            p_top,
            flow_direction,
            q_liq,
            wct,
            rp,
            t_mes,
            hydr_corr_type,
            step_len,
            int_method,
            friction_factor,
            grav_holdup_factor,
        )

        self._last_status = result[2]
        return result[0]

    def __dp_dq(
        self,
        q_liq,
        p_top,
        flow_direction,
        wct,
        rp,
        t_mes,
        hydr_corr_type,
        step_len,
        int_method,
        friction_factor,
        grav_holdup_factor,
    ):
        """ "
        Функция расчета производной давления на bottom_depth от дебита
        """
        dpdq = misc.derivative(
            self.__calc_p_bottom,
            args=(
                p_top,
                flow_direction,
                wct,
                rp,
                t_mes,
                hydr_corr_type,
                step_len,
                int_method,
                friction_factor,
                grav_holdup_factor,
            ),
            x0=q_liq,
            dx=0.0000001,
        )
        return dpdq

    def __dp_error(
        self,
        q_liq,
        p_top,
        p_bottom,
        flow_direction,
        wct,
        rp,
        t_mes,
        hydr_corr_type,
        step_len,
        int_method,
        friction_factor,
        grav_holdup_factor,
    ):
        """
        Функция расчета ошибки в давлении на конце трубы
        """
        return p_bottom - self.__calc_p_bottom(
            q_liq,
            p_top,
            flow_direction,
            wct,
            rp,
            t_mes,
            hydr_corr_type,
            step_len,
            int_method,
            friction_factor,
            grav_holdup_factor,
        )

    def __dp_error_abs(
        self,
        q_liq,
        p_top,
        p_bottom,
        flow_direction,
        wct,
        rp,
        t_mes,
        hydr_corr_type,
        step_len,
        int_method,
        friction_factor,
        grav_holdup_factor,
    ):
        """
        Функция расчета модуля ошибки в давлении на конце трубы
        """
        return abs(
            self.__dp_error(
                q_liq,
                p_top,
                p_bottom,
                flow_direction,
                wct,
                rp,
                t_mes,
                hydr_corr_type,
                step_len,
                int_method,
                friction_factor,
                grav_holdup_factor,
            )
        )
