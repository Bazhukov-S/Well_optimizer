"""
Модуль для дополнительного расчетного функционала
"""

from copy import deepcopy
from typing import Optional, Tuple, Union

import pandas as pd
import scipy.optimize as opt

import ffmt.pvt.adapter as fl
import unifloc.common.trajectory as traj
import unifloc.common.ambient_temperature_distribution as amb
import unifloc.equipment.choke as ch
import unifloc.equipment.esp_electric_system as ees
import unifloc.equipment.natural_separation as nat
import unifloc.equipment.separator as sep
import unifloc.pipe.pipeline as pipe
import unifloc.tools.exceptions as exc


class PipePressSep:
    """
    Класс для расчета давления и доли газа на конце трубы при заданном дебите и давлении на другом
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
            :param pipe_data: словарь с исходными данными для создания трубы
            :param equipment_data: словарь с исходными данными для создания пакера и сепаратора
            :param well_trajectory_data: словарь с исходными данными для создания инклинометрии трубы
            :param ambient_temperature_data: словарь с распределением температуры породы по MD

            Со структурой словарей можно ознакомиться в примере ниже

            Examples:
            --------
        >>> # Пример на основе расчета потенциала ОПТ
        >>> from unifloc.tools.common_calculations import PipePressSep
        >>>
        >>> import pandas as pd
        >>>
        >>> fluid_data = {
        ...     "q_fluid": 100 / 86400,
        ...     "pvt_model_data": {
        ...         "black_oil": {
        ...             "gamma_gas": 0.7,
        ...             "gamma_wat": 1,
        ...             "gamma_oil": 0.8,
        ...             "wct": 0, "phase_ratio": {"type": "GOR", "value": 50},
        ...             "rsb": {"value": 50, "p": 10000000, "t": 303.15},
        ...             "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
        ...             "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
        ...             "table_model_data": None,
        ...             "use_table_model": False,
        ...         }
        ...     },
        ... }
        >>> d = {"MD": [0, 1000], "d": [0.06, 0.08]}
        >>> pipe_data = {"top_depth": 1400, "bottom_depth": 1800, "d": 0.146, "roughness": 0.0001}
        >>>
        >>> df = pd.DataFrame(columns=["MD", "TVD"], data=[[0, 0], [1400, 1200], [1800, 1542.85]])
        >>> well_trajectory_data = {"inclinometry": df}
        >>>
        >>> equipment_data = {"packer": True, "separator": 0.7}
        >>>
        >>> ambient_temperature_data = {"MD": [0, 1800], "T": [293.15, 303.15]}
        >>>
        >>> potential = PipePressSep(
        ...     fluid_data,
        ...     pipe_data,
        ...     equipment_data,
        ...     well_trajectory_data,
        ...     ambient_temperature_data,
        ... )
        >>>
        >>> q_liq = 150 / 86400
        >>> pin = 30 * 101325
        >>> h_esp = 1500
        >>> # Расчет забойного давления и доли газа
        >>> pwf, a_gas = potential.calc_pressure_and_sep(q_liq, pin, h_esp)
        >>>
        """

        # Инициализация класса для расчета флюида
        self.fluid = fl.FluidFlow(**fluid_data)

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

        # Инициализация класса для расчета инклинометрии трубы
        well_trajectory = traj.Trajectory(**well_trajectory_data)

        # Инициализация класса для расчета температуры окружающей породы
        self.amb_temp_dist = amb.AmbientTemperatureDistribution(ambient_temperature_data)

        # Инициализация класса для расчета трубы
        pipe_data.update(
            {
                "fluid": self.fluid,
                "trajectory": well_trajectory,
                "ambient_temperature_distribution": self.amb_temp_dist,
            }
        )
        self.pipe = pipe.Pipeline(**pipe_data)

        # Инициализация класса для расчета сепаратора
        if equipment_data.get("separator") is not None:
            self.separator = sep.Separator(self.pipe.top_depth, **equipment_data["separator"])
        else:
            self.separator = None

        # Инициализация класса для расчета естественной сепарации
        if equipment_data.get("packer") is not None and not equipment_data["packer"]:
            self.natural_sep = nat.NaturalSeparation(self.pipe.top_depth)
        else:
            self.natural_sep = None

    def calc_pressure_and_sep(
        self,
        q_liq: float,
        p_top: float,
        t_top: Optional[float] = None,
        h_top: Optional[float] = None,
        h_bottom: Optional[float] = None,
        hydr_corr_type: str = "BeggsBrill",
        grav_holdup_factor: float = 1,
    ) -> Tuple[float, float]:
        """
        Функция для расчета давления на нижнем конце трубы и доли газа после сепарации на верхнем
        конце.

        Parameters
        ----------
        :param q_liq: дебит жидкости трубы, м3/с
        :param p_top: давление на верхнем конце трубы, Па изб.
        :param t_top: температура на глубине верхнего конца трубы, K
        :param h_top: измеренная глубина верхнего конца трубы, м
        :param h_bottom: измеренная глубина нижнего конца трубы, м
        :param hydr_corr_type: тип гидравлической корреляции
        :param grav_holdup_factor: коэффициент адаптации КРД на гравитацию/holdup, д.ед

        :return: давление на нижнем конце трубы, Па изб.
        :return: доля газа на верхнем конце трубы после сепарации, д.ед.
        """

        self.fluid.reinit()
        self.fluid.q_fluid = q_liq
        self.fluid.reinit_q_fluid(q_liq)
        self.pipe.fluid = deepcopy(self.fluid)

        if not h_top:
            h_top = self.pipe.top_depth

        if h_bottom:
            self.pipe.bottom_depth = h_bottom

        # Если не задана температура на приеме, зададим ее равной пластовой
        if not t_top:
            t_top = float(self.amb_temp_dist.calc_temp(h_top))

        # Расчет свойств флюида до сепарации
        self.fluid.calc_flow(p_top, t_top)

        # Расчет естественной сепарации
        if self.natural_sep:
            k_sep_nat = 0.5
        else:
            k_sep_nat = 0

        # Расчет общей сепарацию
        if self.separator:
            k_sep_gen = self.separator.calc_general_separation(
                k_gas_sep_nat=k_sep_nat,
                gf=self.fluid.gf,
                q_liq=q_liq,
                freq=50
            )
        else:
            k_sep_gen = k_sep_nat

        # Расчет модификации флюида
        self.fluid.modify(p_top, t_top, k_sep_gen)

        # Расчет свойств флюида после сепарации
        self.fluid.calc_flow(p_top, t_top)

        # Доля газа на приеме насоса после сепарации
        gas_fraction = self.fluid.gf

        # Проверка, что глубина спуска насоса не превышает глубину ВДП
        if h_top < self.pipe.bottom_depth:
            # Расчет забойного давления
            self.pipe.top_depth = h_top
            p_bottom, _, _ = self.pipe.calc_pt(
                "top",
                p_top,
                1,
                q_liq,
                t_mes=t_top,
                hydr_corr_type=hydr_corr_type,
                grav_holdup_factor=grav_holdup_factor,
            )
        else:
            raise exc.UniflocPyError(
                f"Заданная глубина {h_top} м верхнего конца трубы "
                f"превышает глубину нижнего конца {self.pipe.bottom_depth} м."
                f" Задайте другую глубину"
            )

        return p_bottom, gas_fraction


def __adapt_func_choke(
    c_choke: float,
    choke: ch.Choke,
    p_out: float,
    p_in: float,
    t_out: float,
    q_liq: float,
    wct: float,
) -> float:
    """
    Функция адаптации штуцера

    :param c_choke: адаптационный коэффициент штуцера
    :param p_out: давление на выходе из штуцера, Па абс.
    :param choke: объект штуцера
    :param p_in: давление на входе в штуцер, Па абс.
    :param t_out: температура на выходе из штуцера, К
    :param q_liq: дебит жидкости, м3/с
    :param wct: обводненность, д. ед.

    :return ошибка в расчетном p_wh, Па
    """
    p_in_calc, *_ = choke.calc_pt(p_out, t_out, 1, q_liq, wct, None, c_choke)
    return p_in - p_in_calc


def __adapt_func_choke_abs(
    c_choke: float,
    choke: ch.Choke,
    p_out: float,
    p_in: float,
    t_out: float,
    q_liq: float,
    wct: float,
) -> float:
    """
    Функция адаптации штуцера

    :param c_choke: адаптационный коэффициент штуцера
    :param p_out: давление на выходе из штуцера, Па абс.
    :param choke: объект штуцера
    :param p_in: давление на входе в штуцер, Па абс.
    :param t_out: температура на выходе из штуцера, К
    :param q_liq: дебит жидкости, м3/с
    :param wct: обводненность, д. ед.

    :return модуль ошибки в расчетном p_wh, Па
    """
    return abs(__adapt_func_choke(c_choke, choke, p_out, p_in, t_out, q_liq, wct))


def adapt_choke(
    choke: ch.Choke, p_out: float, p_in: float, t_out: float, q_liq: float, wct: float
) -> Union[float, dict]:
    """
    Функция, адаптирующая штуцер и возвращающая адаптационный
    коэффициент штуцера

    :param choke: объект штуцера
    :param p_out: давление на выходе из штуцера, Па абс.
    :param p_in: давление на входе в штуцер, Па абс.
    :param t_out: температура на выходе из штуцера, К
    :param q_liq: дебит жидкости, м3/с
    :param wct: обводненность, д. ед.
    :return: адаптационный коэффициент штуцера
    """
    try:
        c_ch = opt.brentq(__adapt_func_choke, a=0.3, b=5, args=(choke, p_out, p_in, t_out, q_liq, wct), xtol=0.01)
    except ValueError:
        try:
            c_ch = opt.brentq(__adapt_func_choke, a=0.01, b=5, args=(choke, p_out, p_in, t_out, q_liq, wct), xtol=0.01)
        except ValueError:
            c_ch = opt.minimize_scalar(
                __adapt_func_choke_abs,
                method="bounded",
                bounds=(0.3, 5),
                args=(choke, p_out, p_in, t_out, q_liq, wct),
            )
            if c_ch.fun > 10000:
                c_ch = {"const": p_in - p_out}
            else:
                c_ch = c_ch.x

    if isinstance(c_ch, float):
        p_in_calc, *_ = choke.calc_pt(p_out, t_out, 1, q_liq, wct, None, c_ch)
        if abs(p_in - p_in_calc) / p_in > 0.05:
            c_ch = {"const": p_in - p_out}

    return c_ch


def _adapt_pump_power(
    c_pump_power: float,
    pump_power: float,
    esp_electric_system: ees.EspElectricSystem,
    motor_i_fact: float,
    fluid_power: float,
    freq_shaft: float,
    t_cable: float,
) -> float:
    """
    Функция оптимизации мощности насоса при адаптации электротехнического расчета

    Parameters
    ----------
    :param c_pump_power: адаптационный коэффициент для мощности насоса
    :param pump_power: электрическая мощность насоса, Вт
    :param esp_electric_system: объект электрической системы УЭЦН
    :param motor_i_fact: фактическая сила тока, А
    :param fluid_power: гидравлическая мощность, Вт
    :param freq_shaft: текущая частота вращения вала, Гц
    :param t_cable: температура на глубине спуска ПЭД, К

    Returns
    -------
    Абсолютное отклонение по силе тока, А
    """
    res = esp_electric_system.calc_electric_esp_system(
        pump_power=pump_power,
        fluid_power=fluid_power,
        freq_shaft=freq_shaft,
        t_cable=t_cable,
        c_pump_power=c_pump_power,
    )
    return abs(res["motor_i"] - motor_i_fact)


def _adapt_load_i(
    c_load_i: float,
    c_pump_power: float,
    pump_power: float,
    esp_electric_system: ees.EspElectricSystem,
    load_fact: float,
    fluid_power: float,
    freq_shaft: float,
    t_cable: float,
):
    """
    Функция оптимизации загрузки по току при адаптации электротехнического расчета

    Parameters
    ----------
    :param c_load_i: адаптационный коэффициент для загрузки по току
    :param c_pump_power: адаптационный коэффициент для мощности насоса
    :param pump_power: электрическая мощность насоса, Вт
    :param esp_electric_system: объект электрической системы УЭЦН
    :param load_fact: фактическая загрузка ПЭД по току, д.ед.
    :param fluid_power: гидравлическая мощность, Вт
    :param freq_shaft: текущая частота вращения вала, Гц
    :param t_cable: температура на глубине спуска ПЭД, К

    Returns
    -------
    Абсолютное отклонение по загрузке, д.ед
    """
    res = esp_electric_system.calc_electric_esp_system(
        pump_power, fluid_power, freq_shaft, t_cable, c_pump_power=c_pump_power, c_load_i=c_load_i
    )
    return abs(res["load"] - load_fact)


def _adapt_transform_voltage(
    c_transform_voltage: float,
    c_pump_power: float,
    c_load_i: float,
    pump_power: float,
    esp_electric_system: ees.EspElectricSystem,
    transform_voltage_fact: float,
    fluid_power: float,
    freq_shaft: float,
    t_cable: float,
):
    """
    Функция оптимизации напряжения на трансформаторе при адаптации электротехнического расчета

    Parameters
    ----------
    :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
    :param c_pump_power: адаптационный коэффициент для мощности насоса
    :param c_load_i: адаптационный коэффициент для загрузки по току
    :param pump_power: электрическая мощность насоса, Вт
    :param esp_electric_system: объект электрической системы УЭЦН
    :param transform_voltage_fact: фактическое напряжение на отпайке трансформатора, В
    :param fluid_power: гидравлическая мощность, Вт
    :param freq_shaft: текущая частота вращения вала, Гц
    :param t_cable: температура на глубине спуска ПЭД, К

    Returns
    -------
    Абсолютное отклонение по загрузке, д.ед
    """
    res = esp_electric_system.calc_electric_esp_system(
        pump_power,
        fluid_power,
        freq_shaft,
        t_cable,
        c_pump_power=c_pump_power,
        c_load_i=c_load_i,
        c_transform_voltage=c_transform_voltage,
    )
    div = abs(res["transform_voltage"] - transform_voltage_fact)
    return div


def _adapt_cs_power(
    c_cs_power: float,
    c_pump_power: float,
    c_load_i: float,
    c_transform_power: float,
    pump_power: float,
    esp_electric_system: ees.EspElectricSystem,
    cs_power_fact: float,
    fluid_power: float,
    freq_shaft: float,
    t_cable: float,
):
    """
    Функция оптимизации напряжения на трансформаторе при адаптации электротехнического расчета

    Parameters
    ----------
    :param c_cs_power: адаптационный коэффицицент для мощности на СУ
    :param c_pump_power: адаптационный коэффициент для мощности насоса
    :param c_load_i: адаптационный коэффициент для загрузки по току
    :param c_transform_power: адаптационный коэффицицент для мощности на трансформаторе
    :param pump_power: электрическая мощность насоса, Вт
    :param esp_electric_system: объект электрической системы УЭЦН
    :param cs_power_fact: фактическая активная мощность на станции управления, Вт
    :param fluid_power: гидравлическая мощность, Вт
    :param freq_shaft: текущая частота вращения вала, Гц
    :param t_cable: температура на глубине спуска ПЭД, К

    Returns
    -------
    Абсолютное отклонение по загрузке, д.ед
    """
    res = esp_electric_system.calc_electric_esp_system(
        pump_power,
        fluid_power,
        freq_shaft,
        t_cable,
        c_pump_power=c_pump_power,
        c_load_i=c_load_i,
        c_cs_power=c_cs_power,
        c_transform_voltage=c_transform_power,
    )

    div = abs(res["cs_power"] - cs_power_fact)
    return div


def adapt_elsys(
    esp_electric_system: ees.EspElectricSystem,
    pump_power: float,
    fluid_power: float,
    freq_shaft: float,
    t_cable: float,
    cosf_fact: Optional[float] = None,
    motor_i_fact: Optional[float] = None,
    load_fact: Optional[float] = None,
    transform_voltage_fact: Optional[float] = None,
    cs_power_fact: Optional[float] = None,
) -> dict:
    """
    Функция расчета адаптационных коэффициентов для электротехнического расчета УЭЦН
    Расчет адаптационных коэффициентов происходит поочередно - снизу-вверх по компоновке УЭЦН

    Parameters
    ----------
    :param esp_electric_system: объект электрической системы УЭЦН
    :param pump_power: электрическая мощность насоса, Вт
    :param fluid_power: гидравлическая мощность, Вт
    :param freq_shaft: текущая частота вращения вала, Гц
    :param t_cable: температура на глубине спуска ПЭД, К
    :param cosf_fact: фактический косинус мощности, д.ед. - optional, пока не используется
    :param motor_i_fact: фактическая сила тока, А - optional
    :param load_fact: фактическая загрузка ПЭД по току, д.ед. - optional
    :param transform_voltage_fact: фактическое напряжение на отпайке трансформатора, В - optional
    :param cs_power_fact: фактическая активная мощность на станции управления, Вт - optional

    Returns
    -------
    Словарь с адаптационными коэффициентами:
        адаптационный коэффициент для мощности насоса
        адаптационный коэффициент для загрузки по току
        адаптационный коэффицицент для напряжения на трансформаторе
        адаптационный коэффицицент для мощности на СУ
    """

    c_pump_power = 1
    c_load_i = 1
    c_transform_voltage = 1
    c_cs_power = 1

    if pump_power != 0 and fluid_power != 0:
        # Пересчет номинальной мощности ПЭД, сепаратора и протектора в зависимости от текущей частоты
        esp_electric_system._calc_equip_power(freq_shaft)

        if motor_i_fact is not None:
            c_pump_power = opt.minimize_scalar(
                _adapt_pump_power,
                args=(pump_power, esp_electric_system, motor_i_fact, fluid_power, freq_shaft, t_cable),
                bounds=(0, 100),
                method="bounded",
                options={"xatol": 0.01}
            ).x

        if load_fact is not None:
            c_load_i = opt.minimize_scalar(
                _adapt_load_i,
                args=(c_pump_power, pump_power, esp_electric_system, load_fact, fluid_power, freq_shaft, t_cable),
                bounds=(0, 100),
                method="bounded",
                options={"xatol": 0.01}
            ).x

        if transform_voltage_fact:
            c_transform_voltage = opt.minimize_scalar(
                _adapt_transform_voltage,
                args=(
                    c_pump_power,
                    c_load_i,
                    pump_power,
                    esp_electric_system,
                    transform_voltage_fact,
                    fluid_power,
                    freq_shaft,
                    t_cable,
                ),
                bounds=(0, 100),
                method="bounded",
                options={"xatol": 0.01}
            ).x

        if cs_power_fact:
            c_cs_power = opt.minimize_scalar(
                _adapt_cs_power,
                args=(
                    c_pump_power,
                    c_load_i,
                    c_transform_voltage,
                    pump_power,
                    esp_electric_system,
                    cs_power_fact,
                    fluid_power,
                    freq_shaft,
                    t_cable,
                ),
                bounds=(0, 100),
                method="bounded",
                options={"xatol": 0.01}
            ).x

    return {
        "c_pump_power": c_pump_power,
        "c_cs_power": c_cs_power,
        "c_transform_voltage": c_transform_voltage,
        "c_load_i": c_load_i,
    }
