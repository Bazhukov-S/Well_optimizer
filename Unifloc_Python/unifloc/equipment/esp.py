"""
Модуль, описывающий класс по работе с электроцентробежным насосом
"""
from copy import deepcopy
from functools import cached_property
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.integrate as integr
import scipy.interpolate as interp

import ffmt.pvt.adapter as fl
import unifloc.equipment.equipment as eq
import unifloc.service._constants as const
import unifloc.service._tools as tls
import unifloc.tools.exceptions as exc


class Esp(eq.Equipment):
    """
    Класс, описывающий работу электроцентробежного насоса
    """

    def __init__(
        self,
        h_mes: float,
        stages: int,
        esp_data: Union[pd.Series, dict],
        fluid: fl.FluidFlow,
        viscosity_correction: bool = True,
        gas_correction: bool = True,
    ):
        """
        Parameters
        ----------
        :param h_mes: глубина, на которой установлен насос, м - float
        :param stages: количество ступеней насоса - integer
        :param esp_data: паспортные данные ЭЦН - pd.Series или dict
        :param viscosity_correction: флаг учета поправки на вязкость - boolean
        :param gas_correction: флаг учета поправки на газ - boolean
        :param fluid: объект флюида

        Examples:
        --------
        >>> from unifloc.pvt.fluid_flow import FluidFlow
        >>> from unifloc.equipment.esp import Esp
        >>> # Инициализация исходных данных
        >>> # Считывание из json-базы паспортных характеристик насоса
        >>> esp_data = {"ID": 99999, "source": "legacy", "manufacturer": "Reda",
        ...             "name": "DN500", "stages_max": 400, "rate_nom_sm3day": 30,
        ...             "rate_opt_min_sm3day": 20,
        ...             "rate_opt_max_sm3day": 40, "rate_max_sm3day": 66,
        ...             "slip_nom_rpm": 3000, "freq_Hz": 50, "eff_max": 0.4,
        ...             "height_stage_m": 0.035, "Series": 4, "d_od_mm": 86,
        ...             "d_cas_min_mm": 112, "d_shaft_mm": 17, "area_shaft_mm2": 227,
        ...             "power_limit_shaft_kW": 72, "power_limit_shaft_high_kW": 120,
        ...             "power_limit_shaft_max_kW": 150, "pressure_limit_housing_atma": 390,
        ...             "d_motor_od_mm": 95,
        ...             "rate_points": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 66],
        ...             "head_points": [4.88, 4.73, 4.66, 4.61, 4.52, 4.35, 4.1, 3.74,
        ...                             3.2800000000000002, 2.73, 2.11, 1.45, 0.77, 0],
        ...             "power_points": [0.02, 0.022, 0.025, 0.027, 0.03, 0.032, 0.035,
        ...                              0.038, 0.041, 0.043000000000000003,
        ...                              0.046, 0.049, 0.052000000000000005, 0.055],
        ...             "eff_points": [0, 0.12, 0.21, 0.29, 0.35000000000000003, 0.38,
        ...                            0.4, 0.39, 0.37, 0.32, 0.26, 0.19, 0.1, 0]
        ...             }
        >>> # Инициализация свойств флюида
        >>> q_fluid = 20
        >>> wct = 0.1
        >>> fluid_data = {
        ...     "q_fluid": q_fluid / 86400, "pvt_model_data": {
        ...         "black_oil": {
        ...             "gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.95,
        ...             "wct": wct, "phase_ratio": {"type": "GOR", "value": 50}
        ...         }
        ...     }
        ... }
        >>> # Инициализация модели флюида
        >>> fluid1 = FluidFlow(**fluid_data)
        >>> # Инициализация класса ЭЦН
        >>> h_mes = 1600
        >>> stages = 100
        >>> viscosity_correction = False
        >>> gas_correction = False
        >>> esp_calculation = Esp(h_mes, stages, esp_data,
        ...                       fluid1, viscosity_correction, gas_correction,
        ...                       )
        >>> # Необходимо подать давление и температуру на приеме/выкиде
        >>> # насоса в зависимости от направления расчета
        >>> # (например, если расчет идет от устья к забою, то подаем параметр direction_to="in",
        >>> # температуру и давление
        >>> # на выкиде  и в результате получаем давление на приеме насоса)
        >>> p = 120 * 101325
        >>> t = 30 + 273.15
        >>> freq = 50
        >>> direction_to = "in"
        >>> # Поправка на напор (опционально, по умолчанию None)
        >>> head_factor = 1
        >>> # Коэффициент проскальзывания (опционально, по умолчанию 0.97222)
        >>> slippage_factor = 0.97222
        >>> # Расчет давления и температуры на выкиде или на приеме
        >>> # (в зависимости от направления расчета)
        >>> pt_calc = esp_calculation.calc_pt(
        ...     p,
        ...     t,
        ...     freq,
        ...     q_fluid/86400,
        ...     wct,
        ...     direction_to=direction_to,
        ...     head_factor=head_factor,
        ...     slippage_factor=slippage_factor
        ... )
        """
        # TODO: v 1.5.0: уточнить использование стольких атрибутов
        super().__init__(h_mes)
        self.__stages = stages
        self.__esp_data = None

        self.freq_nom = None
        self.__hr_func = None
        self.__pr_func = None
        self.__er_func = None
        self.rate_points = None
        self.head_points = None
        self.power_points = None
        self.eff_points = None
        self.esp_data = esp_data

        self.fluid = fluid
        self.viscosity_correction = viscosity_correction
        self.gas_correction = gas_correction
        self.slippage_factor = 0.97222

        self.power_fluid = 0
        self.power_esp = 0
        self.efficiency = 0
        self.dp = 0
        self.head = 0
        self.q_mix = 0

        self.rate_points_corr = None
        self.head_points_corr = None
        self.power_points_corr = None
        self.eff_points_corr = None
        self.nom_eff = None
        self.nom_rate = None
        self.nom_head = None
        self.nom_power = None
        self.distributions = dict()

    @property
    def stages(self):
        """
        Число ступеней
        """
        return self.__stages

    @stages.setter
    def stages(self, value):
        """
        Число ступеней
        """
        self.__stages = value

    @property
    def esp_data(self):
        """
        Номинальные данные насоса
        """
        return self.__esp_data

    @esp_data.setter
    def esp_data(self, value):
        """
        Задание номинальных свойств насоса
        """
        if isinstance(value, pd.Series):
            self.__esp_data = value
        elif isinstance(value, dict):
            self.__esp_data = pd.Series(value)
        else:
            raise exc.UniflocPyError(f"Неподдерживаемый тип данных для esp_data - {type(value)}")

        self.freq_nom = self.__esp_data.freq_Hz if "freq_Hz" in self.__esp_data else 50

        self.rate_points = np.array(self.__esp_data.rate_points) if "rate_points" in self.__esp_data else None
        self.head_points = np.array(self.__esp_data.head_points) if "head_points" in self.__esp_data else None
        self.power_points = np.array(self.__esp_data.power_points) if "power_points" in self.__esp_data else None
        self.eff_points = np.array(self.__esp_data.eff_points) if "eff_points" in self.__esp_data else None

        if (
            self.rate_points is not None
            and self.head_points is not None
            and self.power_points is not None
            and self.eff_points is not None
        ):
            self.__hr_func_verification()
        else:
            raise exc.UniflocPyError("Не заданы точки НРХ. Расчет ЭЦН невозможен")

        self.__renew_pump_func()

    @cached_property
    def hr_func(self):
        """
        Вычисляемый атрибут - функция ЭЦН Дебит-Напор

        :return: функция ЭЦН Дебит-Напор
        -------

        """
        self.__hr_func = interp.interp1d(
            self.rate_points_corr,
            self.head_points_corr,
            kind="quadratic",
            fill_value=0,
            bounds_error=False,
        )

        return self.__hr_func

    @cached_property
    def pr_func(self):
        """
        Вычисляемый атрибут - функция ЭЦН Дебит-Мощность

        :return: функция ЭЦН Дебит-Мощность
        -------

        """
        self.__pr_func = interp.interp1d(
            self.rate_points_corr,
            self.power_points_corr,
            kind="quadratic",
            fill_value="extrapolate",
        )
        return self.__pr_func

    @cached_property
    def er_func(self):
        """
        Вычисляемый атрибут - функция ЭЦН Дебит-КПД

        :return: функция ЭЦН Дебит-КПД
        -------

        """
        self.__er_func = interp.interp1d(
            self.rate_points_corr,
            self.eff_points_corr,
            kind="quadratic",
            fill_value=0,
            bounds_error=False,
        )
        return self.__er_func

    @cached_property
    def r_k_visccorr_func(self):
        """
        Интерполятор по кривой поправки дебита на вязкость

        :return: корректировочный коэффициент дебита на вязкость
        -------

        """
        r_k_visccorr_func = interp.interp1d(
            const.NU_SSU_POINTS,
            const.Q_COEFF_POINTS,
            kind="cubic",
            fill_value="extrapolate",
        )
        return r_k_visccorr_func

    @cached_property
    def h_k_visccorr_func(self):
        """
        Интерполятор по кривой поправки напора на вязкость

        :return: корректировочный коэффициент напора на вязкость
        -------

        """
        h_k_visccorr_func = interp.interp1d(
            const.NU_SSU_POINTS,
            const.HEAD_COEFFS_POINTS,
            kind="linear",
            fill_value="extrapolate",
        )
        return h_k_visccorr_func

    @cached_property
    def e_k_visccorr_func(self):
        """
        Интерполятор по кривой поправки КПД на вязкость

        :return: корректировочный коэффициент КПД на вязкость
        -------

        """
        e_k_visccorr_func = interp.interp1d(
            const.NU_SSU_POINTS,
            const.EFF_COEFF_POINTS,
            kind="linear",
            fill_value="extrapolate",
        )
        return e_k_visccorr_func

    @cached_property
    def p_k_visccorr_func(self):
        """
        Интерполятор по кривой поправки мощности на вязкость

        :return: корректировочный коэффициент мощности на вязкость
        -------

        """
        p_k_visccorr_func = interp.interp1d(
            const.NU_SSU_POINTS,
            const.POWER_COEFF_POINTS,
            kind="linear",
            fill_value="extrapolate",
        )
        return p_k_visccorr_func

    @cached_property
    def gascorr_func(self):
        """
        Интерполятор по кривой поправки НРХ на свободный газ

        Parameters
        ----------
        :return: корректировочный коэффициент НРХ под влияниянием свободного газа
        -------
        """
        gascorr_func = interp.interp2d(
            const.GAS_COEF_CORR_X,
            const.GAS_COEF_CORR_Y,
            const.GAS_COEF_CORR,
        )
        return gascorr_func

    def _gas_corr(self, freq_ratio: float) -> float:
        """
        Расчет деградации напорно-расходной характеристики под влиянием свободного газа

        Parameters
        ----------
        :param freq_ratio: отношение фактической частоты к номинальной, д.ед

        :return: коэффициент деградации насоса из-за влияния газа, д.ед
        References
        ----------
        SPE 206468
        """
        # Коэффициент подачи
        k = self.fluid.ql / (self.nom_rate[0] * freq_ratio)

        gas_corr = self.gascorr_func(self.fluid.gf, k).item()
        # Проверка на отрицательный коэффициент деградации(зона k > 0.6-1.0)
        if gas_corr < 0:
            gas_corr = 0
        return gas_corr

    def __renew_pump_func(self):
        """
        Обнуление интерполяционных полиномов
        """
        variables = vars(self)

        for key in ["hr_func", "pr_func", "er_func"]:
            if key in variables:
                delattr(self, key)

    def __hr_func_verification(self):
        """
        Проверка на наличие нулевого напора в кривой НРХ

        Returns
        -------
        Обновляет точки напорно-расходной характеристики
        """

        # Добавим точки, в которых напор равен 0, если таких нет
        if self.head_points[-1] != 0:
            rate_fun = interp.interp1d(
                self.__esp_data.head_points,
                self.__esp_data.rate_points,
                kind="linear",
                fill_value="extrapolate",
            )
            self.rate_points = np.append(self.rate_points, rate_fun(0))
            self.head_points = np.append(self.head_points, 0)

            # Обновим точку по КПД, в которой напор равен 0
            eff_fun = interp.interp1d(
                self.__esp_data.rate_points,
                self.__esp_data.eff_points,
                kind="linear",
                fill_value="extrapolate",
            )
            self.eff_points = np.append(self.eff_points, eff_fun(self.rate_points[-1]))

            # Обновим точку по мощности, в которой напор равен 0
            power_fun = interp.interp1d(
                self.__esp_data.rate_points,
                self.__esp_data.power_points,
                kind="linear",
                fill_value="extrapolate",
            )
            self.power_points = np.append(self.power_points, power_fun(self.rate_points[-1]))

        if self.rate_points[0] != 0:
            head_fun = interp.interp1d(
                self.__esp_data.rate_points,
                self.__esp_data.head_points,
                kind="linear",
                fill_value="extrapolate",
            )
            self.head_points = np.insert(self.head_points, 0, head_fun(0))
            self.rate_points = np.insert(self.rate_points, 0, 0)

            # Обновим точку по КПД, в которой дебит равен 0
            self.eff_points = np.insert(self.eff_points, 0, 0)

            # Обновим точку по мощности, в которой дебит равен 0
            self.power_points = np.insert(self.power_points, 0, 0)

        self.rate_points = self.rate_points / 86400

    def __hr_nom_point(self):
        """
        Функция для определения и записи номинальной точки НРХ в атрибуты

        Returns
        -------
        :param Номинальный расход, м3/сут
        :param Номинальный напор, м
        :param Номинальная мощность, кВт
        :param Номинальный(максимальный) КПД, д.ед.
        """
        # Номинальный (максимальный) КПД
        self.nom_eff = max(self.eff_points)
        # Номинальный расход по НРХ
        self.nom_rate = self.rate_points[np.where(self.eff_points == self.nom_eff)]
        # Номинальный напор по НРХ
        self.nom_head = self.head_points[np.where(self.eff_points == self.nom_eff)]
        # Номинальная мощность по НРХ
        self.nom_power = self.power_points[np.where(self.eff_points == self.nom_eff)]

    def __correct_viscosity(self, nu_mix):
        """
        Функция для коррекции характеристик на вязкость флюида

        Parameters
        ----------
        :param nu_mix : кинематическая вязкость флюида, cCт

        Returns
        -------
        Обновляет интерполяционные полиномы для напора и кпд
        """

        # Конвертация по Такасу
        nu_mix_ssu = 2.273 * (nu_mix + (nu_mix * nu_mix + 158.4) ** 0.5)

        # Определяем корректировочные коэффициенты по таблице в зависимости от вязкости
        corr_visc_q = self.r_k_visccorr_func(nu_mix_ssu)
        corr_visc_h = self.h_k_visccorr_func(nu_mix_ssu)
        corr_visc_eff = self.e_k_visccorr_func(nu_mix_ssu)

        # Скорректированная НРХ
        q_points_corr = self.rate_points_corr * corr_visc_q
        head_points_corr = self.head_points_corr * corr_visc_h
        eff_points_corr = self.eff_points_corr * corr_visc_eff

        # Меняем интерполяционный полином, будем работать с новой НРХ
        self.hr_func = interp.interp1d(
            q_points_corr,
            head_points_corr,
            kind="linear",
            fill_value=0,
            bounds_error=False,
        )

        self.er_func = interp.interp1d(
            q_points_corr,
            eff_points_corr,
            kind="linear",
            fill_value=0,
            bounds_error=False,
        )

    def __integr_func_pressure(self, stage, y, head_factor, temp_grad):
        """
        Функция для интегрирования солвером по ступеням

        Parameters
        ----------
        :param stage: ступень текущая
        :param y: массив игреков - функции - y=f(stages), которые интегрируем
        :param head_factor: коэффициент поправки на напор
        :param temp_grad: температурный градиент, К/ступень

        Returns
        -------
        градиент напора, давления, полезной мощности, затраченной мощности, температуры
        """
        # Условие прекращения интегрирования
        if np.isnan(y[1]) or y[1] <= 0:
            return False

        # Зададим ограничение по максимальной температуре флюида в насосе (377.11 С -> 644.26 K)
        if y[4] > 644.26:
            y[4] = 644.26
        # Зададим ограничение по минимальной температуре флюида в насосе (5 С -> 268.15 K)
        elif y[4] < 268.15:
            y[4] = 268.15

        # Обновим свойства флюида
        self.fluid.calc_flow(y[1].item(), y[4].item())

        # Скорректируем НРХ на вязкость, если необходимо
        if self.viscosity_correction:
            # Переводим динамическую вязкость в кинематическую
            nu_mix = self.fluid.mum / (self.fluid.rm / 1000)
            self.__correct_viscosity(nu_mix)

        # Напор с учетом частоты, числа ступеней, поправки на напор и дебита смеси
        head_grad = self.hr_func(self.fluid.qm) * head_factor

        # Градиент давления по ступеням
        dp_grad = self.fluid.rm * head_grad * 9.80665

        # Градиент мощности флюида по ступеням
        power_fl_grad = self.fluid.qm * dp_grad

        # Градиент электрической мощности по ступеням
        power_esp_grad = power_fl_grad / max(self.er_func(self.fluid.qm), 0.01)

        # КПД по ступеням
        if power_esp_grad == 0:
            eff_grad = 0
        else:
            eff_grad = power_fl_grad / power_esp_grad

        if temp_grad is None:
            # Нагрев флюида в насосе по ступеням
            if eff_grad == 0:
                temp_grad = 0
            else:
                temp_grad = 9.80665 * head_grad * (1 - eff_grad) / (self.fluid.heat_capacity_mixture * eff_grad)

        return head_grad, dp_grad, power_fl_grad, power_esp_grad, temp_grad

    @staticmethod
    def __lower_limit(stage, y, *args):
        """
        Проверка на минимальное значение давления при интегрировании
        """

        return y[1] - 90000

    def __integrate_esp(self, p, t, head_factor, int_method, direction_to, temp_grad):
        """
        Функция для интегрирования давления по длине насоса

        Parameters
        ----------
        :param p: начальное давление для интегрирования, Па
        :param t: начальная температура, К
        :param head_factor: коэффициент поправки на напор
        :param int_method: метод интегрирования
        :param temp_grad: температурный градиент, К/ступень

        Returns
        -------
        распределение давления по ступеням
        """
        # Определение множителя для направления расчета
        if direction_to == "dis":
            stage0 = 0
            stage1 = self.stages
        else:
            stage0 = self.stages
            stage1 = 0

        # TODO: v1.5.0: доделать случай для малого числа ступеней
        self.__lower_limit.terminal = True
        dp_esp_integration = integr.solve_ivp(
            self.__integr_func_pressure,
            t_span=(stage0, stage1),
            y0=[0, p, 0, 0, t],
            method=int_method,
            args=(head_factor, temp_grad),
            rtol=0.001,
            atol=0.001,
            events=self.__lower_limit,
        )

        return dp_esp_integration.t, dp_esp_integration.y, dp_esp_integration.status

    def calc_pt(
        self,
        p: float,
        t: float,
        freq: float,
        q_liq: Optional[float] = None,
        wct: Optional[float] = None,
        phase_ratio_value: Optional[float] = None,
        direction_to: str = "dis",
        head_factor: Optional[float] = None,
        slippage_factor: float = 0.97222,
        int_method: str = "RK23",
        extra_output: bool = False,
        temp_grad: Optional[float] = None,
    ) -> tuple:
        """
        Функция для расчета давления и температуры на приеме/выкиде ЭЦН

        Parameters

        :param p: давление на приеме либо на выкиде (зависит от направления расчета), Па
        :param t: температура на приеме либо на выкиде (зависит от направления расчета), К
        :param freq: частота вала без учета проскальзывания, Гц
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param phase_ratio_value: соотношение газа к флюиду для выбранного типа фазового соотношения,
            ст. м3 газа/ст. м3 флюид
        :param direction_to: направление расчета:
                             -dis - от приема к выкиду (по умолчанию);
                             -in - от выкида к приему.
        :param head_factor: коэффициент поправки на напор, д.ед
        :param slippage_factor: коэффициент проскальзывания ЭЦН, д.ед. По умолчанию = 0.97222
        :param int_method: метод интегрирования: ‘RK45’, ‘RK23’, ‘DOP853’, ‘Radau’, ‘BDF’, ‘LSODA’
        :param extra_output: флаг для дополнительного расчета распределений и сохранения в атрибуты
        :param temp_grad: температурный градиент:
                                - None - расчет нагрева флюида в насосе включен (по умолчанию);
                                - 0 - расчет нагрева флюида в насосе отключен;
                                - любое число - расчет нагрева идет с заданным
                                температурным градиентом, К/ступень
        Returns
        -------
        :return: давление на приеме/выкиде, Па
        :return: температура на приеме/выкиде, К
        :return: статус расчета:
                            0 - успешен;
                            1 - во время расчета достигнуто минимальное давление;
                            -1 - завершен с ошибкой интегрирования
        """
        # проверка для остановленной скважины
        if freq == 0:
            return p, t, 0

        if q_liq is not None:
            self.fluid.q_fluid = q_liq
            self.fluid.reinit_q_fluid(q_liq)

        if wct is not None:
            self.fluid.wct = wct
            self.fluid.reinit_wct(wct)

        if phase_ratio_value is not None:
            fluid_t = deepcopy(self.fluid)
            pvt_data = fluid_t.pvt_model_data
            phase_ratio = pvt_data["black_oil"]["phase_ratio"]
            self.fluid.reinit()
            phase_ratio.update({"value": phase_ratio_value})
            self.fluid.reinit_phase_ratio(phase_ratio)


        self.slippage_factor = slippage_factor
        freq *= slippage_factor

        # Отношение фактической частоты к номинальной
        freq_ratio = freq / self.freq_nom

        # Если поправка не задана, то считаем равной 1
        if head_factor is None:
            head_factor = 1

        # Вывод номинальной точки
        self.__hr_nom_point()

        self.fluid.calc_flow(p, t)
        # Поправка на свободный газ на приеме насоса
        if self.gas_correction:
            k_gas_degr = self._gas_corr(freq_ratio)
        else:
            k_gas_degr = 1

        # Коррекция НРХ на частоту и свободный газ
        self.rate_points_corr = freq_ratio * self.rate_points
        self.head_points_corr = freq_ratio**2 * self.head_points * k_gas_degr
        self.power_points_corr = freq_ratio**3 * self.power_points
        self.eff_points_corr = self.eff_points

        # Удаление закешированных значений функций насоса
        self.__renew_pump_func()

        # Расчет распределения давления по длине ЭЦН
        stage_array, y_array, status = self.__integrate_esp(p, t, head_factor, int_method, direction_to, temp_grad)
        head_array = np.abs(y_array[0, :])

        pressure_array = y_array[1, :]
        power_fluid_array = np.abs(y_array[2, :])
        power_esp_array = np.abs(y_array[3, :])
        temp_array = y_array[4, :]

        self.power_fluid = power_fluid_array[-1]
        self.power_esp = power_esp_array[-1]

        self.efficiency = self.power_fluid / self.power_esp if self.power_esp != 0 else 0

        self.head = head_array[-1]
        self.dp = abs(pressure_array[0] - pressure_array[-1])
        p_finish = pressure_array[-1]
        t_finish = temp_array[-1]

        # Сохранение распределений
        self.distributions["depth"] = np.array([self.h_mes, self.h_mes])

        if direction_to == "in":
            self.distributions["p"] = np.array([p, p_finish])
            self.distributions["t"] = np.array([t, t_finish])
        else:
            self.distributions["p"] = np.array([p_finish, p])
            self.distributions["t"] = np.array([t_finish, t])

        if extra_output:
            self.distributions.update(
                tls.make_output_attrs(self.fluid, self.distributions["p"], self.distributions["t"])
            )

        return p_finish, t_finish, status
