"""
Модуль, описывающий работу штуцера
"""
import math
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import scipy.optimize as opt

import ffmt.pvt.adapter as fl
import unifloc.equipment.equipment as eq
import unifloc.service._constants as const
import unifloc.service._tools as tls
import unifloc.tools.exceptions as exc


class Choke(eq.Equipment):
    """
    Класс, описывающий работу штуцера
    """

    def __init__(
        self,
        h_mes: float,
        d: float,
        d_up: float,
        fluid: fl.FluidFlow,
        correlations: Dict[str, str] = const.CHOKE_CORRS,
        temperature_drop: bool = False,
    ):
        """

        Parameters
        ----------
        :param h_mes:  глубина, на которой установлен штуцер, м
        :param d: диаметр штуцера, м
        :param d_up: диаметр трубы выше по потоку, м
        :param fluid: объект для расчета флюида
        :param correlations: название корреляций для расчета перепада давления в штуцере
               обязательные ключи: "subcritical", "critical"
        :param temperature_drop: флаг включения/отключения расчета перепада температуры на штуцере:
                                - True - перепад температуры учитывается (расчет эффекта
                                Джоуля-Томсона)
                                - False - перепад температуры не учитывается

        Examples:
        --------
        >>> # Пример исходных данных для штуцера
        >>> from unifloc.pvt.fluid_flow import FluidFlow
        >>> from unifloc.equipment.choke import Choke
        >>>
        >>>
        >>> fluid_data = {
        ...     "q_fluid": 100 / 86400, "pvt_model_data": {
        ...         "black_oil": {
        ...             "gamma_gas": 0.6, "gamma_wat": 1, "gamma_oil": 0.8,
        ...             "wct": 0.1,
        ...             "phase_ratio": {"type": "GOR", "value": 80},
        ...             "oil_correlations": {
        ...                 "pb": "Standing",
        ...                 "rs": "Standing",
        ...                 "rho": "Standing",
        ...                 "b": "Standing",
        ...                 "mu": "Beggs",
        ...                 "compr": "Vasquez",
        ...                 "hc": "const"
        ...             },
        ...             "gas_correlations": {
        ...                 "ppc": "Standing",
        ...                 "tpc": "Standing",
        ...                 "z": "Dranchuk",
        ...                 "mu": "Lee",
        ...                 "hc": "const"
        ...             },
        ...             "water_correlations": {
        ...                 "b": "McCain",
        ...                 "compr": "Kriel",
        ...                 "rho": "Standing",
        ...                 "mu": "McCain",
        ...                 "hc": "const"
        ...             },
        ...             "rsb": {"value": 50, "p": 10000000, "t": 303.15},
        ...             "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
        ...             "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
        ...             "table_model_data": None, "use_table_model": False
        ...         }
        ...     }
        ... }
        >>> h_mes = 1000
        >>> d_choke = 0.005
        >>> d_up = 0.062
        >>>
        >>> p_fl = 1000000
        >>> t_fl = 303.15
        >>>
        >>> fluid = FluidFlow(**fluid_data)
        >>> test = Choke(h_mes=h_mes, d=d_choke, d_up=d_up, fluid=fluid)
        >>> results = test.calc_pt(p_fl, t_fl, 1)
        """
        super().__init__(h_mes)

        self.correlations = {k: v.lower() for k, v in correlations.items()}
        self._d_up = d_up
        self.d = d

        self.s_bean = math.pi * d**2 / 4
        self.fluid = fluid
        self.c_choke = 0.6
        self.c_vg = None
        self.c_vl = None
        self.z_l = 1
        self.extra_dp = 0
        self.dp = None
        self.dt = None
        self.distributions = dict()
        self.temperature_drop = temperature_drop
        self.dist_depth = np.array([h_mes, h_mes])

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d = value
        self.d_ratio = value

    @property
    def d_ratio(self):
        return self._d_ratio

    @d_ratio.setter
    def d_ratio(self, value):
        self._d_ratio = value / self._d_up
        if self._d_ratio >= 1:
            raise exc.UniflocPyError(
                f"Диаметр штуцера {value} >= диаметра трубы выше по потоку {self._d_up} - расчет невозможен."
            )

    def __repr__(self):
        return f"Choke, d_ratio: {self.d_ratio}, correlations: {self.correlations}"

    def __calc_dp_mechanistic(self, fluid_velocity: float, rho_mix: float, gf: float, z: float) -> float:
        """
        Метод расчета перепада давления на штуцере по механистической корреляции

        Parameters
        ----------
        :param fluid_velocity: скорость потока, м/с
        :param rho_mix: плотность смеси, кг/м3
        :param gf: доля газа, д.ед
        :param z: z - фактор

        :return: перепад давления на штуцере, Па
        -------
        """
        dp = (rho_mix * (fluid_velocity**2) / 2) * (
            ((1 - gf) / ((self.c_vl * self.z_l) ** 2)) + (gf / ((self.c_vg * z) ** 2))
        )
        return dp

    def __calc_q_mechanistic(self, dp: float, rho_mix: float, gf: float, z: float) -> float:
        """
        Метод расчета дебита жидкости в штуцере по механистической корреляции

        Parameters
        ----------
        :param dp: перепад давления на штуцере, Па
        :param rho_mix: плотность смеси, кг/м3
        :param gf: доля газа, д.ед
        :param z: z - фактор

        :return: дебит жидкости, м3/с
        -------
        """
        wct = self.fluid.wct
        oil_frac = 1 - wct
        q_mix = self.s_bean * math.sqrt(
            (2 * dp) / (rho_mix * (((1 - gf) / ((self.c_vl * self.z_l) ** 2)) + (gf / ((self.c_vg * z) ** 2))))
        )

        if self.fluid.phase_ratio["type"].lower() == "gor":
            q_liq = q_mix / (
                oil_frac * self.fluid.bo
                + self.fluid.bg * (oil_frac * self.fluid.rp - oil_frac * self.fluid.rs)
                + wct * self.fluid.bw
            )
        else:
            q_liq = q_mix / (
                oil_frac * self.fluid.bo
                + self.fluid.bg * (self.fluid.rp - oil_frac * self.fluid.rs)
                + wct * self.fluid.bw
            )

        return q_liq

    def __calc_dp(
        self,
        regime_type: str,
        fluid_velocity: float,
        rho_mix: float,
        gf: float,
        z: float,
        c_choke: Optional[float] = None,
    ) -> float:
        """
        Метод расчета перепада давления на штуцере

        Parameters
        ----------
        :param regime_type: режим течения флюида через штуцер
        :param fluid_velocity: скорость потока, м/с
        :param rho_mix: плотность смеси, кг/м3
        :param gf: доля газа, д.ед
        :param z: z - фактор
        :param c_choke: коэффициент калибровки штуцера

        :return: перепад давления на штуцере, Па
        -------
        """
        corr_name = self.correlations[regime_type]
        if corr_name in {"mechanistic", "api14b"}:
            self.__calc_choke_calibration(corr_name, c_choke)
            return self.__calc_dp_mechanistic(fluid_velocity, rho_mix, gf, z)
        raise exc.NotImplementedChokeCorrError(
            f"Корреляция {corr_name} еще не реализована."
            f"Используйте другую корреляцию",
            corr_name,
        )

    def __calc_q(
        self,
        regime_type: str,
        dp: float,
        rho_mix: float,
        gf: float,
        z: float,
        c_choke: Optional[float] = None,
    ) -> float:
        """
        Метод расчета дебита жидкости в штуцере

        Parameters
        ----------
        :param regime_type: режим течения флюида через штуцер
        :param dp: перепад давления на штуцере, Па
        :param rho_mix: плотность смеси, кг/м3
        :param gf: доля газа, д.ед
        :param z: z - фактор
        :param c_choke: коэффициент калибровки штуцера

        :return: дебит жидкости, м3/с
        -------
        """
        corr_name = self.correlations[regime_type]
        if corr_name in {"mechanistic", "api14b"}:
            self.__calc_choke_calibration(corr_name, c_choke)
            return self.__calc_q_mechanistic(dp, rho_mix, gf, z)
        raise exc.NotImplementedChokeCorrError(
            f"Корреляция {corr_name} еще не реализована." f"Используйте другую корреляцию",
            {corr_name},
        )

    def __calc_choke_calibration(
        self,
        correlation: str,
        c_choke: Optional[float] = None
    ):
        """
        Метод расчета калибровки штуцера

        Parameters
        ----------
        :param correlation: название корреляции
        :param c_choke: коэффициент калибровки штуцера

        """
        if correlation == "mechanistic":
            self.c_vg, self.c_vl = 1.3, 1.3
            if c_choke is not None:
                #  v.1.3.17 - ограничения на границы удалены
                # if 0 < c_choke <= 1.3:
                self.c_choke = c_choke
                # else:
                #     return
            d_den = 1 - self.d_ratio**4
            if d_den > 0:
                c_v = self.c_choke / math.sqrt(d_den)
                self.c_vl, self.c_vg = c_v, c_v
        elif correlation == "api14b":
            self.c_vl, self.c_vg = 0.85, 0.9

    @staticmethod
    def __set_regime_type(current_param: float, critical_param: float) -> str:
        """
        Метод определения режима течения флюида через штуцер

        Parameters
        ----------
        :param current_param: текущее значение параметра
        :param critical_param: критериальное значение параметра

        :return: тип режима течения
        -------
        """
        if abs(current_param - critical_param) < 0.01:
            return "critical"
        if current_param > critical_param:
            return "supercritical"
        return "subcritical"

    def __calc_sonic_velocity(
        self,
        p_out: float,
        gas_density: float,
        gf: float,
        wf: float
    ) -> float:
        """
        Метод расчета скорости звука

        Parameters
        ----------
        :param p_out: давление ниже по потоку (на выходе из штуцера), Па
        :param gas_density: плотность газа ниже по потоку (на выходе из штуцера), кг/м3
        :param wf: доля воды в потоке при условиях на выходе из штуцера, д.ед.
        :param gf: доля газа в потоке при условиях на выходе из штуцера, д.ед.

        :return: скорость звука, м/с
        -------
        """
        if gf == 0:
            # скорость звука в керосине при 25С = 4343 ft/s = 1323.7464 м/с
            # скорость звука в воде при 25С = 4897 ft/s = 1492.6056 м/с
            sonic_velocity = (1 - wf) * 1323.7464 + wf * 1492.6056
        else:
            sonic_velocity = math.sqrt(const.CP_CV * p_out / gas_density)

        return sonic_velocity

    def __crit_vel_error(
        self,
        p_crit: float,
        t_crit: float,
        sonic_vel: float) -> float:
        """
        Метод расчета ошибки в скорости звука

        Parameters
        ----------
        :param p_crit: критическое давление, Па
        :param t_crit: критическая температура, К
        :param sonic_vel: скорость звука, м/с

        :return: разница между рассчитанной скоростью флюида и скоростью звука, м/с
        -------
        """
        self.fluid.calc_flow(p_crit, t_crit)
        fluid_vel = self.fluid.qm / self.s_bean
        return fluid_vel - sonic_vel

    def __calc_pt(
        self,
        p_mes: float,
        t_mes: float,
        flow_direction: int,
        c_choke: float,
        catch_supercritical: bool,
    ) -> tuple:
        """
        Метод итеративного пересчета давления и температуры на другом конце штуцера

        Parameters
        ----------
        :param p_mes: измеренное давление за или перед штуцером
                      (в зависимости от flow_direction), Па
        :param t_mes: измеренная температура за или перед штуцером
                      (в зависимости от flow_direction), К
        :param flow_direction: направление потока
                               1 - к p_mes;
                               -1 - от p_mes
        :param c_choke: коэффициент калибровки штуцера
        :param catch_supercritical: флаг, определяющий выдачу ошибки для суперкритического режима за штуцером

        :return:- давление на другом конце штуцера, Па;
                 - температура на другом конце штуцера, К;
                 - статус расчета:
                    0 - успешен;
                    1 - во время расчета достигнуто минимальное давление;
                    -1 - во время расчета достигнуто критическое давление
                 - экстра-перепад давления, возникающий при расчете в критической области \
                 от известного давления за штуцером, Па
        -------
        """
        status = 0
        self.extra_dp = 0

        p_mes_new = p_mes
        t_mes_new = t_mes

        if flow_direction == 1:
            self.fluid.calc_flow(p_mes, t_mes)

            # Расчет скорости звука
            sonic_velocity = self.__calc_sonic_velocity(
                p_mes,
                self.fluid.rg,
                self.fluid.gf,
                self.fluid.wf,
            )
            fluid_velocity = self.fluid.qm / self.s_bean
            regime_type = self.__set_regime_type(fluid_velocity, sonic_velocity)

            # Расчет критического давления
            try:
                p_crit = opt.brentq(
                    self.__crit_vel_error,
                    a=101325,
                    b=100000000,
                    args=(t_mes, sonic_velocity),
                )
            except ValueError:
                p_crit = opt.minimize_scalar(
                    self.__crit_vel_error,
                    args=(t_mes, sonic_velocity),
                    bounds=(101325, 100000000),
                    method="bounded",
                ).x

            if self.temperature_drop:
                jt = self.fluid.calc_joule_thomson_coeff(t_mes, p_crit)
                dt = (p_crit - p_mes) * jt
            else:
                dt = 0
            t_crit = t_mes + dt

            if regime_type == "supercritical":
                p_mes_new = p_crit
                t_mes_new = t_crit

            # Расчет давления на другом конце штуцера
            try:
                p_calc = opt.brentq(
                    self.__calc_p_error,
                    a=p_mes_new,
                    b=100000000,
                    args=(
                        p_mes_new,
                        t_mes_new,
                        flow_direction,
                        c_choke,
                        sonic_velocity,
                    ),
                    xtol=10000,
                    maxiter=100,
                )
            except ValueError:
                p_calc = opt.minimize_scalar(
                    self.__calc_p_error,
                    args=(
                        p_mes_new,
                        t_mes_new,
                        flow_direction,
                        c_choke,
                        sonic_velocity,
                    ),
                    bounds=(p_mes_new, 100000000),
                    method="bounded",
                ).x
            if regime_type == "supercritical":
                self.dp = p_calc - p_crit * flow_direction
                self.extra_dp = p_crit - p_mes
        else:
            # Расчет давления на другом конце штуцера
            try:
                p_calc = opt.brentq(
                    self.__calc_p_error,
                    a=101325,
                    b=100000000,
                    args=(p_mes_new, t_mes_new, flow_direction, c_choke),
                    xtol=10000,
                    maxiter=100,
                )
            except ValueError:
                p_calc = opt.minimize_scalar(
                    self.__calc_p_error,
                    p_mes_new,
                    args=(p_mes_new, t_mes_new, flow_direction, c_choke),
                    bounds=(1, p_mes_new),
                    method="bounded",
                ).x

            self.fluid.calc_flow(p_calc, t_mes + flow_direction * self.dt)

            # Расчет скорости звука
            sonic_velocity = self.__calc_sonic_velocity(
                p_calc,
                self.fluid.rg,
                self.fluid.gf,
                self.fluid.wf,
            )
            fluid_velocity = self.fluid.qm / self.s_bean
            regime_type = self.__set_regime_type(fluid_velocity, sonic_velocity)
            if catch_supercritical:
                if regime_type == "supercritical" or p_mes - self.dp < 0:
                    raise exc.UniflocPyError("Режим за штуцером суперкритический. Расчет остановлен.")

        p_finish = p_mes + flow_direction * self.dp + flow_direction * self.extra_dp
        t_finish = t_mes + flow_direction * self.dt
        return p_finish, t_finish, status, self.extra_dp

    def __calc_p_error(
        self,
        p_iter: float,
        p_mes: float,
        t_mes: float,
        flow_direction: int,
        c_choke: float,
        sonic_velocity: Optional[float] = None,
    ) -> float:
        """
        Метод расчета ошибки в давлении на другом конце штуцера

        Parameters
        ----------
        :param p_iter: итерируемое давление на другом конце штуцера в зависимости от заданного
        измеренного, Па
        :param p_mes: измеренное давление на входе или выходе из штуцера, Па
        :param t_mes: измеренная температура на входе или выходе из штуцера, К
        :param flow_direction: направление потока
                               1 - к p_mes;
                               -1 - от p_mes
        :param c_choke: коэффициент калибровки штуцера
        :param sonic_velocity: скорость звука при условиях на выходе из штуцера
                                (подается в случае расчета давления на входе в штуцер), м/с

        :return: разница между рассчитанным давлением на одном конце штуцера и измеренным на
        другом, Па
        -------
        """
        if self.temperature_drop:
            jt_old = self.fluid.calc_joule_thomson_coeff(t_mes, p_iter)
            dt_old = flow_direction * (p_iter - p_mes) * jt_old
        else:
            dt_old = 0
        t_iter = t_mes + flow_direction * dt_old

        z = 1 - ((0.41 + 0.35 * self.d_ratio**4) / const.CP_CV) * ((p_iter - p_mes) / p_iter)

        self.fluid.calc_flow(p_iter, t_iter)

        if flow_direction == -1:
            sonic_velocity = self.__calc_sonic_velocity(
                p_iter,
                self.fluid.rg,
                self.fluid.gf,
                self.fluid.wf,
            )
            p_crit = opt.minimize_scalar(
                self.__crit_vel_error,
                p_mes,
                args=(t_iter, sonic_velocity),
                bounds=(1, p_mes),
                method="bounded",
            ).x

            z = 1 - ((0.41 + 0.35 * self.d_ratio**4) / const.CP_CV) * ((p_mes - p_iter) / p_mes)

            self.fluid.calc_flow(p_mes, t_mes)

        fluid_velocity = self.fluid.qm / self.s_bean
        regime_type = self.__set_regime_type(fluid_velocity, sonic_velocity)

        if regime_type == "supercritical":
            regime_type = "critical"
            if flow_direction == -1:
                z = 1 - ((0.41 + 0.35 * self.d_ratio**4) / (const.CP_CV)) * ((p_mes - p_crit) / p_mes)

        dp = self.__calc_dp(
            regime_type,
            fluid_velocity,
            self.fluid.rm,
            self.fluid.gf,
            z,
            c_choke,
        )

        if self.temperature_drop and p_mes + flow_direction * dp > 0:
            jt = self.fluid.calc_joule_thomson_coeff(t_iter, p_mes + flow_direction * dp)
            dt = dp * jt
        else:
            dt = dt_old

        p_calc = p_iter - flow_direction * dp
        self.dp = dp
        self.dt = dt

        return p_calc - p_mes

    def calc_pt(
        self,
        p_mes: float,
        t_mes: float,
        flow_direction: float,
        q_liq: Optional[float] = None,
        wct: Optional[float] = None,
        phase_ratio_value: Optional[float] = None,
        c_choke: Optional[float] = None,
        extra_output: bool = False,
        catch_supercritical: bool = True,
    ) -> tuple:
        """
        Расчет давления и температуры на другом конце штуцера

        Parameters
        ----------
        :param p_mes: измеренное давление за или перед штуцером
                      (в зависимости от flow_direction), Па
        :param t_mes: измеренная температура за или перед штуцером
                      (в зависимости от flow_direction), К
        :param flow_direction: направление потока
                               1 - к p_mes;
                               -1 - от p_mes
        :param q_liq: дебит жидкости, м3/c
        :param wct: обводненность, доли ед.
        :param phase_ratio_value: соотношение газа к флюиду для выбранного типа фазового соотношения,
            ст. м3 газа/ст. м3 флюид
        :param c_choke: коэффициент калибровки штуцера
        :param extra_output: флаг расчета дополнительных распределений
        :param catch_supercritical: флаг, определяющий выдачу ошибки для суперкритического режима за штуцером

        Returns
        -------
        :return: - давление на другом конце штуцера, Па абс.;
                 - температура на другом конце штуцера, К;
                 - статус расчета:
                    0 - успешен;
                    1 - во время расчета достигнуто минимальное давление;
                    -1 - во время расчета достигнуто критическое давление
                 - экстра-перепад давления, возникающий при расчете в критической области \
                 от известного давления за штуцером, Па
        """
        self.distributions = {"depth": self.dist_depth}

        if wct is not None:
            self.fluid.reinit_wct(wct)

        if phase_ratio_value is not None:
            fluid_t = deepcopy(self.fluid)
            pvt_data = fluid_t.pvt_model_data
            phase_ratio = pvt_data["black_oil"]["phase_ratio"]
            self.fluid.reinit()
            phase_ratio.update({"value": phase_ratio_value})
            self.fluid.reinit_phase_ratio(phase_ratio)

        if q_liq is not None:
            self.fluid.reinit_q_fluid(q_liq)

        # Проверка корректности направления потока
        if flow_direction not in {1, -1}:
            raise exc.UniflocPyError(f"Неправильно задано направление потока - {flow_direction}")

        p, t, status, extra_dp = self.__calc_pt(p_mes, t_mes, flow_direction, c_choke, catch_supercritical)

        # Сохранение распределений
        if flow_direction == 1:
            self.distributions["p"] = np.array([p_mes, p])
            self.distributions["t"] = np.array([t_mes, t])
        else:
            self.distributions["p"] = np.array([p, p_mes])
            self.distributions["t"] = np.array([t, t_mes])

        if extra_output:
            self.distributions.update(
                tls.make_output_attrs(self.fluid, self.distributions["p"], self.distributions["t"])
            )

        return p, t, status, extra_dp

    def calc_qliq(
        self,
        p_in: float,
        t_in: float,
        p_out: float,
        t_out: float,
        wct: Optional[float] = None,
        phase_ratio_value: Optional[float] = None,
        c_choke: Optional[float] = None,
        explicit: bool = False,
    ) -> float:
        """
        Расчет дебита жидкости при известных давлениях на входе и выходе штуцера

        Parameters
        ----------
        :param p_in: давление на входе в штуцер, Па
        :param t_in: температура на входе в штуцер, К
        :param p_out: давление на выходе из штуцера, Па
        :param t_out: температура на выходе из штуцера, К
        :param wct: обводненность, доли ед.
        :param phase_ratio_value: соотношение газа к флюиду для выбранного типа фазового соотношения,
            ст. м3 газа/ст. м3 флюид
        :param c_choke: коэффициент калибровки штуцера
        :param explicit: флаг явного расчета дебита, не рекомендуется использовать при больших ГФ и дебитах жидкости, тк
            будет выдавать неправильный результат на критическом режиме. При использовании данного флага считается, что
            режим докритический

        :return: дебит жидкости, м3/с

        Examples:
        --------
        >>> # Пример исходных данных для штуцера
        >>> from unifloc.pvt.fluid_flow import FluidFlow
        >>> from unifloc.equipment.choke import Choke
        >>>
        >>> fluid_data = {'q_fluid': 100 / 86400, 'wct': 0,
        ...              'pvt_model_data': {
        ...                  'black_oil': {'gamma_gas': 0.6, 'gamma_wat': 1, 'gamma_oil': 0.8,
        ...                                'phase_ratio': {"type": "GOR", "value": 80}}}}
        >>> fluid = FluidFlow(**fluid_data)
        >>> choke = Choke(1000, 0.01, 0.062, fluid)
        >>> # Быстрый расчет (учитывает только докритический режим)
        >>> q_liq_expl = choke.calc_qliq(120 * 101325, 303.15, 100 * 101325, 303.15,
        >>> explicit=True, c_choke=5)
        >>> # Долгий (более правильный) расчет (медленнее в 4300 раз)
        >>> q_liq_impl = choke.calc_qliq(120 * 101325, 303.15, 100 * 101325, 303.15,
        >>> explicit=False, c_choke=5)
        """
        if wct is not None:
            self.fluid.reinit_wct(wct)

        if phase_ratio_value is not None:
            fluid_t = deepcopy(self.fluid)
            pvt_data = fluid_t.pvt_model_data
            phase_ratio = pvt_data["black_oil"]["phase_ratio"]
            self.fluid.reinit()
            phase_ratio.update({"value": phase_ratio_value})
            self.fluid.reinit_phase_ratio(phase_ratio)

        if p_in > p_out:
            return self.__calc_qliq(p_in, t_in, p_out, t_out, c_choke, explicit)

        raise exc.UniflocPyError(
            f"Давление на входе, {p_in} <= давления на выходе, {p_out} - течение флюида невозможно."
        )

    def __calc_qliq(
        self,
        p_in: float,
        t_in: float,
        p_out: float,
        t_out: float,
        c_choke: Optional[float] = None,
        explicit: bool = False,
    ) -> float:
        """
        Расчет дебита жидкости при известных давлениях на входе и выходе штуцера

        Parameters
        ----------
        :param p_in: давление на входе в штуцер, Па
        :param t_in: температура на входе в штуцер, К
        :param p_out: давление на выходе из штуцера, Па
        :param t_out: температура на выходе из штуцера, К
        :param c_choke: коэффициент калибровки штуцера
        :param explicit: флаг явного расчета

        :return: дебит жидкости, м3/с
        -------
        """
        if explicit:
            p, t = p_in, t_in
        else:
            p, t = p_out, t_out

        self.fluid.calc_flow(p, t)
        wct = self.fluid.wct
        oil_frac = 1 - wct

        rs = self.fluid.rs
        bg = self.fluid.bg
        bo = self.fluid.bo
        bw = self.fluid.bw

        if self.fluid.phase_ratio["type"].lower() == "gor":
            fg = bg * oil_frac * (self.fluid.rp - rs)
        else:
            fg = bg * (self.fluid.rp - rs * oil_frac)

        fo = oil_frac * bo
        fw = wct * bw
        fm = fg + fo + fw
        gf = fg / fm

        sonic_velocity = self.__calc_sonic_velocity(p_out, self.fluid.rg, gf, self.fluid.wf)
        z = 1 - ((0.41 + 0.35 * self.d_ratio**4) / const.CP_CV) * ((p_in - p_out) / p_in)

        if explicit:
            rho_mix = (fo * self.fluid.ro + fg * self.fluid.rg + fw * self.fluid.rw) / fm
            return self.__calc_q("subcritical", p_in - p_out, rho_mix, gf, z, c_choke)

        q_liq = opt.brentq(
            self.__calc_dp_error,
            a=0.00001,
            b=10,
            args=(p_in, t_in, p_out, t_out, sonic_velocity, z, c_choke),
        )
        return q_liq

    def __calc_dp_error(
        self,
        q_iter: float,
        p_in: float,
        t_in: float,
        p_out: float,
        t_out: float,
        sonic_velocity: float,
        z: float,
        c_choke: float,
    ) -> float:
        """
        Метод расчета ошибки в перепаде давления

        Parameters
        ----------
        :param q_iter: итерируемый дебит жидкости, м3/с
        :param p_in: давление на входе в штуцер, Па
        :param t_in: температура на входе в штуцер, К
        :param p_out: давление на выходе из штуцера, Па
        :param t_out: температура на выходе из штуцера, К
        :param sonic_velocity: скорость звука после штуцера, м/с
        :param z: z-фактор
        :param c_choke: коэффициент калибровки штуцера

        :return: разница между рассчитанным и заданным перепадами давления, Па
        -------
        """

        self.extra_dp = 0

        self.fluid.q_fluid = q_iter
        self.fluid.reinit_q_fluid(q_iter)

        self.fluid.calc_flow(p_out, t_out)
        fluid_velocity_out = self.fluid.qm / self.s_bean
        regime_type = self.__set_regime_type(fluid_velocity_out, sonic_velocity)

        try:
            self.p_crit = opt.brentq(self.__crit_vel_error, a=1, b=10000000000, args=(t_out, sonic_velocity))
        except ValueError:
            self.p_crit = opt.minimize_scalar(
                self.__crit_vel_error,
                p_out,
                args=(t_out, sonic_velocity),
                bounds=(p_out, p_in),
                method="bounded",
            ).x
        self.fluid.calc_flow(p_in, t_in)
        fluid_velocity = self.fluid.qm / self.s_bean

        if regime_type == "supercritical":
            regime_type = "critical"
            z = 1 - ((0.41 + 0.35 * self.d_ratio**4) / const.CP_CV) * ((p_in - self.p_crit) / p_in)
            self.extra_dp = self.p_crit - p_out

        self.dp = self.__calc_dp(regime_type, fluid_velocity, self.fluid.rm, self.fluid.gf, z, c_choke)

        return self.dp + self.extra_dp - (p_in - p_out)
