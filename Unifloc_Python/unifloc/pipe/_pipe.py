"""
Модуль, для интегрирования трубы
"""
from copy import deepcopy
from math import pi

import numpy as np
import scipy.integrate as integr

import unifloc.pipe._ansari as an
import unifloc.pipe._beggsbrill as bb
import unifloc.pipe._dunsros as dr
import unifloc.pipe._gray as gr
import unifloc.pipe._hagedornbrown as hb
import unifloc.pipe._hasankabir as hk
import unifloc.pipe._static as st
import unifloc.pipe._tempcorr as tcor
import unifloc.service._constants as const
import unifloc.service._patches as patch
import unifloc.service._tools as tls
import unifloc.tools.exceptions as exc


class Pipe:
    """
    Класс трубы. Служит для расчета градиента давления в трубе
    при помощи различных многофазных корреляций.
    """

    __slots__ = [
        "fluid",
        "roughness",
        "_d",
        "_d_tub_out",
        "_d_cas_in",
        "_hydrcorr",
        "hydr_corr_type",
        "tempcorr",
        "v_mix",
        "v_mix_krit",
        "s_wall",
    ]

    def __init__(
        self,
        fluid,
        d,
        roughness,
        hydr_corr_type="Gray",
        s_wall=None,
        d_tub_out=None,
        d_cas_in=None,
    ):
        """

        Parameters
        ----------
        :param fluid: объект PVT модели флюида
        :param d: внутренний диаметр трубы, м
        :param hydr_corr_type: гидравлическая корреляция, строка
        :param s_wall: толщина стенки, м
        :param roughness: шероховатость трубы, м
        :param d_tub_out: внешний диаметр НКТ, м
        :param d_cas_in: внутренний диаметр ЭК, м
        """

        self.fluid = fluid
        self._d = d
        self._d_tub_out = d_tub_out
        self._d_cas_in = d_cas_in
        self.s_wall = s_wall
        self.roughness = roughness
        self._hydrcorr = None
        self.hydr_corr_type = hydr_corr_type
        self.hydrcorr = hydr_corr_type.lower()
        self.tempcorr = tcor.TempCorr()

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d = value
        self._hydrcorr.d = value

    @property
    def d_tub_out(self):
        return self._d_tub_out

    @d_tub_out.setter
    def d_tub_out(self, value):
        self._d_tub_out = value
        if self.hydr_corr_type == "HasanKabir":
            self._hydrcorr.d_tub_out = value

    @property
    def d_cas_in(self):
        return self._d_cas_in

    @d_cas_in.setter
    def d_cas_in(self, value):
        self._d_cas_in = value
        if self.hydr_corr_type == "HasanKabir":
            self._hydrcorr.d_cas_in = value

    @property
    def hydrcorr(self):
        return self._hydrcorr

    @hydrcorr.setter
    def hydrcorr(self, value):
        """
        Изменение типа корреляции в Pipe и дочерних классах

        Parameters
        ----------
        :param value: тип гидравлической корреляции

        :return: Тип корреляции

        """
        if value.lower() == "ansari":
            self._hydrcorr = an.Ansari(self.d)
            self.hydr_corr_type = "Ansari"
        elif value.lower() == "beggsbrill":
            self._hydrcorr = bb.BeggsBrill(self.d)
            self.hydr_corr_type = "BeggsBrill"
        elif value.lower() == "dunsros":
            self._hydrcorr = dr.DunsRos(self.d)
            self.hydr_corr_type = "DunsRos"
        elif value.lower() == "gray":
            self._hydrcorr = gr.Gray(self.d)
            self.hydr_corr_type = "Gray"
        elif value.lower() == "hagedornbrown":
            self._hydrcorr = hb.HagedornBrown(self.d)
            self.hydr_corr_type = "HagedornBrown"
        elif value.lower() == "hasankabir":
            self._hydrcorr = hk.HasanKabir(d=self.d, d_tub_out=self.d_tub_out, d_cas_in=self.d_cas_in)
            self.hydr_corr_type = "HasanKabir"
        elif value.lower() == "static":
            self._hydrcorr = st.Static(self.d)
            self.hydr_corr_type = "Static"
        else:
            raise exc.NotImplementedHydrCorrError(
                f"Корреляция {value} еще не реализована."
                f"Используйте другие корреляции",
                value,
            )

    def __extract_output(self, p, t):
        pvt_array = tls.extract_output_fluid(self.fluid)
        corr_array = np.empty(len(const.DISTRS_PIPE))
        corr_array[0] = self.hydrcorr.dp_dl
        corr_array[1] = self.hydrcorr.dp_dl_fr
        corr_array[2] = self.hydrcorr.dp_dl_gr
        corr_array[3] = self.hydrcorr.dp_dl_acc
        corr_array[4] = self.hydrcorr.hl
        corr_array[5] = self.hydrcorr.ff
        corr_array[6] = self.hydrcorr.vsl
        corr_array[7] = self.hydrcorr.vsg
        corr_array[8] = self.hydrcorr.vsm
        corr_array[9] = self.hydrcorr.fp
        corr_array[10] = self.hydrcorr.ll
        corr_array[11] = self.hydrcorr.n_re
        corr_array[12] = self.hydrcorr.angle
        corr_array[13] = self.hydrcorr.vl
        corr_array[14] = self.hydrcorr.vg

        # version 1
        # Определение критической скорости выноса жидкости из ГК скважины
        corr_array[15] = self.hydrcorr.calc_vmix(p=p, t=t, z=self.fluid.z, qg=self.fluid.qg, d=self.d)
        corr_array[16] = self.hydrcorr.calc_vmix_krit(stlg=self.fluid.stwg, rho_liq=self.fluid.rho_wat, rho_gas=self.fluid.rho_gas)
        return np.append(pvt_array, corr_array)

    def calc_extra_output(
        self,
        pressure_array,
        temperature_array,
        depth_array,
        trajectory,
        amb_temp_dist,
        directions,
        grav_holdup_factor,
        friction_factor,
        d_func,
        d_tub_out_func,
        d_cas_in_func,
        heat_balance,
    ) -> dict:
        """
        Расчет экстра-выводных параметров

        Parameters
        ----------
        :param pressure_array: массив давлений
        :param temperature_array: массив температур
        :param depth_array: массив глубин
        :param trajectory: объект с траекторией
        :param amb_temp_dist: объект с распределением температуры породы
        :param directions: множитель для направления потока, флаг направления расчета
        :param grav_holdup_factor: коэффициент гравитации/проскальзывания
        :param friction_factor: коэффициент трения
        :param d_func: объект расчет диаметра НКТ по глубине скважины
        :param d_tub_out_func: объект расчет внешнего диаметра НКТ по глубине скважины
        :param d_cas_in_func: объект расчет внутреннего диаметра ЭК по глубине скважины
        :param heat_balance: опция учета теплопотерь

        :return: словарь с распределением параметров по глубине
        """
        result_data = np.empty([len(const.DISTRS), len(pressure_array)])

        p_prev, t_prev, h_prev = pressure_array[0], temperature_array[0], depth_array[0]
        for i, p in enumerate(pressure_array):
            self.__integr_func(
                [depth_array[i], h_prev],
                [[p, temperature_array[i]], [p_prev, t_prev]],
                trajectory,
                amb_temp_dist,
                directions,
                grav_holdup_factor,
                friction_factor,
                d_func,
                d_tub_out_func,
                d_cas_in_func,
                heat_balance,
            )
            result_data[:, i] = self.__extract_output(p, temperature_array[i])
            p_prev, t_prev, h_prev = p, temperature_array[i], depth_array[i]
        result = {k: result_data[i] for i, k in enumerate(const.DISTRS)}

        # Проверка распределений на существование
        result = tls.check_nan(result)

        return result

    def __integr_func(
        self,
        hh,
        pt,
        trajectory,
        amb_temp_dist,
        directions,
        holdup_factor,
        friction_factor,
        d_func,
        d_tub_out_func,
        d_cas_in_func,
        heat_balance,
    ):
        """
        Функция для интегрирования трубы

        Parameters
        ----------
        :param hh: текущая глубина, предыдущая успешная глубина, м
        :param pt: текущее давление и температура, давление и температура на предыдущей успешной глубине , Па и К
        :param trajectory: объект с траекторией
        :param amb_temp_dist: объект с распределением температуры породы
        :param directions: множитель для направления потока, флаг направления расчета
        :param holdup_factor: коэффициент истинного содержания жидкости/гравитации
        :param friction_factor: коэффициент трения
        :param d_func: объект расчет диаметра по глубине скважины
        :param d_tub_out_func: объект расчет внешнего диаметра НКТ по глубине скважины
        :param d_cas_in_func: объект расчет внутреннего диаметра ЭК по глубине скважины
        :param heat_balance: опция учета теплопотерь

        :return: градиент давления в заданной точке трубы
        при заданных термобарических условиях, Па/м
        :return: градиент температуры в заданной точке трубы
        при заданных термобарических условиях, К/м
        """
        # Условие прекращения интегрирования
        if np.isnan(pt[0][0]) or pt[0][0] <= 0:
            return False

        p, t = pt[0]
        h = hh[0]

        # Отработка первого шага, когда еще нет предыдущей успешной глубины, давления и температуры
        if hh[1]:
            p_old, t_old = pt[1]
            h_prev = hh[1]
        else:
            p_old, t_old = pt[0]
            h_prev = hh[0]

        # Пересчет плотности газа на прошлом шаге
        old_fluid = deepcopy(self.fluid)
        old_fluid.calc_flow(p_old, t_old)
        rho_gas_prev = old_fluid.rg

        # Пересчет скорости газа на прошлом шаге
        vgas_prev = old_fluid.qg / (pi * self.d**2 / 4)

        # Вычисление диаметра
        if d_func is not None:
            self.d = d_func(h).item()
        if d_tub_out_func is not None:
            self.d_tub_out = d_tub_out_func(h).item()
        if d_cas_in_func is not None:
            self.d_cas_in = d_cas_in_func(h).item()

        # Пересчет PVT свойств на заданной глубине
        self.fluid.calc_flow(p, t)

        # Вычисление угла
        theta_deg = trajectory.calc_angle(h_prev, h)

        # Расчет градиента давления, используя необходимую гидравлическую корреляцию
        dp_dl = directions[0] * self.hydrcorr.calc_grad(
            theta_deg=theta_deg,
            eps_m=self.roughness,
            ql_rc_m3day=self.fluid.ql,
            qg_rc_m3day=self.fluid.qg,
            mul_rc_cp=self.fluid.mul,
            mug_rc_cp=self.fluid.mug,
            sigma_l_nm=self.fluid.stlg,
            rho_lrc_kgm3=self.fluid.rl,
            rho_grc_kgm3=self.fluid.rg,
            c_calibr_grav=holdup_factor,
            c_calibr_fric=friction_factor,
            h_mes=h,
            flow_direction=directions[0],
            vgas_prev=vgas_prev,
            rho_gas_prev=rho_gas_prev,
            h_mes_prev=h_prev,
            calc_acc=True,
            rho_mix_rc_kgm3=self.fluid.rm,
            p=p,
        )

        # Расчет распределения температуры с учетом теплопотерь
        if heat_balance:
            t_amb = float(amb_temp_dist.calc_temp(h))
            dt_dl = self.tempcorr.calc_grad(
                rho_n_kgm3=self.fluid.rm,
                dp_dl=dp_dl,
                d=self.hydrcorr.d,
                theta_deg=theta_deg,
                cp_n=self.fluid.heat_capacity_mixture,
                t_amb=t_amb,
                qm_rc_m3sec=self.fluid.qm,
                s_wall_tube=self.s_wall,
                t_prev=t_old,
                jt=self.fluid.calc_joule_thomson_coeff(t, p),
            )
        else:
            # Расчет геотермического градиента
            dt_dl = amb_temp_dist.calc_geotemp_grad(h)

        return dp_dl, dt_dl

    @staticmethod
    def __lower_limit(h, pt, *args):
        """
        Проверка на минимальное значение давления при интегрировании
        """

        return pt[0] - 90000

    def integrate_pipe(
        self,
        p0,
        t0,
        h0,
        h1,
        trajectory,
        amb_temp_dist,
        int_method,
        directions,
        friction_factor,
        holdup_factor,
        steps,
        d_func,
        d_tub_out_func,
        d_cas_in_func,
        heat_balance,
    ):
        """
        Метод для интегрирования давления, температуры в трубе

        Parameters
        ----------
        :param p0: начальное давление, Па
        :param t0: начальная температура, К
        :param h0: начальная глубина, м
        :param h1: граничная глубина, м
        :param trajectory: объект с траекторией
        :param amb_temp_dist: объект с распределением температуры породы
        :param int_method: метод интегрирования
        :param directions: множитель для направления потока, флаг направления расчета
        :param friction_factor: коэффициент поправки на трение
        :param holdup_factor: коэффициент поправки на истинное содержание жидкости
        :param steps: массив узлов для которых необходимо расcчитать давление
        :param d_func: диаметр НКТ, функция или число, м
        :param d_tub_out_func: внешний диаметр НКТ, функция или число, м
        :param d_cas_in_func: диаметр ЭК, функция или число, м
        :param heat_balance: опция учета теплопотерь


        Returns
        -------
        :return: массив глубин, м и давлений, Па

        """

        # Патч для solve_ivp, позволяет возвращать глубину, давление и температуру на прошлом успешном шаге
        patch.patch_solve_ivp()

        self.__lower_limit.terminal = True
        dptdl_integration = integr.solve_ivp(
            self.__integr_func,
            t_span=(h0, h1),
            y0=[p0, t0],
            method=int_method,
            args=(
                trajectory,
                amb_temp_dist,
                directions,
                holdup_factor,
                friction_factor,
                d_func,
                d_tub_out_func,
                d_cas_in_func,
                heat_balance,
            ),
            t_eval=steps,
            events=self.__lower_limit,
        )
        # Откатываем патч, что бы не было проблем с применением solve_ivp в других частях юнифлока
        patch.depatch_solve_ivp()
        return dptdl_integration
