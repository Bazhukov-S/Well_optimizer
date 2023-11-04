import math as mt

import scipy.constants as const
import scipy.optimize as sp

import unifloc.pipe._dunsros as dr


class HasanKabir(dr.DunsRos):
    """
    Класс гидравлической корреляции HasanKabir
    """

    __slots__ = [
        "stlg",
        "rl",
        "rg",
        "mul",
        "mug",
        "flow_pattern_name",
        "v_dt_msec",
        "h_ls",
        "h_lf",
        "len_s_m",
        "rho_mix_kgm3",
        "fc_s",
        "_d_tub_out",
        "_d_cas_in",
    ]

    def __init__(
        self,
        d: float,
        d_tub_out: float,
        d_cas_in: float,
    ) -> None:
        """
        :param d: диаметр трубы (в случае расчета затруба - гидравлический диаметр), м
        :param d_tub_out: внешний диаметр НКТ, м
        :param d_cas_in: внутренний диаметр ЭК, м
        """
        super().__init__(d)
        self._d_tub_out = d_tub_out  # внешний диаметр НКТ, м
        self._d_cas_in = d_cas_in  # внутренний диаметр ЭК, м

    @property
    def d_tub_out(self):
        return self._d_tub_out

    @d_tub_out.setter
    def d_tub_out(self, value):
        self._d_tub_out = value

    @property
    def d_cas_in(self):
        return self._d_cas_in

    @d_cas_in.setter
    def d_cas_in(self, value):
        self._d_cas_in = value

    def calc_params(
        self,
        theta_deg: float,
        ql_rc_m3day: float,
        qg_rc_m3day: float,
        mul_rc_cp: float,
        mug_rc_cp: float,
        sigma_l_nm: float,
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        rho_mix_rc_kgm3: float,
        **kwargs
    ):
        """
        Метод расчета дополнительных параметров, необходимых для расчета градиента давления в трубе
        по методике Hasan Kabir

        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param qg_rc_m3day: расход газа в P,T условиях, м3/с
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        :param mug_rc_cp: вязкость газа в P,T условиях, сПз
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param rho_mix_rc_kgm3: плотность ГЖС в P,T условиях, кг/м3
        ----------
        """
        if ql_rc_m3day == 0 and qg_rc_m3day == 0:
            # специально отработать случай нулевого дебита
            self.vsl = 0
            self.vsg = 0
            self.hl = 0
            self.rho_s_kgm3 = 0
            self.fc_s = 0
            self.v_dt_msec = 0
            self.vsm = 0
            self.vg = 0
            self.angle = 0
        else:
            self.angle = theta_deg
            self.stlg = sigma_l_nm
            self.rl = rho_lrc_kgm3
            self.rg = rho_grc_kgm3
            self.mul = mul_rc_cp
            self.mug = mug_rc_cp
            f_m2 = const.pi * ((self.d_cas_in / 2) ** 2 - (self.d_tub_out / 2) ** 2)
            self.vsg = qg_rc_m3day / f_m2
            self.vsl = ql_rc_m3day / f_m2
            self.vsm = (qg_rc_m3day + ql_rc_m3day) / f_m2

            self.calc_rho_mix(rho_mix_rc_kgm3)

    def _vmix_disp(self, x, fc: float):
        """
        Метод для определения критической скорости переходного режима по методике Caetano

        Parameters
        ----------
        :param x: искомая величина
        :param fc: коэффициент трения, безразмерн.

        :return: разницу между частями уравнения Caetano
        --------
        """

        right_part = (
            2
            * (x ** 1.2)
            * (fc ** 0.4)
            * ((2 / (self.d_cas_in - self.d_tub_out)) ** 0.4)
            * ((self.rl / self.stlg) ** 0.6)
            * (0.4 * self.stlg / ((self.rl - self.rg) * 9.81) ** 0.5)
        )
        left_part = 0.725 + 4.15 * (self.vsg / x) ** 0.5
        return right_part - left_part

    @staticmethod
    def _fc_gd(x, fca: float, n_re: float):
        """
        Метод для определения коэффициента трения в турбулентном течении по методике Ганна-Дарлинга


        Parameters
        ----------
        :param x: искомая величина
        :param fca: коэффициент трения Фаннинга
        :param n_re: число Рейнольдса

        :return: разницу между частями уравнения Гунна-Дарлинга
        --------
        Ref: Gunn, D.J. and Darling, C.W.W. (1963) Fluid Flow and Energy Losses in Non-Circular Conduits.
            Transactions of the Institution of Chemical Engineers, 41, 163-173.
        """
        right_part = (
            4
            * mt.log10(
                n_re
                * (x * (16 / fca) ** (0.45 * mt.exp(-(n_re - 3000) / (10 ** 6)))) ** 0.5
            )
            - 0.4
        )
        left_part = (
            1 / (x * (16 / fca) ** (0.45 * mt.exp(-(n_re - 3000) / 10 ** 6))) ** 0.5
        )
        return right_part - left_part

    def _calc_fp(self, rho_mix_rc_kgm3: float):
        """
        Расчет режима потока Flow Pattern

        Parameters
        ----------
        :param rho_mix_rc_kgm3: объемная плотность смеси, кг/м3
        ----------
        """
        v_d_msec = (
            1.53 * (9.81 * self.stlg * (self.rl - self.rg) / (self.rl) ** 2) ** 0.25
        )

        vsg_bs = ((1.2 * self.vsl + v_d_msec) / (4 - 1.2)) * mt.sin(
            self.angle * mt.pi / 180
        )

        vsg_an = (
            3.1 * (self.stlg * 9.81 * (self.rl - self.rg) / (self.rg) ** 2) ** 0.25 + 1
        ) * mt.sin(self.angle * mt.pi / 180)

        fc = self._fc(rho_mix_rc_kgm3)
        vsm_disp = sp.fsolve(self._vmix_disp, args=fc, x0=6, maxfev=13)
        self._set_flow_pattrn(vsg_bs, vsg_an, vsm_disp)

    def _set_flow_pattrn(
        self,
        vsg_bs: float,
        vsg_an: float,
        vsm_disp: float,
    ):
        """
        Метод для определения структуры потока
        Parameters:
        ----------
        :param vsg_bs: критическая скорость перехода в пробковый, м/с
        :param vsg_an: критическая скорость перехода в кольцевой, м/с
        :param vsm_disp: критическая скорость перехода в переходный режим, м/с
        ----------
        """
        if self.vsg <= vsg_bs:
            self.fp = 0
            self.flow_pattern_name = "Пузырьковый режим"
        elif self.vsg >= vsg_bs and (0.25 * self.vsg) < 0.52 and self.vsm < vsm_disp:
            self.fp = 2
            self.flow_pattern_name = "Пробковый режим"
        elif self.vsg >= vsg_bs and (0.25 * self.vsg) >= 0.52:
            self.fp = 3
            self.flow_pattern_name = "Вспененный режим"
        elif self.vsm >= vsm_disp:
            self.fp = 1
            self.flow_pattern_name = "Дисперсионно-пузырьковый режим"
        elif self.vsg >= vsg_an:
            self.fp = 4
            self.flow_pattern_name = "Кольцевой режим"

    def _fc(self, rho_mix: float) -> float:
        """
        Метод для расчета коэффициента трения

        Parameters:
        ----------
        :param rho_mix: плотность ГЖС, кг/м3

        :return: коэффициент трения, безразмерн.
        --------
        """
        k_ratio_d = self.d_tub_out / self.d_cas_in
        mu_mix_pasec = self.vsl / self.vsm * self.mul + self.vsg / self.vsm * self.mug
        n_re = self.fric.calc_n_re(
            self.d_cas_in - self.d_tub_out, rho_mix, self.vsm, mu_mix_pasec
        )
        fca = (
            16
            * (1 - k_ratio_d) ** 2
            / (
                (1 - k_ratio_d ** 4) / (1 - k_ratio_d ** 2)
                - (1 - k_ratio_d ** 2) / mt.log(1 / k_ratio_d)
            )
        )
        if n_re < 3000:
            fric = fca / n_re
        else:
            fric = sp.fsolve(
                HasanKabir._fc_gd,
                args=(fca, n_re),
                x0=0.021,
                maxfev=13,
            )
        return fric

    def _calc_bubbly(self):
        """
        Метод для расчета истинной объемной концентрации газа в пузырьковом режиме
        """
        v_d_msec = (
            1.53 * (9.81 * self.stlg * (self.rl - self.rg) / (self.rl) ** 2) ** 0.25
        )
        v_gas_msec = 1.2 * self.vsm + v_d_msec
        epsi = self.vsg / v_gas_msec
        self.hl = 1 - epsi
        self.rho_s_kgm3 = self.rl * (1 - epsi) + self.rg * epsi
        self.fc_s = self._fc(self.rho_s_kgm3)

    def _calc_slug(self):
        """
        Метод для расчета истинной объемной концентрации газа в пробковом режиме
        """
        self.v_dt_msec = (
            1.2 * (self.vsg + self.vsl)
            + 0.345
            * (9.81 * (self.d_tub_out + self.d_cas_in)) ** 0.5
            * mt.sin(self.angle * mt.pi / 180) ** 0.5
            * (1 + mt.cos(self.angle * mt.pi / 180)) ** 1.2
        )
        epsi_s = self.vsg / (1.2 * self.vsm + self.v_dt_msec)
        epsi_t = self.vsg / (1.15 * self.vsm + self.v_dt_msec)
        self.h_ls = 1 - epsi_s
        self.h_lf = 1 - epsi_t
        if self.vsg > 0.4:
            self.len_s_m = 0.1 / epsi_s
            epsi = (1 - self.len_s_m) * epsi_t + 0.1
        else:
            self.len_s_m = 0.25 * self.vsg / epsi_s
            epsi = (1 - self.len_s_m) * epsi_t + 0.25 * self.vsg
        self.hl = 1 - epsi
        rho_slug_kgm3 = self.rg * epsi_s + self.rl * (1 - epsi_s)
        self.rho_s_kgm3 = rho_slug_kgm3 * self.len_s_m
        self.fc_s = self._fc(rho_slug_kgm3)

    def _act_fl(self, len_ls: float, v_lls: float, v_gls: float) -> float:
        """
        Метод для вычисления фактический длины пленки жидкости в пробковом режиме

        Parameters:
        ----------
        :param len_ls: относительная длина промежуточного участка, д.ед
        :param v_lls: скорость жидкости в промежуточном участке, м/с
        :param v_gls: скорость газа в промежуточном участке, м/с

        :return: фактическую длину пленки жидкости в пробковом режиме, м
        --------
        """
        coef_b = (
            -2
            * (1 - self.vsg / self.v_dt_msec)
            * ((self.vsg - v_gls * (1 - self.h_ls)) / self.v_dt_msec)
            * len_ls
            + (2 / 9.81) * (self.v_dt_msec - v_lls) ** 2 * self.h_ls ** 2
        ) / (1 - self.vsg / self.v_dt_msec) ** 2
        coef_c = (
            ((self.vsg - v_gls * (1 - self.h_ls)) / self.v_dt_msec * len_ls)
            / (1 - self.vsg / self.v_dt_msec)
        ) ** 2
        discr = coef_b ** 2 - 4 * coef_c
        if discr > 0:
            x1 = (-coef_b + mt.sqrt(discr)) / 2
            x2 = (-coef_b - mt.sqrt(discr)) / 2
            if x1 >= 0 and x2 >= 0:
                resh = min(x1, x2)
            elif x1 < 0 and x2 < 0:
                resh = 0.000001
            else:
                resh = x1 if x1 >= 0 else x2
        elif discr == 0:
            resh = -coef_b / 2
        else:
            resh = 0.000001
        return resh

    def _acc_grad_p(self) -> float:
        """
        Метод для нахождения градиента давления на ускорения в пробковом режиме

        Parameters:
        ----------
        :return: градиент давления на ускорение в пробковом режиме, Па/м
        --------
        """

        len_ls = 16 * (self.d_cas_in - self.d_tub_out)
        len_su = len_ls / self.len_s_m
        v_lls = (self.vsl + self.vsg) - 1.53 * (
            (self.rl - self.rg) * 9.81 * self.stlg / (self.rl ** 2)
        ) ** 0.25 * self.h_ls ** 0.5 * (1 - self.h_ls)
        v_gls = (
            1.53
            * ((self.rl - self.rg) * 9.81 * self.stlg / (self.rl ** 2)) ** 0.25
            * self.h_ls ** 0.5
        ) + v_lls
        act_len_lf = self._act_fl(len_ls, v_lls, v_gls)
        v_llf = mt.fabs((9.81 * 2 * act_len_lf) ** 0.5 - self.v_dt_msec)
        grad_p_acc = (
            self.rl * (self.h_lf / len_su) * (v_llf - self.v_dt_msec) * (v_llf - v_lls)
        )
        grad_p_acc_res = grad_p_acc if grad_p_acc >= 0 else 0
        return grad_p_acc_res

    def _acc_grad_p_an(self) -> float:
        """
        Метод для расчета потерь на ускорения в кольцевом режиме потока

        Parameters:
        ----------
        :return: градиент давления на ускорение в кольцевом режиме потока, Па/м
        --------
        """
        v_dt_msec = (0.345 + 0.1 * (self.d_tub_out / self.d_cas_in)) * (
            (9.81 * self.d_cas_in * (self.rl - self.rg) / (self.rg)) ** 0.5
        )
        len_su = 1
        act_len_lf = len_su
        v_llf = (9.81 * 2 * act_len_lf) ** 0.5 - v_dt_msec
        grad_p_acc_an = self.rl * (self.hl / len_su) * (v_llf - v_dt_msec) * v_llf
        return grad_p_acc_an

    def _calc_hl(self):
        """
        Метод для вычисления концентрации газа в потоке кольцевой структуры
        """
        k_ratio_d = self.d_tub_out / self.d_cas_in
        angle_wt_average = (
            1
            / (1 - k_ratio_d ** 2)
            * (
                2 * mt.asin(k_ratio_d)
                + 2 * k_ratio_d * (1 - k_ratio_d ** 2) ** 0.5
                - const.pi * k_ratio_d ** 2
            )
        )
        t_ratio = angle_wt_average / ((2 * const.pi - angle_wt_average) * k_ratio_d)
        delta_i = 0.005 * t_ratio
        phi = 10 ** 4 * self.vsg * self.mug / self.stlg * (self.rg / self.rl) ** 0.5
        fe = 1 - mt.exp((phi - 1.5) * (-0.125))
        self.hl = (
            4
            / (self.d_cas_in * (1 - k_ratio_d ** 2))
            * (
                0.005 * (1 - 0.005 / self.d_cas_in)
                + delta_i * k_ratio_d * (1 + delta_i / self.d_tub_out)
                + self.vsl
                * fe
                / ((self.vsl * fe + self.vsg) * (1 - k_ratio_d ** 2))
                * (
                    1
                    - k_ratio_d ** 2
                    - 4 * 0.005 / self.d_cas_in * (1 - 0.005 / self.d_cas_in)
                    - 4
                    * delta_i
                    * k_ratio_d
                    / self.d_cas_in
                    * (1 + delta_i / self.d_tub_out)
                )
            )
        )
        epsi = 1 - self.hl
        self.rho_s_kgm3 = self.rl * (1 - epsi) + self.rg * epsi
        self.fc_s = self._fc(self.rho_s_kgm3)

    def calc_rho_mix(self, rho_mix_rc_kgm3: float):
        """
        Метод для расчета плотности смеси

        Parameters
        ----------
        :param rho_mix_rc_kgm3: объемная плотность смеси, кг/м3
        ----------
        """
        self._calc_fp(rho_mix_rc_kgm3)
        if self.fp == 0 or self.fp == 1:
            self._calc_bubbly()
        elif self.fp == 2 or self.fp == 3:
            self._calc_slug()
        elif self.fp == 4:
            self._calc_hl()

    def calc_fric(self, *args, **kwargs) -> float:
        """
        Метод расчета градиента давления в трубе с учетом трения по методике Хасана-Кабира

        Parameters
        ----------
        :return: градиент давлениядавления с учетом трения, Па/м
        --------
        """
        self.dp_dl_fr = (
            4 * self.fc_s / (self.d_cas_in - self.d_tub_out) * self.vsm ** 2 / 2
        ) * self.rho_s_kgm3
        return self.dp_dl_fr

    def calc_dp_dl_acc(self, **kwargs) -> float:
        """
        Метод расчета градиента давления в трубе с учетом инерции по методике Хасана-Кабира

        Parameters
        ----------
        :return: градиент давления с учетом инерции, Па/м
        --------
        """
        if self.fp == 0 or self.fp == 1:
            self.dp_dl_acc = 0
        elif self.fp == 2 or self.fp == 3:
            self.dp_dl_acc = self._acc_grad_p()
        elif self.fp == 4:
            self.dp_dl_acc = self._acc_grad_p_an()
        return self.dp_dl_acc
