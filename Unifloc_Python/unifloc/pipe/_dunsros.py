import numpy as np

import unifloc.pipe._hydrcorr as hr
import unifloc.service._constants as cnst


class DunsRos(hr.HydrCorr):
    """
    Класс гидравлической корреляции Duns & Ros
    """

    __slots__ = ["n_d", "ek", "a", "dr_const"]

    def __init__(self, d):
        super().__init__(d)
        self.n_d = None  # diameter number
        self.ek = None  # безразмерная кинетическая энергия
        self.a = None  # коэффициент переходного режима
        self.dr_const = cnst.DR_const

    @staticmethod
    def _calc_fp(
        n_gv: float,
        n_b_s: float,
        n_s_tr: float,
        n_tr_m: float,
    ) -> float:
        """
        Определение режима потока (flow pattern)
        Parameters
        ----------
        :param n_gv: gas velocity number
        :param n_b_s: граница между пузырьковым и пробковым режимами
        :param n_s_tr: граница между пробковым и переходным
        :param n_tr_m: граница между переходным и эмульсионным

        :return: номер режима потока, безразмерн.
            * 0 - Bubble flow pattern / Пузырьковый режим потока
            * 1 - Slug flow pattern / Пробковый режим потока
            * 2 - Transition flow pattern / Переходный режим потока
            * 3 - Mist flow pattern / Эмульсионный режим потока
        """

        if n_gv <= n_b_s:
            fp = 0
        elif n_b_s < n_gv <= n_s_tr:
            fp = 1
        elif n_s_tr < n_gv < n_tr_m:
            fp = 2
        else:
            fp = 3

        return fp

    def _calc_hl(
        self,
        rho_lrc_kgm3: float,
        sigma_l_nm: float,
        fp: float,
        vsm: float,
        vsl: float,
        n_gv: float,
        n_lv: float,
        n_l: float,
        n_d: float,
    ) -> float:
        """
        Расчет истинного содержания жидкости (liquid holdup)
        Parameters
        ----------
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Н/м

        :return: истинное содержание жидкости (liquid holdup)
        -------
        """
        if fp == 0:
            f1 = self.dr_const["f1"](n_l)
            f2 = self.dr_const["f2"](n_l)
            f3 = self.dr_const["f3"](n_l)
            f4 = self.dr_const["f4"](n_l)
            f3_hatch = f3 - f4 / n_d
            s = f1 + f2 * n_lv + f3_hatch * (n_gv / (1 + n_lv)) ** 2
        else:
            f5 = self.dr_const["f5"](n_l)
            f6 = self.dr_const["f6"](n_l)
            f7 = self.dr_const["f7"](n_l)
            f6_hatch = 0.029 * n_d + f6
            s = (1 + f5) * ((n_gv**0.982 + f6_hatch) / (1 + f7 * n_lv) ** 2)

        vs = s / (rho_lrc_kgm3 / (9.81 * sigma_l_nm)) ** 0.25
        hl = (vs - vsm + ((vsm - vs) ** 2 + 4 * vs * vsl) ** 0.5) / (2 * vs)

        return hl

    @staticmethod
    def _calc_rho_s(
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        fp: float,
        hl: float,
        ll: float,
        n_gv: float,
        n_tr_m: float,
        ek: float,
        a: float,
    ) -> float:
        """
        Расчет плотности смеси с учетом проскальзывания
        Parameters
        ----------
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param fp: номер режима потока, безразмерн.
            * 0 - Bubble flow pattern / Пузырьковый режим потока
            * 1 - Slug flow pattern / Пробковый режим потока
            * 2 - Transition flow pattern / Переходный режим потока
            * 3 - Mist flow pattern / Эмульсионный режим потока
        :param hl: истинное содержание жидкости (liquid holdup), безразмерн.
        :param ll: объемное содержание жидкости, безразмерн.
        :param n_gv: gas velocity number, безразмерн.
        :param n_tr_m: граница между переходным и эмульсионным режиммами, безразмерн.
        :param: ek: безразмерная кинетическая энергия, безразмерн.
        :param: a: коэффициент переходного режима, безразмерн.

        :return: плотность смеси с учетом проскальзывания, кг/м3
        -------
        """

        rho_s_1 = rho_lrc_kgm3 * hl + rho_grc_kgm3 * (1 - hl)

        if fp == 2:
            rho_grc_new = rho_grc_kgm3 * n_gv / n_tr_m
        else:
            rho_grc_new = rho_grc_kgm3

        rho_s_2 = (rho_lrc_kgm3 * ll + rho_grc_new * (1 - ll)) / (1 - ek)

        rho_s_kgm3 = rho_s_1 * a + rho_s_2 * (1 - a)

        return rho_s_kgm3

    def calc_grav(self, theta_deg: float, c_calibr_grav: float) -> float:
        """
        Функция, вычисляющая градиент давления на гравитацию
        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы
        :param c_calibr_grav: калибровочный коэффициент для слагаемого
                              градиента давления, вызванного гравитацие

        :return: градиент давления на гравитацию, Па/м
        -------
        """

        self.dp_dl_gr = self.rho_s_kgm3 * 9.81 * np.sin(theta_deg / 180 * np.pi) * c_calibr_grav

        return self.dp_dl_gr

    def _calc_ff_l(
        self,
        d: float,
        eps_m: float,
        rho_lrc_kgm3: float,
        mul_rc_cp: float,
        vsl: float,
        vsg: float,
        n_d: float,
    ) -> float:
        """
        Функция, вычисляющая коэффициент трения жидкой фазы по корреляции Duns & Ros (fp = 0, 1)
        Parameters
        ----------
        :param d: диаметр трубы, м
        :param eps_m: шероховатость стенки трубы, м
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        :param vsl: скорость жидкой фазы, м/с
        :param vsg: скорость газовой фазы, м/с
        :param n_d: diameter number, безразмерн.

        :return: коэффициент трения жидкой фазы, безразмерн.
        -------
        """

        rel_rough = eps_m / d

        # Расчет числа Рейнольдса
        n_re_l = self.fric.calc_n_re(d, rho_lrc_kgm3, vsl, mul_rc_cp)

        # Расчет коэффициента трения
        f1 = self.fric.calc_norm_ff(n_re_l, rel_rough, 1)

        coef = f1 / 4 * vsg / vsl * n_d ** (2 / 3)

        f2 = self.dr_const["f2_y"](coef)

        f3 = 1 + f1 / 4 * (vsg / (50 * vsl)) ** 0.5

        ff_l = f1 * f2 / f3

        return ff_l

    def _calc_ff_g(
        self,
        d: float,
        eps_m: float,
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        mul_rc_cp: float,
        mug_rc_cp: float,
        sigma_l_nm: float,
        vsg: float,
    ) -> float:
        """
        Функция, вычисляющая коэффициент трения газовой фазы по корреляции Duns & Ros (fp = 3)
        Parameters
        ----------
        :param d: диаметр трубы, м
        :param eps_m: шероховатость стенки трубы, м
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        :param mug_rc_cp: вязкость газа в P,T условиях, сПз
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Н/м
        :param vsg: скорость газовой фазы, м/с

        :return: коэффициент трения газовой фазы, безразмерн.
        -------
        """

        # Расчет числа Вебера
        n_we = rho_grc_kgm3 * vsg**2 * eps_m / sigma_l_nm

        # Расчет безразмерного показателя вязкости жидкости
        n_mu = (mul_rc_cp * 1e-3) ** 2 / rho_lrc_kgm3 / sigma_l_nm / eps_m

        # Расчет относительной шероховатости
        if n_mu * n_we <= 0.005:
            rel_rough_new = 0.0749 * sigma_l_nm / rho_grc_kgm3 / vsg**2 / d
        else:
            rel_rough_new = 0.3713 * sigma_l_nm * (n_mu * n_we) ** 0.302 / rho_grc_kgm3 / vsg**2 / d

        if rel_rough_new > 0.05:
            ff_g = 4 * ((4 * np.log10(0.27 * rel_rough_new)) ** (-2) + 0.067 * rel_rough_new**1.73)
        else:
            rel_rough = eps_m / d
            # Расчет числа Рейнольдса
            n_re_g = self.fric.calc_n_re(d, rho_grc_kgm3, vsg, mug_rc_cp)
            ff_g = self.fric.calc_norm_ff(n_re_g, rel_rough, 1)

        return ff_g

    def calc_fric(
        self,
        eps_m: float,
        mul_rc_cp: float,
        mug_rc_cp: float,
        c_calibr_fric: float,
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        sigma_l_nm: float,
        **kwargs
    ) -> float:
        """
        Функция, вычисляющая градиент давления на трение

        Parameters
        ----------
        :param eps_m: шероховатость стенки трубы, м
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        :param mug_rc_cp: вязкость газа в P,T условиях, сПз
        :param c_calibr_fric: калибровочный коэффициент для слагаемого
                              градиента давления, вызванного трением
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Н/м

        :return: градиент давления Па/м
        -------
        """

        if self.vsl == 0 and self.vsg == 0:
            self.dp_dl_fr = 0
        else:
            ff_l = self._calc_ff_l(self._d, eps_m, rho_lrc_kgm3, mul_rc_cp, self.vsl, self.vsg, self.n_d)

            grad_fric_1 = ff_l * rho_lrc_kgm3 * self.vsl * self.vsm / (2 * self._d)

            if self.fp == 2 or self.fp == 3:
                # Параметры для эмульсионного режима
                d_new = self._d - 2 * eps_m
                vsg_new = self.vsg * self.d**2 / d_new**2

                self.n_re = self.fric.calc_n_re(self._d, rho_grc_kgm3, vsg_new, mug_rc_cp)

                ff_g = self._calc_ff_g(
                    d_new,
                    eps_m,
                    rho_lrc_kgm3,
                    rho_grc_kgm3,
                    mul_rc_cp,
                    mug_rc_cp,
                    sigma_l_nm,
                    vsg_new,
                )
                grad_fric_2 = ff_g * rho_grc_kgm3 * vsg_new**2 / (2 * d_new) / (1 - self.ek)
            else:
                self.n_re = self.fric.calc_n_re(self._d, rho_lrc_kgm3, self.vsl, mul_rc_cp)

                grad_fric_2 = 0

            self.dp_dl_fr = (grad_fric_1 * self.a + grad_fric_2 * (1 - self.a)) * c_calibr_fric

        return self.dp_dl_fr

    def calc_params(
        self,
        theta_deg: float,
        ql_rc_m3day: float,
        qg_rc_m3day: float,
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        sigma_l_nm: float,
        p: float,
        mul_rc_cp: float,
        **kwargs
    ):
        """
        Метод расчета дополнительных параметров гидравлической корреляции Duns & Ros

        Parameters
        :param theta_deg: угол наклона трубы, градусы
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param qg_rc_m3day: расход газа в P,T условиях, м3/с
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Н/м
        :param p: текущее давление, Па
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        """

        if ql_rc_m3day == 0 and qg_rc_m3day == 0:
            # Случай нулевого дебита
            self.vsl = 0
            self.vsg = 0
            self.vsm = 0
            self.n_d = self._d * (rho_lrc_kgm3 * 9.81 / sigma_l_nm) ** 0.5
            self.fp = 0
            self.ll = 1
            self.rho_n_kgm3 = rho_lrc_kgm3 * self.ll + rho_grc_kgm3 * (1 - self.ll)
            self.hl = 1
            self.rho_s_kgm3 = self.rho_n_kgm3
            self.vl = 0
            self.vg = 0
        else:
            # Инициализация наклона трубы
            self.angle = theta_deg

            # Определение приведенных скоростей жидкости и газа
            self.vsl = ql_rc_m3day / (np.pi * self._d**2 / 4)
            self.vsg = qg_rc_m3day / (np.pi * self._d**2 / 4)

            # Определение скорости смеси
            self.vsm = self.vsl + self.vsg

            # Определение безразмерных коэффициентов
            n_gv = self.vsg * (rho_lrc_kgm3 / (9.81 * sigma_l_nm)) ** 0.25
            n_lv = self.vsl * (rho_lrc_kgm3 / (9.81 * sigma_l_nm)) ** 0.25
            self.n_d = self._d * (rho_lrc_kgm3 * 9.81 / sigma_l_nm) ** 0.5
            n_l = (mul_rc_cp * 1e-3) * (9.81 / (rho_lrc_kgm3 * sigma_l_nm**3)) ** 0.25

            # Безразмерные коэффициенты для границы между пузырьковым и пробковым режимами
            l1 = self.dr_const["l1"](self.n_d)
            l2 = self.dr_const["l2"](self.n_d)

            # Граница между пузырьковым и пробковым режимами
            n_b_s = l1 + l2 * n_lv
            # Граница между пробковым и переходным режимами
            n_s_tr = 50 + 36 * n_lv
            # Граница между переходным и эмульсионным режимами
            n_tr_m = 75 + 84 * n_lv**0.75

            # Определение режима потока (0, 1, 2, 3)
            self.fp = self._calc_fp(n_gv, n_b_s, n_s_tr, n_tr_m)

            # Определение коэффициента переходного режима
            if self.fp == 0 or self.fp == 1:
                self.a = 1
            elif self.fp == 2:
                self.a = (n_tr_m - n_gv) / (n_tr_m - n_s_tr)
            else:
                self.a = 0

            # Определение объемного содержания жидкости
            self.ll = max(ql_rc_m3day / (ql_rc_m3day + qg_rc_m3day), 0.000001)

            # Определение плотности смеси
            self.rho_n_kgm3 = rho_lrc_kgm3 * self.ll + rho_grc_kgm3 * (1 - self.ll)

            # Определение безразмерной кинетической энергии
            if self.fp == 2 or self.fp == 3:
                self.ek = self.vsm * self.vsg * self.rho_n_kgm3 / p
            else:
                self.ek = 0

            # Определение истинного содержания жидкости
            if self.fp != 3:
                hl = self._calc_hl(
                    rho_lrc_kgm3,
                    sigma_l_nm,
                    self.fp,
                    self.vsm,
                    self.vsl,
                    n_gv,
                    n_lv,
                    n_l,
                    self.n_d,
                )
                if self.fp != 2:
                    self.hl = hl
                else:
                    self.hl = hl * self.a + self.ll * (1 - self.a)
            else:
                hl = self.ll
                self.hl = hl

            # Определение плотности смеси с учетом проскальзывания
            self.rho_s_kgm3 = self._calc_rho_s(
                rho_lrc_kgm3,
                rho_grc_kgm3,
                self.fp,
                hl,
                self.ll,
                n_gv,
                n_tr_m,
                self.ek,
                self.a,
            )

            # Вычисление истинной скорости жидкости
            self.vl = self.vsl / self.hl if self.hl != 0 else 0

            # Вычисление истинной скорости газа
            self.vg = self.vsg / (1 - self.hl) if self.hl != 1 else 0
