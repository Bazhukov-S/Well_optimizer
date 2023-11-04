"""
Модуль, описывающий класс гидравлической корреляции Ансари
"""
import math as mt

import scipy.optimize as opt

import unifloc.pipe._dunsros as dr


class Ansari(dr.DunsRos):
    """
    Класс расчета градиента давления и сопутствующих данных при помощи корреляции Ansari
    """

    __slots__ = ["fe", "dp_dl_sc", "dd", "hg_ls"]

    def _calc_fp(
        self,
        sigma_l_nm: float,
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        mul_rc_cp: float,
        mug_rc_cp: float,
        eps_m: float,
    ) -> int:
        """
        Определение структуры потока по карте режимов потока

        Parameters
        ----------
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param mul_rc_cp: вязкость жидкости в P,T условиях, Па*с
        :param mug_rc_cp: вязкость газа в P,T условиях, Па*с
        :param eps_m: шероховатость стенки трубы, м

        :return: номер режима, безразмерн.
                режим потока:
                * 0 - пузырьковый;
                * 1 - пробковый;
                * 2 - кольцевой;
        -------
        """
        # Расчет критических параметров перехода между режимами
        vs = 1.53 * (9.81 * sigma_l_nm * (rho_lrc_kgm3 - rho_grc_kgm3) / rho_lrc_kgm3**2) ** 0.25
        vsg_slug = 0.25 * vs + 0.333 * self.vsl
        vsg_an = 3.1 * (9.81 * sigma_l_nm * (rho_lrc_kgm3 - rho_grc_kgm3) / rho_grc_kgm3**2) ** 0.25
        d_min = 19.01 * ((rho_lrc_kgm3 - rho_grc_kgm3) * sigma_l_nm / 9.81 / rho_lrc_kgm3**2) ** 0.5

        # Определение режима потока
        if vsg_slug <= self.vsg < vsg_an:
            fp = 1
        elif self.vsg >= vsg_an:
            fp = self._calc_annulus(
                sigma_l_nm,
                rho_lrc_kgm3,
                rho_grc_kgm3,
                mul_rc_cp,
                mug_rc_cp,
                eps_m,
                vs,
                d_min,
            )
        else:
            fp = 0

        return fp

    def _calc_annulus(
        self,
        sigma_l_nm: float,
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        mul_rc_cp: float,
        mug_rc_cp: float,
        eps_m: float,
        vs: float,
        d_min: float,
    ) -> int:
        """
        Метод для определение истинного режима потока

        Parameters
        ----------
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон*м
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        :param mug_rc_cp: вязкость газа в P,T условиях, сПз
        :param eps_m: шероховатость стенки трубы, м
        :param vs: cкорость проскальзывания, м/c
        :param d_min: критический диаметр трубы пузырькового режима, м

        :return: истинный номер режима, безразмерн.
                    режим потока:
                    * 0 - пузырьковый;
                    * 1 - пробковый;
                    * 2 - кольцевой;
        -------
        """
        # Расчет кольцевого режима
        v_krit = 10 * self.vsg * mug_rc_cp / (sigma_l_nm) * (rho_grc_kgm3 / rho_lrc_kgm3) ** 0.5
        self.fe = 1 - mt.exp(-0.125 * (v_krit - 1.5))
        if self.fe < 0:
            self.fe = 0
        elif self.fe > 1:
            self.fe = 1
        self.hl = self.fe * self.vsl / (self.fe * self.vsl + self.vsg)

        vsc = self.fe * self.vsl + self.vsg
        mu_sc = mul_rc_cp * self.hl + mug_rc_cp * (1 - self.hl)
        rho_c = rho_lrc_kgm3 * self.hl + rho_grc_kgm3 * (1 - self.hl)

        roughness_d = eps_m / self.d
        n_re_sc = self.fric.calc_n_re(rho_c, vsc, mu_sc, self.d)
        f_sc = self.fric.calc_norm_ff(n_re_sc, roughness_d, 1)
        # Вычисление градиента на трение по газу
        self.dp_dl_sc = f_sc * rho_c * vsc**2 / (2 * self.d)

        n_re_sl = self.fric.calc_n_re(rho_lrc_kgm3, self.vsl, mul_rc_cp, self.d)
        f_sl = self.fric.calc_norm_ff(n_re_sl, roughness_d, 1)

        # Вычисление градиента на трение по жидкости
        dp_dl_sl = f_sl * rho_lrc_kgm3 * self.vsl**2 / (2 * self.d)

        # Вычисление истинного содержания жидкости в жидкостной пленке
        n_re_f = self.fric.calc_n_re(rho_lrc_kgm3, self.vsl, mul_rc_cp, self.d) * (1 - self.fe)
        f_f = self.fric.calc_norm_ff(n_re_f, roughness_d, 1)
        b = (1 - self.fe) ** 2 * f_f / f_sl if self.fe < 0.9999 else 1
        xm = dp_dl_sl / self.dp_dl_sc * b
        ym = 9.81 * (rho_lrc_kgm3 - rho_c) * mt.sin(self.angle / 180 * mt.pi) / self.dp_dl_sc
        self.dd = min(
            self.newton(
                self._root_dd,
                x0=0.25,
                fprime=self._root_ddfev,
                tol=0.000000015,
                args=(rho_lrc_kgm3, rho_grc_kgm3, xm, ym),
                maxiter=500,
            ),
            0.499,
        )
        h_lf = 4 * self.dd * (1 - self.dd)

        h_krit = h_lf + self.hl * (self.d - 2 * self.d * self.dd) ** 2 / self.d**2
        vsl_slug = 3 * (self.vsg - 0.25 * vs * mt.sin(self.angle / 180 * mt.pi))

        # Определение истинного режима потока
        if h_krit < 0.12 and 0 < self.dd < 1 and self.fe != 1:
            fp = 2
        elif self.vsl <= vsl_slug or self.d < d_min:
            fp = 1
        else:
            fp = 0

        return fp

    def _calc_bubble(self, x: float, rho_lrc_kgm3: float, rho_grc_kgm3: float, sigma_l_nm: float) -> float:
        """
        Метод, описывающий функцию истинного содержания жидкости в пузырьковом режиме

        Parameters
        ----------
        :param x: истинное содержание жидкости, дол.ед
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м

        :return: истинное содержание жидкости, дол.ед
        """
        right_part = 1.53 * (9.81 * sigma_l_nm * (rho_lrc_kgm3 - rho_grc_kgm3) / rho_lrc_kgm3**2) ** 0.25 * x**0.5
        left_part = self.vsg / (1 - x) - 1.2 * self.vsm
        return right_part - left_part

    def _calc_hl(self, fp: int, rho_lrc_kgm3: float, rho_grc_kgm3: float, sigma_l_nm: float) -> float:
        """
        Расчет истинного содержания жидкости (liquid holdup)

        Parameters
        ----------
        :param fp: режим потока:
                * 0 - пузырьковый;
                * 1 - пробковый;
                * 2 - кольцевой;
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param qg_rc_m3day: расход газа в P,T условиях, м3/с
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м

        :return: истинное содержание жидкости, дол.ед
        """
        if fp == 0:
            h_l = float(
                opt.fsolve(
                    self._calc_bubble,
                    x0=0.99,
                    args=(rho_lrc_kgm3, rho_grc_kgm3, sigma_l_nm),
                )
            )
        elif fp == 1:
            # Определение скоростей элементов потока в пробковом режиме
            v_tb = 1.2 * self.vsm + 0.35 * (9.81 * self.d * (rho_lrc_kgm3 - rho_grc_kgm3) / rho_lrc_kgm3) ** 0.5
            self.hg_ls = self.vsg / (0.425 + 2.65 * self.vsm)
            hl_ls = 1 - self.hg_ls
            vg_ls = (
                1.2 * self.vsm
                + 1.53 * (9.81 * sigma_l_nm * (rho_lrc_kgm3 - rho_grc_kgm3) / rho_lrc_kgm3**2) ** 0.25 * hl_ls**0.5
            )
            a = self.hg_ls * v_tb + hl_ls * (
                self.vsm
                - self.hg_ls
                * (
                    1.53
                    * (sigma_l_nm * 9.81 * (rho_lrc_kgm3 - rho_grc_kgm3) / rho_lrc_kgm3**2) ** 0.25
                    * hl_ls**0.5
                )
            )
            hl_tb = float(
                opt.newton(
                    self._root_hltb,
                    x0=0.15,
                    fprime=self._root_hltbfev,
                    args=(v_tb, a),
                    rtol=1.48e-12,
                )
            )
            vl_tb = 9.916 * (9.81 * self.d * (1 - (1 - hl_tb) ** 0.5)) ** 0.5
            vl_ls = v_tb - ((v_tb + vl_tb) * hl_tb / hl_ls)
            if self.hg_ls > 0.25:
                vg_ls = vl_ls
            vg_tb = v_tb - (v_tb - vg_ls) * (1 - hl_ls) / (1 - hl_tb)

            # Определение истинного газосодержания
            beta2 = (vl_ls * hl_ls - self.vsl) / (vl_ls * hl_ls + vl_tb * hl_tb)
            beta1 = (self.vsg - vg_ls * self.hg_ls) / (vg_tb * (1 - hl_tb) - vg_ls * self.hg_ls)
            if abs(beta2 - beta1) > 0.11:
                beta = (beta1 + beta2) / 2
            else:
                beta = beta2

            # Вычисление плотности жидкостной пробки
            self.rho_n_kgm3 = rho_lrc_kgm3 * hl_ls + rho_grc_kgm3 * self.hg_ls
            h_l = 1 - beta
        elif fp == 2:
            h_l = self.hl
        return h_l

    @staticmethod
    def newton(
        func,
        x0: float,
        fprime: object,
        args: tuple = (),
        tol: float = 1.48e-8,
        maxiter: int = 50,
    ) -> float:
        """
        Функция для решения уравнений методом Ньютона-Рафсона

        :param func: исходная функция
        :param x0: нулевая точка
        :param fprime: производная исходной точки
        :param args: кортеж с аргументами, передаваемыми в функцию/производную
        :param tol: допустимая погрешность
        :param maxiter: максимальное кол-во итераций

        :return: искомое значение
        """
        for itr in range(maxiter):
            fval = func(x0, *args)
            if fval == 0:
                return x0
            fder = fprime(x0, *args)
            if fder == 0:
                return x0
            newton_step = fval / fder
            x = x0 - newton_step
            if abs(x0 - x) <= tol and abs((x0 - x) / x0) <= tol:
                return x
            x0 = x
        return x

    def _root_ddfev(self, x: float, rho_lrc_kgm3: float, rho_grc_kgm3: float, xm: float, *args) -> float:
        """
        Метод, описывающий производную функцию относительной толщины пленки жидкости в кольцевом режиме

        Parameters
        ----------
        :param x: относительная толщина пленки жидкости, безразмерн.
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param xm: модифицированный параметр Локхарта и Мартинелли, безразмерн.

        :return: f'(x)
        -------
        """
        if self.fe < 0.9:
            z = 1 + 24 * (rho_lrc_kgm3 / rho_grc_kgm3) ** 0.333 * x
            zdev = 24 * (rho_lrc_kgm3 / rho_grc_kgm3) ** 0.333
        else:
            z = 1 + 300 * x
            zdev = 300
        a = (4 * (1 - 2 * x))
        b = (4 * x * (1 - x))
        c = (1 - 4 * x * (1 - x))
        f = (
            z * a / (b * b) / (c * c * (c**0.5))
            - zdev / (4 * x) / (1 - x) / (c * c * (c**0.5))
            - 2.5 * z * 4 * (1 - 2 * x) / (4 * x) / (1 - x) / (c * c * c * (c**0.5))
            - 3 * xm * a / (b * b * b * b)
        )
        return f

    def _root_dd(self, x: float, rho_lrc_kgm3: float, rho_grc_kgm3: float, xm: float, ym: float) -> float:
        """
        Метод, описывающий функцию относительной толщины пленки жидкости в кольцевом режиме

        Parameters
        ----------
        :param x: относительная толщина пленки жидкости, безразмерн.
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param xm: модифицированный параметр Локхарта и Мартинелли, безразмерн.
        :param ym: модифицированный параметр Локхарта и Мартинелли, безразмерн.

        :return: f(x)
        -------
        """
        if x > 100000 or x < 0:
            x = 0.001
        if self.fe < 0.9:
            z = 1 + 24 * (rho_lrc_kgm3 / rho_grc_kgm3) ** 0.333 * x
            f = ym - z / (4 * x * (1 - x) * (1 - 4 * x * (1 - x)) ** 2.5) + xm / (4 * x * (1 - x)) ** 3
        else:
            z = 1 + 300 * x
            f = ym - z / (4 * x * (1 - x) * (1 - 4 * x * (1 - x)) ** 2.5) + xm / (4 * x * (1 - x)) ** 3

        return f

    def _root_hltb(self, x: float, v_tb: float, a: float) -> float:
        """
        Метод, описывающий функцию истинного содержания жидкости в пузырьке Тейлора

        Parameters
        ----------
        :param x: истинного содержания жидкости в пузырьке Тейлора, д.ед
        :param v_tb: скорость пузырька Тейлора, м/с
        :param a: числовой коэффициент

        :return: f(x)
        -------
        """

        f = 9.916 * (9.81 * self.d) ** 0.5 * (1 - (1 - x) ** 0.5) ** 0.5 * x - v_tb * (1 - x) + a
        return f

    def _root_hltbfev(self, x: float, v_tb: float, *args) -> float:
        """
        Метод, описывающий производную функции истинного содержания жидкости в пузырьке тейлора

        Parameters
        ----------
        :param x: истинного содержания жидкости в пузырьке Тейлора, д.ед
        :param v_tb: скорость пузырька Тейлора, м/с

        :return: f'(x)
        -------
        """
        f = v_tb + 9.916 * (9.81 * self.d) ** 0.5 * (
            (1 - (1 - x) ** 0.5) ** 0.5 + x / (4 * ((1 - x) * (1 - (1 - x) ** 0.5)) ** 0.5)
        )
        return f

    def calc_fric(
        self,
        eps_m: float,
        ql_rc_m3day: float,
        mul_rc_cp: float,
        mug_rc_cp: float,
        c_calibr_fric: float,
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        **kwargs
    ):
        """
        Метод расчета градиента давления в трубе с учетом трения по методике Ansari

        Parameters
        ----------
        :param eps_m: шероховатость стенки трубы, (м)
        :param ql_rc_m3day: дебит жидкости в P,T условиях, (м3/с)
        :param mul_rc_cp: вязкость жидкости в P,T условиях, (сПз)
        :param mug_rc_cp: вязкость газа в P,T условиях, (сПз)
        :param c_calibr_fric: калибровочный коэффициент для слагаемого
                              градиента давления, вызванного трением
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3

        :return: градиент давления Па/м
        -------
        """
        if self.vsm == 0:
            self.dp_dl_fr = 0
        else:
            roughness_d = eps_m / self.d
            if self.fp == 0:
                mu_n_c_p = mul_rc_cp * self.hl + mug_rc_cp * (1 - self.hl)
                self.n_re = self.fric.calc_n_re(self.rho_s_kgm3, self.vsm, mu_n_c_p, self.d)
                self.ff = self.fric.calc_norm_ff(self.n_re, roughness_d, 1)

                self.dp_dl_fr = self.ff * self.rho_s_kgm3 * self.vsm**2 / (2 * self.d) * c_calibr_fric
            elif self.fp == 1:
                mu_n_c_p = mul_rc_cp * (1 - self.hg_ls) + mug_rc_cp * self.hg_ls
                self.n_re = self.fric.calc_n_re(self.rho_n_kgm3, self.vsm, mu_n_c_p, self.d)
                self.ff = self.fric.calc_norm_ff(self.n_re, roughness_d, 1)
                self.dp_dl_fr = self.ff * self.rho_n_kgm3 * self.vsm**2 / (2 * self.d) * self.hl * c_calibr_fric

            elif self.fp == 2:
                if self.fe <= 0.9:
                    z = 1 + 24 * (rho_lrc_kgm3 / rho_grc_kgm3) ** 0.333 * self.dd
                else:
                    z = 1 + 300 * self.dd
                phi_c = z / (1 - 2 * self.dd) ** 5 if self.fe != 1 else 1

                self.dp_dl_fr = phi_c * self.dp_dl_sc * c_calibr_fric

        return self.dp_dl_fr

    def calc_params(
        self,
        theta_deg: float,
        ql_rc_m3day: float,
        qg_rc_m3day: float,
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        sigma_l_nm: float,
        mug_rc_cp: float,
        mul_rc_cp: float,
        eps_m: float,
        **kwargs
    ):
        """
        Метод расчета дополнительных параметров, необходимых для расчета градиента давления в трубе
        по методике Ансари

        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param qg_rc_m3day: расход газа в P,T условиях, м3/с
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м
        :param mug_rc_cp: вязкость газа в P,T условиях, (сПз)
        :param mul_rc_cp: вязкость жидкости в P,T условиях, (сПз)
        :param eps_m: шероховатость стенки трубы, м
        """
        self.angle = theta_deg

        if ql_rc_m3day == 0 and qg_rc_m3day == 0:
            self.vsl = 0
            self.vsg = 0
            self.hl = 1
            self.fp = 0
            self.vsm = 0
            self.rho_n_kgm3 = 0
        else:
            self.vsl = ql_rc_m3day / (3.1415926 * self.d**2 / 4)
            self.vsg = qg_rc_m3day / (3.1415926 * self.d**2 / 4)
            self.vsm = self.vsl + self.vsg

            self.rho_n_kgm3 = rho_lrc_kgm3

            # Определение режима потока
            self.fp = self._calc_fp(sigma_l_nm, rho_lrc_kgm3, rho_grc_kgm3, mul_rc_cp, mug_rc_cp, eps_m)

            # Вычисление истинного содержания жидкости
            self.hl = self._calc_hl(self.fp, rho_lrc_kgm3, rho_grc_kgm3, sigma_l_nm)

        # Расчет плотности смеси с учетом проскальзывания
        self.rho_s_kgm3 = self.rho_n_kgm3 * self.hl + rho_grc_kgm3 * (1 - self.hl)

        # Вычисление истинной скорости жидкости
        self.vl = self.vsl / self.hl if self.hl != 0 else 0
