"""
Модуль, описывающий класс корреляции Беггса-Брилла
"""
import math as mt

import unifloc.pipe._hydrcorr as hr
import unifloc.tools.exceptions as exc


class BeggsBrill(hr.HydrCorr):
    """
    Класс расчета градиента давления и сопутствующих данных при помощи метода Beggs-Brill
    """

    __slots__ = ["n_fr"]

    def __init__(self, d):
        super().__init__(d)
        self.n_fr = None

    @staticmethod
    def __calc_psi(ll, n_fr, n_lv, theta_deg, d, e, f, g):
        """
        Расчет поправки на угол наклона

        Parameters
        ----------
        :param ll: объемное содержание жидкости
        :param n_fr: число Фруда
        :param n_lv: число скорости жидкости
        :param theta_deg: угол наклона к горизонтали
        :param d: коэффициент
        :param e: коэффициент
        :param f: коэффициент
        :param g: коэффициент

        Returns
        -------
        Поправка на угол наклона

        """
        cc = (1 - ll) * mt.log(max(d * ll**e * n_lv**f * n_fr**g, 0.000001))
        cc = max(cc, 0)
        yy = mt.sin(mt.pi / 100 * theta_deg)

        psi = 1 + cc * (yy - 0.333 * yy**3)
        return psi

    @staticmethod
    def _calc_fp(n_fr, ll):
        """
        Определение структуры потока по карте режимов потока

        Parameters
        ----------
        n_fr: Число Фруда, безразмерн.
        ll: Объемное содержание жидкости, безразмерн.

        :return: номер режима по карте режимов потока, безразмерн.
                режим потока:
                * 0 - расслоенный (segregated);
                * 1 - прерывистый (INTERMITTENT);
                * 2 - распределенный (distributed);
                * 3 - переходный (transition);
        -------
        """
        l1 = 316 * ll**0.302
        l2 = 0.0009252 / (ll**2.4684)
        l3 = 0.1 / (ll**1.4516)
        l4 = 0.5 / (ll**6.738)

        if (ll < 0.4 and n_fr >= l1) or (ll >= 0.4 and n_fr > l4):
            fp = 2
        elif (ll < 0.01 and n_fr < l1) or (ll >= 0.01 and n_fr < l2):
            fp = 0
        elif ll >= 0.01 and l3 >= n_fr >= l2:
            fp = 3
        elif (0.4 > ll >= 0.01 and l3 < n_fr <= l1) or (ll > 0.4 and l3 <= n_fr <= l4):
            fp = 1
        else:
            fp = 1
        return fp

    def _calc_hl(self, fp, ll, n_fr, n_lv, theta_deg):
        """
        Расчет истинного содержания жидкости (liquid holdup)

        Parameters
        ----------
        :param fp: режим потока:
            * 0 - расслоенный (segregated);
            * 1 - прерывистый (INTERMITTENT);
            * 2 - распределенный (distributed);
            * 3 - переходный (transition);
        :param ll: Объемное содержание жидкости, безразмерн.
        :param n_fr: число Фруда, безразмерн.
        :param n_lv: число скорости жидкости, безразмерн.
        :param theta_deg: угол наклона трубы, градусы

        :return: истинное содержание жидкости, безразмерн.
        """
        if fp in {0, 3}:
            # Истинное содержание жидкости в горизонтальной трубе
            h_l0 = 0.98 * ll**0.4846 / (n_fr**0.0868)

            h_l0 = min(h_l0, 1)
            h_l0 = max(ll, h_l0)

            # Истинное содержание с учетом угла наклона
            if theta_deg >= 0:  # Поток восходящий
                d = 0.011
                e = -3.768
                f = 3.539
                g = -1.614
            else:  # Поток нисходящий
                d = 4.70
                e = -0.3692
                f = 0.1244
                g = -0.5056

            # Расчет поправки на угол наклона
            psi = self.__calc_psi(ll, n_fr, n_lv, theta_deg, d, e, f, g)
            h_l = h_l0 * psi

            h_l = max(h_l, 0.2 * ll)
            h_l = min(h_l, 1)
            h_l_seg = h_l

        if fp in {1, 3}:
            # Истинное содержание жидкости в горизонтальной трубе
            h_l0 = 0.845 * ll**0.5351 / (n_fr**0.0173)

            h_l0 = min(h_l0, 1)
            h_l0 = max(ll, h_l0)

            # Истинное содержание с учетом угла наклона
            if theta_deg >= 0:  # Поток восходящий
                d = 2.96
                e = 0.305
                f = -0.4473
                g = 0.0978
            else:  # Поток нисходящий
                d = 4.70
                e = -0.3692
                f = 0.1244
                g = -0.5056

            # Расчет поправки на угол наклона
            psi = self.__calc_psi(ll, n_fr, n_lv, theta_deg, d, e, f, g)
            h_l = h_l0 * psi

            h_l = max(h_l, 0.2 * ll)
            h_l = min(1, h_l)

            h_l_int = h_l

        if fp == 3:
            l2 = 0.0009252 / ll**2.4684
            l3 = 0.1 / ll**1.4516
            aa = (l3 - n_fr) / (l3 - l2)
            h_l = aa * h_l_seg + (1 - aa) * h_l_int

        if fp == 2:
            # Истинное содержание жидкости в горизонтальной трубе
            h_l0 = 1.065 * ll**0.5824 / n_fr**0.0609

            h_l0 = min(h_l0, 1)
            h_l0 = max(ll, h_l0)
            # В данном режиме нет поправки на угол,
            # тк этот режим наблюдается только в горизонтальной трубе
            h_l = h_l0

        if fp == "undefined":
            raise exc.UniflocPyError("Режим потока неопределен. Проверьте исходные данные")

        # Поправка Payne et all
        if theta_deg >= 0:
            h_l *= 0.924
            h_l = max(ll, h_l)
        else:
            h_l *= 0.685
        return h_l

    def calc_grav(self, theta_deg, c_calibr_grav):
        """
        Метод расчета градиента давления в трубе с учетом гравитации по методике Беггза-Брилла

        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы
        :param c_calibr_grav: калибровочный коэффициент для слагаемого
                              градиента давления, вызванного гравитацией

        :return: градиент давления, Па/м
        -------
        """

        # Вычисление градиента давления с учетом гравитации
        self.dp_dl_gr = self.rho_s_kgm3 * 9.81 * mt.sin(theta_deg / 180 * mt.pi) * c_calibr_grav

        return self.dp_dl_gr

    def calc_fric(self, eps_m, ql_rc_m3day, mul_rc_cp, mug_rc_cp, c_calibr_fric, **kwargs):
        """
        Метод расчета градиента давления в трубе с учетом трения по методике Беггза-Брилла

        Parameters
        ----------
        :param eps_m: шероховатость стенки трубы, (м)
        :param ql_rc_m3day: дебит жидкости в P,T условиях, (м3/с)
        :param mul_rc_cp: вязкость жидкости в P,T условиях, (сПз)
        :param mug_rc_cp: вязкость газа в P,T условиях, (сПз)
        :param c_calibr_fric: калибровочный коэффициент для слагаемого
                              градиента давления, вызванного трением

        :return: градиент давления Па/м
        -------
        """

        if self.vsm == 0:
            # специально отработать случай нулевого дебита
            self.ff = 0
        else:
            roughness_d = eps_m / self.d  # относительная шероховатость (безразмерн.)

            # No slip mixture viscosity
            mu_n_c_p = mul_rc_cp * self.ll + mug_rc_cp * (1 - self.ll)

            # Расчет числа Рейнольдса
            self.n_re = self.fric.calc_n_re(self.d, self.rho_n_kgm3, self.vsm, mu_n_c_p)

            # Вычисление нормализированного коэффициента трения
            self.f_n = self.fric.calc_norm_ff(self.n_re, roughness_d, 1)

            # Вычисление поправки для коэффициента трения для
            # friction factor correction for multiphase flow
            y = max(self.ll / self.hl**2, 0.000001)
            fy = mt.log(max(y, 0.000001))
            if 1 < y < 1.2:
                s = mt.log(2.2 * y - 1.2)
            else:
                s = fy / (-0.0523 + 3.182 * fy - 0.8725 * fy**2 + 0.01853 * fy**4)

            # Расчет коэффициента трения
            self.ff = self.f_n * mt.exp(s)

        # Вычисление градиента давления с учетом трения
        self.dp_dl_fr = self.ff * self.rho_n_kgm3 * self.vsm**2 / (2 * self.d) * c_calibr_fric

        return self.dp_dl_fr

    def calc_params(self, theta_deg, ql_rc_m3day, qg_rc_m3day, rho_lrc_kgm3, rho_grc_kgm3, sigma_l_nm, **kwargs):
        """
        Метод расчета дополнительных параметров, необходимых для расчета градиента давления в трубе
        по методике Беггза-Брилла

        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param qg_rc_m3day: расход газа в P,T условиях, м3/с
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м
        """

        self.angle = theta_deg
        # TODO: v1.5.0 - упростить
        if ql_rc_m3day == 0 and qg_rc_m3day == 0:
            # специально отработать случай нулевого дебита
            self.ll = 0
            self.vsl = 0
            self.vsg = 0  # Пока = 0, для чисто газовых скважин нужна своя методика
            self.hl = 1
            rho_n_kgm3 = rho_lrc_kgm3 * self.ll + rho_grc_kgm3 * (1 - self.ll)
            self.fp = 0
            self.n_fr = 0
            self.vsm = 0
            self.vg = 0
        else:
            self.ll = max(ql_rc_m3day / (ql_rc_m3day + qg_rc_m3day), 0.000001)

            self.vsl = ql_rc_m3day / (3.1415926 * self.d**2 / 4)
            self.vsg = qg_rc_m3day / (3.1415926 * self.d**2 / 4)

            self.vsm = self.vsl + self.vsg

            # Расчет плотности смеси без учета проскальзывания
            rho_n_kgm3 = rho_lrc_kgm3 * self.ll + rho_grc_kgm3 * (1 - self.ll)

            # вычисление числа Фруда
            self.n_fr = self.vsm**2 / (9.81 * self.d)

            # вычисление числа скорости жидкости
            if rho_n_kgm3 is None or sigma_l_nm is None:
                n_lv = 0
            else:
                n_lv = self.vsl * (rho_lrc_kgm3 / (9.81 * sigma_l_nm)) ** 0.25

            # определение типа потока
            self.fp = self._calc_fp(self.n_fr, self.ll)

            # вычисление истинного содержания жидкости
            self.hl = self._calc_hl(self.fp, self.ll, self.n_fr, n_lv, theta_deg)

            # Вычисление истинной скорости газа
            self.vg = self.vsg / (1 - self.hl) if self.hl != 1 else 0

        # Расчет плотности смеси с учетом проскальзывания
        rho_s_kgm3 = rho_lrc_kgm3 * self.hl + rho_grc_kgm3 * (1 - self.hl)

        self.rho_n_kgm3 = rho_n_kgm3
        self.rho_s_kgm3 = rho_s_kgm3

        # Вычисление истинной скорости жидкости
        self.vl = self.vsl / self.hl if self.hl != 0 else 1
