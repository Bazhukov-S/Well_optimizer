import math as mt

import numpy as np
from scipy.interpolate import interp1d

import unifloc.pipe._hydrcorr as hr
import unifloc.service._constants as cnst


class HagedornBrown(hr.HydrCorr):
    """
    Класс гидравлической корреляции Hagedorn&Brown
    """

    __slots__ = [
        "griffith",
        "phi_func",
        "n_lc_func",
        "hl_phi_func",
    ]

    def __init__(self, d: float):
        super().__init__(d)
        self.griffith = None

        # Интерполяционная функция по безразмерному числу Cb
        self.phi_func = interp1d(
            x=np.array(cnst.CB_HAGEDORN_BROWN_CONSTANTS),
            y=np.array(cnst.PHI_HAGEDORN_BROWN_CONSTANTS),
            fill_value="extrapolate",
            kind="quadratic",
        )

        # Интерполяционная функция по безразмерному числу n_l
        self.n_lc_func = interp1d(
            x=np.array(cnst.N_L_CONSTANTS),
            y=np.array(cnst.N_LC_CONSTANTS),
            fill_value="extrapolate",
            kind="quadratic",
        )

        self.hl_phi_func = interp1d(
            x=np.array(cnst.C_A_CONSTANTS),
            y=np.array(cnst.HL_PHI_CONSTANTS),
            fill_value="extrapolate",
            kind="quadratic",
        )

    def __repr__(self):
        return "HagedornBrown"

    def _calc_fp(self, *args, **kwargs):
        """
        Расчет режима потока Flow Pattern

        В корреляции не учитывается режим потока
        """

    @staticmethod
    def calc_dp_dl_acc(
        rho_gas: float, rho_gas_prev: float, vgas: float, h_mes: float, h_mes_prev: float, *args, **kwargs
    ):
        """
        Функция для вычисления градиента давления с учетом инерции

        :param rho_gas: плотность газа, кг/м3
        :param rho_gas_prev: плотность газа на предыдущем шагу, кг/м3
        :param vgas: скорость газа, м/c
        :param h_mes: глубина, м
        :param h_mes_prev: предыдущая глубина, м

        :return: градиент давления с учетом инерции, Па/м
        """
        d_h_mes = h_mes - h_mes_prev

        if d_h_mes == 0:
            dp_dl_acc = 0
        else:
            d_rho_n_kgm3 = 1 / rho_gas - 1 / rho_gas_prev
            dp_dl_acc = -(rho_gas**2) * vgas**2 * d_rho_n_kgm3 / d_h_mes
        return dp_dl_acc

    def _calc_hl(
        self,
        vsm: float,
        vsg: float,
        vsl: float,
        p: float,
        mul_rc_cp: float,
        rho_lrc_kgm3: float,
        sigma_l_nm: float,
    ):
        """
        Расчет истинного содержания жидкости (liquid holdup)

        Parameters
        ----------
        :param vsm: скорость смеси, м/с
        :param vsg: скорость газа, м/с
        :param vsl: скорость жидкости, м/с
        :param p: давление, Па
        :param mul_rc_cp: вязкость жидкости в P,T условиях, спз
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Н/м
        :return: истинное содержание жидкости, безразмерн.
        """
        a = 1.071 - (0.72769 * (vsm**2)) / self.d

        a = max(a, 0.13)

        b = vsg / vsm

        # Проверка условия использования поправки Гриффитса для пузырькового режима
        if (b - a) >= 0:  # Без поправки Гриффитса
            # Вычисление беразмерных параметров
            n_lv = vsl * (rho_lrc_kgm3 / (9.81 * sigma_l_nm)) ** 0.25
            n_gv = vsg * (rho_lrc_kgm3 / (9.81 * sigma_l_nm)) ** 0.25
            n_d = self.d * (rho_lrc_kgm3 * 9.81 / sigma_l_nm) ** 0.5
            n_l = mul_rc_cp * 0.001 * (9.81 / (rho_lrc_kgm3 * sigma_l_nm**3)) ** 0.25

            # Необходимые условия, в случае если число n_l вышло за рамки интерполяции
            n_l = max(n_l, 0.002)
            n_l = min(n_l, 0.5)

            n_lc_result = self.n_lc_func([n_l])
            n_lc = n_lc_result[0]

            # Интерполяция по безразмерному числу Ca
            ca = (n_lv / n_gv**0.575) * (n_lc / n_d) * (p / 101325) ** 0.1

            ca = max(ca, 0.000002)
            ca = min(ca, 0.01)

            hl_phi_result = self.hl_phi_func([ca])
            hl_phi = hl_phi_result[0]

            # Интерполяция по безразмерному числу Cb
            cb = (n_gv * (n_l**0.38)) / (n_d**2.14)
            cb = max(cb, 0.012)
            cb = min(cb, 0.089)

            phi_result = self.phi_func([cb])
            phi = phi_result[0]

            # Вычисление истинного содержания жидкости
            hl = hl_phi * phi

            # В случае если истинное содержание жидкости меньше объемного - истинное будет равно объемному
            hl = max(hl, self.ll)

            self.griffith = False

        else:  # Использование поправки Гриффитса
            vs = 0.24384  # Скорость восхождения пузырьков газа
            hl = 1 - (0.5 * (1 + vsm / vs - (((1 + vsm / vs) ** 2) - 4 * (vsg / vs)) ** 0.5))
            self.griffith = True

        return hl

    def calc_grav(self, theta_deg: float, c_calibr_grav: float = 1):
        """
        Метод расчета градиента давления в трубе с учетом гравитации по методике Хагедорна-Брауна

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
        Метод расчета градиента давления в трубе с учетом трения по методике Хагедорна-Брауна

        Parameters
        :param eps_m: шероховатость стенки трубы, м
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        :param mug_rc_cp: вязкость газа в P,T условиях, сПз
        :param c_calibr_fric: калибровочный коэффициент для слагаемого
                              градиента давления, вызванного трением
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3

        :return: градиент давления Па/м
        _______
        """

        roughness_d = eps_m / self.d
        mu_n_cp = mul_rc_cp**self.hl * mug_rc_cp ** (1 - self.hl)

        # Если не используется поправка Гриффитса
        if not self.griffith:
            # Вычисление числа Рейнольдса
            self.n_re = self.fric.calc_n_re(self.d, self.rho_n_kgm3, self.vsm, mu_n_cp)
            # Вычисление коэффициента трения
            self.ff = self.fric.calc_norm_ff(self.n_re, roughness_d, 1)

        # Если используется поправка Гриффитса
        elif self.griffith:
            self.n_re = self.fric.calc_n_re(self.d, self.rho_n_kgm3, self.vl, mu_n_cp)

            # Вычисление нормализированного коэффициента трения
            self.f_n = self.fric.calc_norm_ff(self.n_re, roughness_d, 1)
            self.ff = self.f_n

        # Вычисление градиента давления с учетом трения
        self.dp_dl_fr = (self.ff * self.rho_n_kgm3**2 * self.vsm**2) / (2 * self.d * self.rho_s_kgm3)

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
        Метод расчета дополнительных параметров, необходимых для расчета градиента давления в трубе
        по методике Хагедорна-Брауна

        Parameters
        :param theta_deg: угол наклона трубы, градусы
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param qg_rc_m3day: расход газа в P,T условиях, м3/с
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Н/м
        :param p: текущее давление, Па
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        """

        self.angle = theta_deg
        if ql_rc_m3day == 0 and qg_rc_m3day == 0:
            # специально отработать случай нулевого дебита
            self.ll = 1
            self.vsl = 0
            self.vsg = 0
            self.hl = 1
            rho_n_kgm3 = rho_lrc_kgm3 * self.ll + rho_grc_kgm3 * (1 - self.ll)
            self.vsm = 0
            self.rho_s_kgm3 = rho_n_kgm3
            self.ff = 0
            self.vg = 0
        else:
            self.ll = max(ql_rc_m3day / (ql_rc_m3day + qg_rc_m3day), 0.000001)

            # Расчет плотности смеси без учета проскальзывания
            rho_n_kgm3 = rho_lrc_kgm3 * self.ll + rho_grc_kgm3 * (1 - self.ll)
            self.vsl = ql_rc_m3day / (3.1415926 * self.d**2 / 4)
            self.vsg = qg_rc_m3day / (3.1415926 * self.d**2 / 4)
            self.vsm = self.vsl + self.vsg

            # Вычисление истинного содержания жидкости
            self.hl = self._calc_hl(self.vsm, self.vsg, self.vsl, p, mul_rc_cp, rho_lrc_kgm3, sigma_l_nm)

            # Вычисление истинной скорости газа
            self.vg = self.vsg / (1 - self.hl) if self.hl != 1 else 0

        # Расчет плотности смеси с учетом проскальзывания
        rho_s_kgm3 = rho_lrc_kgm3 * self.hl + rho_grc_kgm3 * (1 - self.hl)
        self.rho_n_kgm3 = rho_n_kgm3
        self.rho_s_kgm3 = rho_s_kgm3

        # Вычисление истинной скорости жидкости
        self.vl = self.vsl / self.hl if self.hl != 0 else 0
