"""
Модуль, описывающий класс модифицированной корреляции Грея
"""
import math as mt

import unifloc.pipe._hydrcorr as hr


class Gray(hr.HydrCorr):
    """
    Класс расчета градиента давления и сопутствующих данных при помощи метода Gray
    """

    __slots__ = ["r"]

    def __init__(self, d):
        """

        Parameters
        ----------
        :param d: диаметр трубы, м
        """
        super().__init__(d)
        self.r = None

    def _calc_fp(self, *args, **kwargs):
        """
        Расчет режима потока Flow Pattern

        В корреляции Gray не учитывается режим потока
        """

    def _calc_hl(self, rho_n_kgm3, sigma_l_nm, rho_lrc_kgm3, rho_grc_kgm3, r):
        """
        Расчет истинного содержания жидкости (liquid holdup)

        :param rho_n_kgm3: плотность смеси, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param r: параметр для расчета истинного содержания жидкости по корреляции Gray
        """
        nv = rho_n_kgm3**2 * self.vsm**4 / (9.81 * sigma_l_nm * (rho_lrc_kgm3 - rho_grc_kgm3))
        nd = 9.81 * (rho_lrc_kgm3 - rho_grc_kgm3) * self.d**2 / sigma_l_nm
        b = 0.0814 * (1 - 0.0554 * mt.log(1 + 730 * r / (r + 1)))
        # Результирующая корреляция для объемного содержания жидкости
        hl = 1 - (1 - mt.exp(-2.314 * (nv * (1 + 205 / nd)) ** b)) / (r + 1)
        return hl

    @staticmethod
    def calc_dp_dl_acc(rho_gas, rho_gas_prev, vgas, h_mes, h_mes_prev, *args, **kwargs):
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

    def calc_grav(self, theta_deg, c_calibr_grav):
        """
        Метод расчета градиента давления в трубе с учетом гравитации по методике Грея

        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы
        :param c_calibr_grav: калибровочный коэффициент для слагаемого
        градиента давления, вызванного гравитацией

        :return: градиент давления Па/м
        -------
        """

        # Вычисление градиента давления с учетом гравитации
        self.dp_dl_gr = self.rho_s_kgm3 * 9.81 * mt.sin(theta_deg / 180 * mt.pi) * c_calibr_grav

        return self.dp_dl_gr

    def calc_fric(
        self,
        eps_m,
        ql_rc_m3day,
        mul_rc_cp,
        mug_rc_cp,
        c_calibr_fric,
        rho_lrc_kgm3,
        rho_grc_kgm3,
        sigma_l_nm,
    ):
        """
        Метод расчета градиента давления в трубе с учетом трения по методике Грея

        Parameters
        ----------
        :param eps_m: шероховатость стенки трубы, м
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях , кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        :param mug_rc_cp: вязкость газа в P,T условиях, сПз
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м
        :param c_calibr_fric: калибровочный коэффициент для слагаемого
        градиента давления, вызванного трением

        :return: градиент давления, Па/м
        -------
        """
        mu_n_c_p = mul_rc_cp * self.ll + mug_rc_cp * (1 - self.ll)  # No slip mixture viscosity

        # Расчет числа Рейнольдса
        self.n_re = self.fric.calc_n_re(self.d, self.rho_n_kgm3, self.vsm, mu_n_c_p)

        if self.vsm == 0:
            self.ff = 1
        else:
            if self.vsl == 0:
                e_r = eps_m / self.d
            else:
                # e - Псевдошероховатость
                E1 = 28.5 * sigma_l_nm / (self.rho_n_kgm3 * self.vsm**2)
                if self.r >= 0.007:
                    e = min(E1, self.d)
                else:
                    e = min((eps_m + self.r * (E1 - eps_m) / 0.007), self.d)

                e_r = e / self.d if self.vsg != 0 else eps_m / self.d

            # Расчет коэффициента трения
            self.ff = self.fric.calc_norm_ff(self.n_re, e_r, 1)

        # Вычисление градиента давления с учетом трения
        self.dp_dl_fr = self.ff * self.rho_n_kgm3 * self.vsm**2 / (2 * self.d) * c_calibr_fric

        return self.dp_dl_fr

    def calc_params(self, theta_deg, ql_rc_m3day, qg_rc_m3day, rho_lrc_kgm3, rho_grc_kgm3, sigma_l_nm, **kwargs):
        """
        Метод расчета дополнительных параметров, необходимых для расчета градиента давления в трубе
        по методике Грея

        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param qg_rc_m3day: расход газа в P,T условиях, м3/с
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м

        -------
        """
        self.angle = theta_deg
        if ql_rc_m3day == 0:
            ll = 0
            self.vsl = 0
            self.vsg = qg_rc_m3day / (3.1415926 * self.d**2 / 4)
            self.vsm = self.vsl + self.vsg
            self.hl = 0
            self.vg = 0
            self.vl = 0
            self.r = 0
            # Расчет плотности смеси без учета проскальзывания
            rho_n_kgm3 = rho_lrc_kgm3 * ll + rho_grc_kgm3 * (1 - ll)
        else:
            if (ql_rc_m3day + qg_rc_m3day) > 0:
                ll = ql_rc_m3day / (ql_rc_m3day + qg_rc_m3day)
                if ll > 0.99:
                    ll = 1
            else:
                ll = 1

            self.vsl = ql_rc_m3day / (3.1415926 * self.d**2 / 4)
            self.vsg = qg_rc_m3day / (3.1415926 * self.d**2 / 4)
            self.vsm = self.vsl + self.vsg

            # Расчет плотности смеси без учета проскальзывания
            rho_n_kgm3 = rho_lrc_kgm3 * ll + rho_grc_kgm3 * (1 - ll)

            # вычисление истинного содержания жидкости по оригинальной формуле Грея
            if self.vsg == 0:
                self.hl = 1
                self.r = 1000
            else:
                if self.vsm > 0:
                    # dimensionless superficial liquid to gas ratio parameter
                    r = self.vsl / self.vsg  # if self.vsg != 0 else 1000
                else:
                    r = 1000

                self.r = r
                self.hl = self._calc_hl(rho_n_kgm3, sigma_l_nm, rho_lrc_kgm3, rho_grc_kgm3, r)

            # Вычисление истинной скорости жидкости
            self.vl = self.vsl / self.hl if self.hl != 0 else 0
            # Вычисление истинной скорости газа
            self.vg = self.vsg / (1 - self.hl) if self.hl != 1 else 0

        # Расчет плотности смеси с учетом проскальзывания
        rho_s_kgm3 = rho_lrc_kgm3 * self.hl + rho_grc_kgm3 * (1 - self.hl)

        self.ll = ll
        self.rho_n_kgm3 = rho_n_kgm3
        self.rho_s_kgm3 = rho_s_kgm3
