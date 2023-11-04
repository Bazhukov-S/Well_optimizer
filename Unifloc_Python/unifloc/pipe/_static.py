"""
Модуль, для расчета градиентов остановленной скважины
"""
import math as mt

from unifloc.pipe import _hydrcorr as hr


class Static(hr.HydrCorr):
    def __init__(self, d):
        """
        Класс статического метода расчета градиента давления и сопутствующих данных

        Parameters
        ----------
        :param d: диаметр НКТ, м
        """
        super().__init__(d)

        self.calc_step = 0
        self.calc_type = None

    def _calc_fp(self, *args, **kwargs):
        pass

    def _calc_hl(self, *args, **kwargs):
        pass

    def calc_fric(self, *args, **kwargs):
        self.dp_dl_fr = 0
        return self.dp_dl_fr

    def calc_dp_dl_acc(self, *args, **kwargs):
        self.dp_dl_acc = 0
        return self.dp_dl_acc

    def calc_grav(self, theta_deg, *args):
        """
        Статический метод расчета градиента давления в трубе с учетом гравитации

        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы

        :return: градиент давления, Па/м
        -------
        """

        # Вычисление градиента давления с учетом гравитации для жидкости в затрубном пространстве
        self.dp_dl_gr = self.rho_s_kgm3 * 9.81 * mt.sin(theta_deg / 180 * mt.pi)

        return self.dp_dl_gr

    def calc_params(
        self,
        theta_deg,
        ql_rc_m3day,
        qg_rc_m3day,
        rho_lrc_kgm3,
        rho_grc_kgm3,
        rho_mix_rc_kgm3,
        sigma_l_nm,
        c_calibr_grav,
        **kwargs
    ):
        """
        Метод расчета дополнительных параметров, необходимых для статического метода
        расчета градиента давления в трубе

        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param qg_rc_m3day: расход газа в P,T условиях, м3/с
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param rho_mix_rc_kgm3: плотность смеси в P,T условиях, кг/м3
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м
        :param c_calibr_grav: калибровочный коэффициент для слагаемого
        градиента давления,вызванного гравитацией
        -------
        """
        self.rho_s_kgm3 = rho_mix_rc_kgm3
