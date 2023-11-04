"""
Модуль, описывающий класс абстрактной гидравлической стационарной корреляции
"""
from abc import ABC, abstractmethod
from typing import Optional, Union

import unifloc.pipe._friction as fr


class HydrCorr(ABC):
    """
    Класс описания гидравлической корреляции
    """

    __slots__ = [
        "hl",
        "dp_dl",
        "dp_dl_fr",
        "dp_dl_gr",
        "dp_dl_acc",
        "ff",
        "vsl",
        "vsg",
        "fp",
        "n_fr",
        "vsm",
        "ll",
        "rho_n_kgm3",
        "n_re",
        "f_n",
        "_d",
        "angle",
        "rho_s_kgm3",
        "vl",
        "vg",
        "fric",
    ]

    def __init__(self, d):
        """

        Parameters
        ----------
        :param d: диаметр трубы/НКТ, м
        """
        self.hl = None  # истинное содержание жидкости, безразмерн.
        self.dp_dl = None  # градиент давления суммарная, Па/м
        self.dp_dl_fr = None  # градиент давления из-за трения, Па/м
        self.dp_dl_gr = None  # градиент давления из-за силы тяжести, Па/м
        self.dp_dl_acc = None  # градиент давления из-за инерции, Па/м
        self.ff = None  # коэффициент трения, безразмерн.
        self.vsl = None  # приведенная скорость жидкости, м/с
        self.vsg = None  # приведенная скорость газа, м/с
        self.fp = None  # режим потока (0,1,2 или 3)
        self.n_fr = None  # число Фруда, безразмерн.
        self.vsm = None  # скорость смеси, м/с
        self.ll = None  # объемное содержание жидкости, безразмерн.
        self.rho_n_kgm3 = None
        self.n_re = None  # число Рейнольдса
        self.f_n = None  # однофазный коэффициент трения, безразмерн.
        self._d = d  # диаметр трубы, м
        self.angle = None
        self.rho_s_kgm3 = None
        self.vl = None
        self.vg = None
        self.fric = fr.Friction()

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d = value

    @staticmethod
    def calc_vmix(p: float, t: float, qg: float, d: float, z: Optional[float] = None) -> Union[float, None]:
        """
        Функция для вычисления скорости смеси в газоконденсатной скважине

        Parameters
        ----------
        :param p: давление, Па
        :param t: температура, К
        :param z: коэффициент сверхсжимаемости
        :param qg: дебит газа, м3/с
        :param d: внутренний диаметр НКТ, м

        :return: скорость смеси, м/с
        ----------
        """
        # Отработка случая нулевого дебита газа
        if z == 0 or z is None:
            return None
        vmix = 0.52 * t * z * qg / (10 * p * (d / 10) ** 2)
        return vmix

    @staticmethod
    def calc_vmix_krit(
        stlg: float, rho_liq: Optional[float] = None, rho_gas: Optional[float] = None
    ) -> Union[float, None]:
        """
        Функция для критической скорости смеси в газоконденсатной скважины, ниже которой не будет происходить вынос
        Формула Точигина

        Parameters
        ----------
        :param stlg: поверхностное натяжение, Н/м
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3

        :return: критическая скорость смеси, м/с
        ----------
        """
        # Отработка случая нулевого дебита газа/жидкости
        if rho_gas <= 0 or rho_liq <= 0 or rho_gas is None or rho_liq is None:
            return None
        else:
            vmix_krit = 3.3 * (9.81 * stlg * rho_liq * rho_liq / ((rho_liq - rho_gas) * rho_gas * rho_gas)) ** 0.25
            return vmix_krit

    @staticmethod
    def calc_vmix_krit_tern(
        stlg: float, rho_liq: Optional[float] = None, rho_gas: Optional[float] = None
    ) -> Union[float, None]:
        """
        Функция для критической скорости смеси в газоконденсатной скважины, ниже которой не будет происходить вынос
        Формула Тернера

        Parameters
        ----------
        :param stlg: поверхностное натяжение, Н/м
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3

        :return: критическая скорость смеси, м/с
        ----------
        """
        # Отработка случая нулевого дебита газа/жидкости
        if rho_gas <= 0 or rho_liq <= 0 or rho_gas is None or rho_liq is None:
            return None
        else:
            vmix_krit = 3.71 * (9.81 * stlg * (rho_liq - rho_gas) / (rho_gas * rho_gas)) ** 0.25
            return vmix_krit

    @staticmethod
    def calc_dp_dl_acc(rho_gas, vgas, vgas_prev, h_mes, h_mes_prev, *args, **kwargs):
        """
        Функция для вычисления градиента давления с учетом инерции

        :param rho_gas: плотность газа, кг/м3
        :param vgas: скорость газа, м/c
        :param vgas_prev: скорость газа на предыдущей глубине, м/c
        :param h_mes: глубина, м
        :param h_mes_prev: предыдущая глубина, м

        :return: градиент давления с учетом инерции, Па/м
        """

        # рассчитываем дельты
        d_h_mes = h_mes - h_mes_prev

        # на нулевом шаге градиент с учетом инерции будет равен нулю
        if d_h_mes == 0:
            dp_dl_acc = 0
        else:
            d_vsm_msec = vgas - vgas_prev
            dp_dl_acc = -rho_gas * vgas * d_vsm_msec / d_h_mes

        return dp_dl_acc

    @abstractmethod
    def _calc_fp(self, *args, **kwargs):
        """
        Расчет режима потока Flow Pattern
        """

    @abstractmethod
    def _calc_hl(self, *args, **kwargs):
        """
        Расчет истинного содержания жидкости (liquid holdup)
        """

    @abstractmethod
    def calc_grav(self, *args, **kwargs):
        """
        Расчет градиента давления с учетом гравитации
        """

    @abstractmethod
    def calc_fric(self, *args, **kwargs):
        """
        Расчет градиента давления с учетом трения
        """

    @abstractmethod
    def calc_params(self, *args, **kwargs):
        """
        Расчет параметров, необходимых для расчета градиента давления
        """

    def calc_grad(
        self,
        theta_deg: float,
        eps_m: float,
        ql_rc_m3day: float,
        qg_rc_m3day: float,
        mul_rc_cp: float,
        mug_rc_cp: float,
        sigma_l_nm: float,
        rho_lrc_kgm3: float,
        rho_grc_kgm3: float,
        c_calibr_grav: float = 1,
        c_calibr_fric: float = 1,
        h_mes: float = 0,
        h_mes_prev: float = 0,
        vgas_prev: float = None,
        rho_gas_prev: float = None,
        flow_direction: float = 1,
        calc_acc: float = False,
        rho_mix_rc_kgm3: float = None,
        p: float = None,
    ):
        """
        Расчет градиента потерь давления

        Parameters
        ----------
        :param theta_deg: угол наклона трубы, градусы
        :param eps_m: шероховатость стенки трубы, м
        :param ql_rc_m3day: дебит жидкости в P,T условиях, м3/с
        :param qg_rc_m3day: расход газа в P,T условиях, м3/с
        :param mul_rc_cp: вязкость жидкости в P,T условиях, сПз
        :param mug_rc_cp: вязкость газа в P,T условиях, сПз
        :param sigma_l_nm: коэффициент поверхностного натяжения жидкость-газ, Ньютон/м
        :param rho_lrc_kgm3: плотность жидкости в P,T условиях, кг/м3
        :param rho_grc_kgm3: плотность газа в P,T условиях, кг/м3
        :param c_calibr_grav: калибровочный коэффициент для слагаемого
                              градиента давления, вызванного гравитацией
        :param c_calibr_fric: калибровочный коэффициент для слагаемого
                              градиента давления, вызванного трением
        :param h_mes: измеренная глубина, в которой измерены давление и температура, м
        :param h_mes_prev: измеренная глубина на предыдущем шаге,
                           в которой измерены давление и температура м
        :param vgas_prev: скорость газа на предыдущем шаге, м/с
        :param rho_gas_prev:  плотность газа на предыдущем шаге, кг/м3
        :param flow_direction: множитель на направление потока
        :param calc_acc: флаг расчета градиента на ускорение
        :param rho_mix_rc_kgm3: плотность смеси в P, T условиях, кг/м3
        :param p: текущее давление, Па

        :return: градиент давления, Па/м
        -------
        """

        # Вычисление параметров, необходимых для расчета градиента давления
        self.calc_params(
            theta_deg=theta_deg,
            ql_rc_m3day=ql_rc_m3day,
            qg_rc_m3day=qg_rc_m3day,
            rho_lrc_kgm3=rho_lrc_kgm3,
            rho_grc_kgm3=rho_grc_kgm3,
            sigma_l_nm=sigma_l_nm,
            p=p,
            mul_rc_cp=mul_rc_cp,
            c_calibr_grav=c_calibr_grav,
            rho_mix_rc_kgm3=rho_mix_rc_kgm3,
            mug_rc_cp=mug_rc_cp,
            eps_m=eps_m,
        )

        # Вычисление градиента давления с учетом гравитации
        self.dp_dl_gr = self.calc_grav(theta_deg, c_calibr_grav) * flow_direction

        # Вычисление градиента давления с учетом трения
        self.dp_dl_fr = self.calc_fric(
            eps_m=eps_m,
            ql_rc_m3day=ql_rc_m3day,
            mul_rc_cp=mul_rc_cp,
            mug_rc_cp=mug_rc_cp,
            c_calibr_fric=c_calibr_fric,
            rho_lrc_kgm3=rho_lrc_kgm3,
            rho_grc_kgm3=rho_grc_kgm3,
            sigma_l_nm=sigma_l_nm,
        )

        # Вычисление градиента давления с учетом инерции
        if calc_acc:
            self.dp_dl_acc = self.calc_dp_dl_acc(
                rho_gas=rho_grc_kgm3,
                vgas=self.vsg,
                h_mes=h_mes,
                h_mes_prev=h_mes_prev,
                vgas_prev=vgas_prev,
                rho_gas_prev=rho_gas_prev,
            )
        else:
            self.dp_dl_acc = 0

        self.dp_dl = self.dp_dl_gr + self.dp_dl_fr + self.dp_dl_acc

        return self.dp_dl
