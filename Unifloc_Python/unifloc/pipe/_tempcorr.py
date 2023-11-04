"""
Модуль, описывающий класс температурной корреляции
"""

import math as mt
from typing import Optional

import unifloc.service._constants as cnst


class TempCorr:
    """
    Класс для расчета температурной корреляции в форме градиента
    """

    @staticmethod
    def __relax_param(qm_rc_m3sec: float, rho_n_kgm3: float, d_tube_out: float, cp_n: float, u: float) -> float:
        """
        Функция расчета коэффициента релаксации

        Parameters
        ----------
        :param qm_rc_m3sec: дебит смеси, м3/с
        :param rho_n_kgm3: плотность смеси, кг/м3
        :param d_tube_out: внешний диаметр НКТ, м
        :param cp_n: теплоемкость смеси, Дж/(кг*К)
        :param u: коэффициент теплопередачи для системы "скважинный флюид - НКТ - затруб - обсадная колонна -
                                                                - цементное кольцо - горная порода", Дж/(с*K*м2)

        Returns
        -------
        :return: коэффициент релаксации, м
        """

        relax_param = cp_n * rho_n_kgm3 * qm_rc_m3sec / (mt.pi * d_tube_out * u)

        return relax_param

    def calc_grad(
        self,
        qm_rc_m3sec: float,
        rho_n_kgm3: float,
        dp_dl: float,
        d: float,
        theta_deg: float,
        t_amb: float,
        t_prev: float,
        s_wall_tube: float,
        cp_n: float,
        jt: float,
        u: Optional[float] = None,
    ) -> float:
        """
        Функция расчета градиента температуры по методу Kreith

        Parameters
        ----------
        :param qm_rc_m3sec: дебит смеси, м3/с
        :param rho_n_kgm3: плотность смеси, кг/м3
        :param dp_dl: градиент давления, Па/м
        :param d: диаметр НКТ, м
        :param theta_deg: угол наклона скважины, градусы
        :param t_amb: температура породы на заданной глубине, К
        :param t_prev: температура на предыдущем шаге, К
        :param s_wall_tube: толщина стенки НКТ, м
        :param cp_n: теплоемкость смеси, Дж/(кг*К)
        :param jt: коэффициент Джоуля-Томпсона, К/Па
        :param u: коэффициент теплопередачи для системы "скважинный флюид - НКТ - затруб - обсадная колонна -
                                                                - цементное кольцо - горная порода", Дж/(с*K*м2)

        Returns
        -------
        :return: градиент температуры, K/м
        """

        # Внешний диаметр НКТ
        d_tube_out = d + s_wall_tube * 2

        # Коэффициент общей теплопередачи
        if not u:
            u = cnst.TEMP_CORR["u"]

        # За счет эффекта Джоуля-Томпсона
        dt_dl_jt = dp_dl * jt

        # За счет теплопереноса
        dt_dl_ht = (t_prev - t_amb) / self.__relax_param(
            rho_n_kgm3=rho_n_kgm3,
            qm_rc_m3sec=qm_rc_m3sec,
            cp_n=cp_n,
            u=u,
            d_tube_out=d_tube_out,
        )

        # За счет изменения потенциальной энергии
        dt_dl_grav = 9.81 * mt.sin(theta_deg / 180 * mt.pi) / cp_n

        # Градиент температуры
        dt_dl = dt_dl_ht + dt_dl_jt + dt_dl_grav

        return dt_dl
