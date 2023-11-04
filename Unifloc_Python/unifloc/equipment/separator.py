"""
Модуль, описывающий класс по работе с сепаратором
"""
import json
import os
from functools import cached_property
from typing import Optional

import unifloc.equipment.equipment as eq


class Separator(eq.Equipment):
    """
    Класс для расчета общего коэффициента сепарации газа на насосе
    c учетом газосепаратора и естественной сепарации
    """

    def __init__(
        self,
        h_mes: float,
        k_gas_sep: float = 0,
        sep_name: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        :param h_mes: глубина установки газосепаратора, м - float
        :param k_gas_sep: коэффициент сепарации насоса (газосепаратора), д.ед - float, optional
        :param sep_name: название сепаратора в соответствии с БД - str, optional

        Examples:
        --------
        >>> from unifloc.equipment.natural_separation import NaturalSeparation
        >>> from unifloc.equipment.separator import Separator
        >>> # Расчет естественной сепарации
        >>> # Инициализация исходных данных для расчета естественной сепарации
        >>> h_mes_ = 1800
        >>> d_tub = 0.063
        >>> d_cas = 0.130
        >>> q_fluid = 100 / 86400
        >>> q_gas = 0
        >>> sigma_l = 24
        >>> rho_liq = 836
        >>> rho_gas = 0.84
        >>> # Инициализация класса расчета естественной сепарации
        >>> k_sep_nat_calc = NaturalSeparation(h_mes_)
        >>> # Расчет коэффициента естественной сепарации
        >>> k_sep_natural = k_sep_nat_calc.calc_separation(d_tub, d_cas, q_fluid, q_gas,
        ...                                                sigma_l, rho_liq, rho_gas)
        >>> # Инициализация исходных данных для расчета общей сепарации
        >>> k_gas_sep = 0.7
        >>> # Инициализация класса расчета общей сепарации
        >>> k_gassep_calc = Separator(h_mes_, k_gas_sep)
        >>> # Расчет коэффициента общей сепарации
        >>> k_sep_total = k_gassep_calc.calc_general_separation(k_sep_natural)
        """

        super().__init__(h_mes)

        self.num_sep = None if sep_name is None else self._get_separator(sep_name)
        self.k_gas_sep = k_gas_sep
        self.k_gassep_general = 0
        self.__seps_db = None

    @cached_property
    def seps_db(self) -> list:
        """
        Метод подгрузки БД сепараторов

        Parameters
        ----------
        :return: БД сепараторов
        -------
        """
        with open(os.path.join(os.path.dirname(__file__), "seps.json"), encoding="utf-8") as jsonFile:
            self.__seps_db = json.load(jsonFile)
        return self.__seps_db

    def calc_general_separation(
        self,
        k_gas_sep_nat: float,
        gf: Optional[float] = None,
        q_liq: Optional[float] = None,
        freq: Optional[float] = None,
    ) -> float:
        """
        Функция расчета общего коэффициента сепарации с учетом
        газосепаратора и естественной сепарации

        Parameters
        ----------
        :param k_gas_sep_nat: коэффициент естественной сепарации, д.ед
        :param gf: доля газа на приеме ГС, д.ед.
        :param q_liq: дебит жидкости, м3/с
        :param freq: частота вращения вала ЭЦН, Гц

        :return k_gassep_general: коэффициент общей сепарации, д.ед
        -------
        """
        # Коэффициент газосепарации газосепаратора
        if self.num_sep is not None and self.k_gas_sep == 0 and gf is not None:
            self.k_gas_sep = self.calc_k_gas_sep(self.num_sep, gf=gf, q_liq=q_liq, freq=freq)

        # Общий коэффициент газосепарации
        self.k_gassep_general = k_gas_sep_nat + (1 - k_gas_sep_nat) * self.k_gas_sep

        return self.k_gassep_general

    def _get_separator(self, sep_name: str) -> str:
        """
        Метод определения соответствия сепаратора БД

        Parameters
        ----------
        :param sep_name: название сепаратора(диспергатора), в соответсвии с БД

        :return: номер газосепаратора в соответсвие с БД
        -------
        """
        for key in self.seps_db[0].keys():
            for i in self.seps_db[0][key]:
                if i == sep_name:
                    return key
        else:
            return "29"

    @staticmethod
    def _sepfactor_approx_qgf(theta_array: list, gf: float, qu: float, f: float) -> float:
        """
        Метод вычисления значению апроксимирующей функции

        Parameters
        ----------
        :param theta_array: список с численными коэффициентами
        :param gf: доля газа на входе в газосепаратор, %
        :param qu: дебит жидкости, м3/сутки
        :param f: число оборотов вала ГС(ЭЦН) в минуту

        :return func: значение апроксимирующей функции
        -------
        """
        func = (
            theta_array[0] * gf**4
            + theta_array[1] * gf**3
            + theta_array[2] * gf**2 * qu
            + theta_array[3] * gf**2
            + theta_array[4] * gf * qu
            + theta_array[5] * gf * f
            + theta_array[6] * qu**2
            + theta_array[7] * qu * f
            + theta_array[8] * gf
            + theta_array[9] * qu
            + theta_array[10] * f
            + theta_array[11]
        )
        return func

    @staticmethod
    def _sepfactor_approx_gf(theta_array: list, gf: float, qu: float, f: float) -> float:
        """
        Метод вычисления производной по газосодержанию апроксимирующей функции

        Parameters
        ----------
        :param theta_array: список с численными коэффициентами
        :param gf: доля газа на входе в газосепаратор, %
        :param qu: дебит жидкости, м3/сутки
        :param f: число оборотов вала ГС(ЭЦН) в минуту

        :return dfunc_gf: производная по газосодержанию апроксимирующей функции, дол.ед
        -------
        """
        dfunc_gf = (
            4 * theta_array[0] * gf**3
            + 3 * theta_array[1] * gf**2
            + 2 * theta_array[2] * gf * qu
            + 2 * theta_array[3] * gf
            + theta_array[4] * qu
            + theta_array[5] * f
            + theta_array[8]
        )
        return dfunc_gf

    @staticmethod
    def _sepfactor_approx_q(theta_array: list, gf: float, qu: float, f: float) -> float:
        """
        Метод вычисления производной по дебиту апроксимирующей функции

        Parameters
        ----------
        :param theta_array: список с численными коэффициентами
        :param gf: доля газа на входе в газосепаратор, %
        :param qu: дебит жидкости, м3/сутки
        :param f: число оборотов вала ГС(ЭЦН) в минуту

        :return dfunc_qu: производная по дебиту апроксимирующей функции, дол.ед
        -------
        """
        dfunc_qu = (
            theta_array[2] * gf**2
            + theta_array[4] * gf
            + 2 * theta_array[6] * qu
            + theta_array[7] * f
            + theta_array[9]
        )
        return dfunc_qu

    def calc_k_gas_sep(
        self,
        num_sep: str,
        gf: float,
        q_liq: float,
        freq: float = 50,
        k_gas_sep: float = 0.7,
    ) -> float:
        """
        Вычисление коэффициента сепрации газосепаратора в точке

        Parameters
        ----------
        :param num_sep:  номер газосепаратора в соответсвие с БД
        :param gf: доля газа, д.ед.
        :param q_liq: дебит жидкости, м3/с
        :param freq: частота вращения вала ЭЦН, Гц
        :param k_gas_sep:  коэффициент сепарации газосепаратора(паспортный), д.ед

        :return my_sepfactor: коэффициент сепарации газосепаратора(расчетный), д.ед
        -------
        """
        # приводим данные в размерности для расчета
        gf = gf * 100
        qu = q_liq * 86400
        f = freq * 60

        theta_array = self.seps_db[1][num_sep]
        sep_par = self.seps_db[2][num_sep]

        my_gfmin = sep_par[0]
        my_gfmax = sep_par[1]
        my_qmin = sep_par[2]
        my_qmax = sep_par[3]

        if gf > 100 or qu > 1.3 * my_qmax:
            return 0

        if my_gfmin <= gf <= my_gfmax:
            if my_qmin <= my_qmax <= my_qmax:
                my_sepfactor = self._sepfactor_approx_qgf(theta_array, gf, qu, f)
            # Если точка лежит вне замеренной области, используем сшивки
            # (1) Сшивка с параболическим цилиндром только по дебиту в области нуля
            elif qu < my_qmax:
                func = self._sepfactor_approx_qgf(theta_array, gf, my_qmax, f)
                dfunc = self._sepfactor_approx_q(theta_array, gf, my_qmax, f)
                ka = dfunc / my_qmax
                kb = -1 * dfunc
                kc = func
                my_sepfactor = ka * qu**2 + kb * qu + kc
            # (2) Сшивка с параболическим цилиндром только по дебиту в области максимума
            elif qu > my_qmax:
                # mQ2
                mX2 = 1.3 * my_qmax
                func = self._sepfactor_approx_qgf(theta_array, gf, my_qmax, f)
                dfunc = self._sepfactor_approx_q(theta_array, gf, my_qmax, f)
                r = my_qmax - mX2
                ka = (dfunc * r - func) / r / r
                kb = dfunc - 2 * ka * my_qmax
                kc = -ka * mX2**2 - kb * mX2
                my_sepfactor = ka * qu**2 + kb * qu + kc
        # (3) Сшивка с параболическим цилиндром только по газу в области нуля
        elif my_qmin <= my_qmax <= my_qmax:
            if gf < my_gfmin:
                func = self._sepfactor_approx_qgf(theta_array, my_gfmin, qu, f)
                dfunc = self._sepfactor_approx_gf(theta_array, my_gfmin, qu, f)
                ka = (my_gfmin * dfunc - func) / my_gfmin / my_gfmin
                kb = dfunc - 2 * my_gfmin * ka
                my_sepfactor = ka * gf**2 + kb * gf
            # (4) Сшивка с параболическим цилиндром только по газу в области максимума
            elif gf > my_gfmax:
                func = self._sepfactor_approx_qgf(theta_array, my_gfmax, qu, f)
                dfunc = self._sepfactor_approx_gf(theta_array, my_gfmax, qu, f)
                r = my_gfmax - 100
                ka = (dfunc * r - func) / r / r
                kb = dfunc - 2 * ka * my_gfmax
                kc = -ka * 100**2 - kb * 100
                my_sepfactor = ka * gf**2 + kb * gf + kc
        # (5) Сшивка с параболоидом вращения по газу в области нуля и по дебиту в области нуля
        elif gf < my_gfmin:
            if qu < my_qmax:
                func = self._sepfactor_approx_qgf(theta_array, my_gfmin, my_qmax, f)
                dfunc = self._sepfactor_approx_q(theta_array, my_gfmin, my_qmax, f)
                ka = dfunc / my_qmax
                kb = -1 * dfunc
                kc = func
                func = ka * qu**2 + kb * qu + kc
                dfunc = self._sepfactor_approx_gf(theta_array, my_gfmin, my_qmax, f)
                ka = (my_gfmin * dfunc - func) / my_gfmin / my_gfmin
                kb = dfunc - 2 * my_gfmin * ka
                my_sepfactor = ka * gf**2 + kb * gf
            # (6) Сшивка с параболоидом вращения по газу в области нуля и по дебиту в области максимума
            else:
                # mQ2
                mX2 = 1.3 * my_qmax
                func = self._sepfactor_approx_qgf(theta_array, my_gfmin, my_qmax, f)
                dfunc = self._sepfactor_approx_q(theta_array, my_gfmin, my_qmax, f)
                r = my_qmax - mX2
                ka = (dfunc * r - func) / r / r
                kb = dfunc - 2 * ka * my_qmax
                kc = -ka * mX2**2 - kb * mX2
                func = ka * qu**2 + kb * qu + kc
                dfunc = self._sepfactor_approx_gf(theta_array, my_gfmin, my_qmax, f)
                ka = (my_qmax * dfunc - func) / my_gfmin / my_gfmin
                kb = dfunc - 2 * ka * my_gfmin
                my_sepfactor = ka * gf**2 + kb * gf
        # (7) Сшивка с параболоидом вращения по газу в области максимума и по дебиту в области нуля
        elif gf > my_gfmax:
            if qu < my_qmax:
                func = self._sepfactor_approx_qgf(theta_array, my_gfmax, my_qmax, f)
                dfunc = self._sepfactor_approx_q(theta_array, my_gfmax, my_qmax, f)
                ka = dfunc / my_qmax
                kb = -dfunc
                kc = func
                func = ka * qu**2 + kb * qu + kc
                # mG2
                dfunc = self._sepfactor_approx_gf(theta_array, my_gfmax, my_qmax, f)
                r = my_gfmax - 100
                ka = (dfunc * r - func) / r / r
                kb = dfunc - 2 * ka * my_gfmax
                kc = -ka * 100**2 - kb * 100
                my_sepfactor = ka * gf**2 + kb * gf + kc
            # (8) Сшивка с параболоидом вращения по газу в области максимума и по дебиту в области максимума
            else:
                # mQ2
                mX2 = 1.3 * my_qmax
                func = self._sepfactor_approx_qgf(theta_array, my_gfmax, my_qmax, f)
                dfunc = self._sepfactor_approx_q(theta_array, my_gfmax, my_qmax, f)
                r = my_qmax - mX2
                ka = (dfunc * r - func) / r / r
                kb = dfunc - 2 * ka * my_qmax
                kc = -ka * mX2**2 - kb * mX2
                func = ka * qu**2 + kb * qu + kc
                # mG2
                dfunc = self._sepfactor_approx_gf(theta_array, my_gfmax, my_qmax, f)
                r = my_gfmax - 100
                ka = (dfunc * r - func) / r / r
                kb = dfunc - 2 * ka * my_gfmax
                kc = -ka * 100**2 - kb * 100
                my_sepfactor = ka * gf**2 + kb * gf + kc
        else:
            my_sepfactor = k_gas_sep * 100

        # Обрезка по значению
        if my_sepfactor < 0:
            my_sepfactor = 10
        elif my_sepfactor > 100:
            my_sepfactor = 100

        return my_sepfactor / 100
