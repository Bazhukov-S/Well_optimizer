"""
Модуль, описывающий класс по работе с естественной сепарацией
"""
from typing import Union

import scipy.interpolate as interp

import unifloc.equipment.equipment as eq


class NaturalSeparation(eq.Equipment):
    """
    Класс, описывающий естественную сепарацию газа
    """

    def __init__(self, h_mes: float):
        """
        Parameters
        ----------
        :param h_mes: глубина, на которой производится расчет
                     (чаще всего глубина спуска НТК или насоса), м  - float

        Examples:
        --------
        >>> from unifloc.equipment.natural_separation import NaturalSeparation
        >>> # Инициализация исходных данных
        >>> h_mes_ = 1800
        >>> d_tub = 0.063
        >>> d_cas = 0.130
        >>> q_fluid = 100/86400
        >>> q_gas = 0
        >>> sigma_l = 24
        >>> rho_liq = 836
        >>> rho_gas = 0.84
        >>> # Инициализация объекта для расчета естественной сепарации
        >>> k_sep_nat_calc = NaturalSeparation(h_mes_)
        >>> # Расчет коэффициента естественной сепарации
        >>> k_sep_natural = k_sep_nat_calc.calc_separation(d_tub, d_cas, q_fluid, q_gas,
        ...                                                sigma_l, rho_liq, rho_gas)
        """
        super().__init__(h_mes)

    @staticmethod
    def determine_flow_pattern(n_fr: float, lambda_l: float) -> int:
        """
        Определение структуры потока по карте режимов потока

        Parameters
        ----------
        :param n_fr: Число Фруда, безразмерн.
        :param lambda_l: Объемное содержание жидкости, безразмерн.

        :return: номер режима по карте режимов потока, безразмерн.
                режим потока:
                * 0 - расслоенный (segregated);
                * 1 - прерывистый (INTERMITTENT);
                * 2 - распределенный (distributed);
                * 3 - переходный (transition);
        -------
        """
        l1 = 316 * lambda_l**0.302
        l2 = 0.0009252 / (lambda_l**2.4684)
        l3 = 0.1 / (lambda_l**1.4516)
        l4 = 0.5 / (lambda_l**6.738)

        if (lambda_l < 0.4 and n_fr >= l1) or (lambda_l >= 0.4 and n_fr > l4):
            flow_pattern = 2
        elif (lambda_l < 0.01 and n_fr < l1) or (lambda_l >= 0.01 and n_fr < l2):
            flow_pattern = 0
        elif lambda_l >= 0.01 and l3 >= n_fr >= l2:
            flow_pattern = 3
        elif (0.4 > lambda_l >= 0.01 and l3 < n_fr <= l1) or (lambda_l > 0.4 and l3 <= n_fr <= l4):
            flow_pattern = 1
        else:
            # flow_pattern = "undefined"
            flow_pattern = 1
        return flow_pattern

    @staticmethod
    def __calc_nat_sep(v_sl, v_inf):
        """
        Функция расчета естественной сепарации по корреляции Marquez
        Parameters
        ----------
        :param v_sl: приведенная скорость жидкости, м/с
        :param v_inf: скорость проскальзывания газа в вертикальном направления, м/с
        :return: коэффициент естественной сепарации, д.ед.

        References
        ----------
        R.Marquez, "Modelling downhole natural separation", 2004
        """
        a = -0.0093
        b = 57.758
        c = 34.4
        d = 1.308
        m = v_sl / v_inf

        # Коэффициент естественной сепарации
        if m > 13:
            k_gassep_natural = 0
        else:
            k_gassep_natural = ((1 + (a * b + c * m**d) / (b + m**d)) ** 272 + m**272) ** (1 / 272) - m
        return k_gassep_natural

    def calc_separation(
        self,
        d_tub: Union[float, interp.interp1d],
        d_cas: Union[float, interp.interp1d],
        q_liq: float,
        q_gas: float,
        sigma_l: float,
        rho_liq: float,
        rho_gas: float,
    ) -> float:
        """
        Расчет приведенной скорости жидкости и скорости проскальзывания газа в
        вертикальном направления, необходимых для расчета естественной сепарации

        Parameters
        ----------
        :param d_tub: внутренний диаметр НКТ, м
        :param d_cas: внутренний диаметр эксплуатационной колонны, м
        :param q_liq: дебит жидкости, м3/с
        :param q_gas: дебит газа, м3/с
        :param sigma_l: поверхностное натяжение жидкость-газ, Н/м
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3

        :return: коэффициент естественной сепарации, д.ед.
        """
        if q_liq == 0 or d_tub == d_cas:
            return 1

        if isinstance(d_tub, interp.interp1d):
            d_tub = d_tub(self.h_mes).item()

        if isinstance(d_cas, interp.interp1d):
            d_cas = d_cas(self.h_mes).item()

        a_p = 3.14 * (d_cas**2 - d_tub**2) / 4
        v_sg = q_gas / a_p
        v_sl = q_liq / a_p
        v_mix = v_sg + v_sl
        n_fr = v_mix**2 / (9.81 * (d_cas - d_tub))
        lambda_l = q_liq / (q_liq + q_gas)
        flow_pattern = self.determine_flow_pattern(n_fr, lambda_l)

        if flow_pattern in (0, 1):
            v_inf = 1.53 * (9.81 * sigma_l * (rho_liq - rho_gas) / rho_liq**2) ** 0.25
        else:
            v_inf = 1.41 * (9.81 * sigma_l * (rho_liq - rho_gas) / rho_liq**2) ** 0.25

        return self.__calc_nat_sep(v_sl, v_inf)
