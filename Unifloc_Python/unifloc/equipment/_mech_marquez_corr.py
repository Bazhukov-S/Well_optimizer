"""
Модуль, описывающий расчет коэффициента естественной сепарации
"""
import math

import scipy.integrate as integr

import unifloc.pipe._friction as fr


class MechMarquezCorr:
    """
    Класс, описывающий механическую корреляцию Маркеза для расчета коэффициента естественной сепарации
    """

    def __init__(self, d_pump: float, d_cas: float):
        """

        Parameters
        ----------
        :param d_pump: внешний диаметр насоса, м
        :param d_cas: внутренний диаметр ЭК, м
        ----------
        """
        self.r_pump = d_pump / 2
        self.r_cas = d_cas / 2
        self.h_p = 0.03
        self.tan_beta = (self.r_cas - self.r_pump) / self.h_p
        self.d_h = d_cas - d_pump
        self.s_annular = math.pi * (self.r_cas**2 - self.r_pump**2)
        self.fric = fr.Friction()

    @staticmethod
    def v_slip(
        rho_liq: float,
        rho_gas: float,
        stlg: float,
        fp: int,
    ) -> float:
        """
        Метод расчета скорости проскальзывания

        Parameters
        ----------
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3
        :param stlg: поверхностное натяжение, Н/м
        :param fp: режим потока

        :return: скорость проскальзывания, м/с
        --------
        """
        if fp == 1:
            v_slip = 2 ** (1 / 2) * (9.81 * stlg * (rho_liq - rho_gas) / (rho_liq**2)) ** (1 / 4)

        else:
            v_slip = 1.53 * (9.81 * stlg * (rho_liq - rho_gas) / (rho_liq**2)) ** (1 / 4)
        return v_slip

    @staticmethod
    def calc_r_d(v_sg: float, v_slip: float) -> float:
        """
        Метод расчета характерного радиуса пузырька

        Parameters
        ----------
        :param v_sg: приведенная скорость газа, м/с
        :param v_slip: скорость проскальзывания, м/с

        :return: характерный радиус пузырька,м
        --------
        """
        return 3 / (5660.705 * (1.000001 - math.exp(-2.5483248 * v_sg))) * v_sg / (v_sg + v_slip)

    @staticmethod
    def trans_a(
        v_sl: float,
        v_slip: float,
    ) -> float:
        """
        Метод расчета критической скорости A

        Parameters
        ----------
        :param v_sl: приведенная скорость жидкости, м/с
        :param v_slip: скорость проскальзывания, м/с

        :return: критическая скорость A, м/с
        --------
        """
        return v_sl / 4 + 0.20 * v_slip

    def trans_b(
        self,
        rho_liq: float,
        rho_gas: float,
        stlg: float,
        v_m: float,
        f: float,
    ) -> float:
        """
        Метод расчета критической скорости B

        Parameters
        ----------
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3
        :param stlg: поверхностное натяжение, Н/м
        :param v_m: приведенная скорость смеси, м/с
        :param f: коэффициент трения, безразм

        :return: критическая скорость B, м/с
        --------
        """
        a = (
            2
            * (0.4 * stlg / ((rho_liq - rho_gas) * 9.81)) ** (1 / 2)
            * (rho_liq / stlg) ** (3 / 5)
            * (2 / self.d_h) ** (2 / 5)
            * f ** (2 / 5)
        )
        b = ((a) * v_m ** (6 / 5) - 0.725) / 4.15
        return v_m * (b) ** 2

    @staticmethod
    def trans_c(v_sl: float, v_slip: float) -> float:
        """
        Метод расчета критической скорости C

        Parameters
        ----------
        :param v_sl: приведенная скорость жидкости, м/с
        :param v_slip: скорость проскальзывания, м/с

        :return: критическая скорость C, м/с
        --------
        """
        return 1.083 * v_sl + 0.52 * v_slip

    def check_if_slug(
        self,
        v_sl: float,
        v_sg: float,
        rho_liq: float,
        rho_gas: float,
        stlg: float,
        v_mix: float,
        f: float,
        v_dt: float,
    ) -> bool:
        """
        Условие принадлежности к пробковому участку карты режимов потока

        Parameters
        ----------
        :param v_sl: приведенная скорость жидкости, м/с
        :param v_sg: приведенная скорость газа, м/с
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3
        :param stlg: поверхностное натяжение, Н/м
        :param v_mix: приведенная скорость смеси, м/с
        :param f: коэф. трения, безразм.
        :param v_dt: скорость пузырька Тейлора, м/с
        ----------
        """
        v_sl_slug = (1.2 * v_sl + v_dt) / 2.8
        if v_sl < v_sl_slug:
            below_a = v_sg >= self.trans_a(
                v_sl,
                v_dt,
            )
            below_b = v_sg >= self.trans_b(rho_liq, rho_gas, stlg, v_mix, f)
            return below_a and below_b
        else:
            below_s = v_sg >= self.trans_c(v_sl, v_dt)
            return below_s

    def define_fp(
        self,
        v_sl: float,
        v_sg: float,
        rho_liq: float,
        rho_gas: float,
        rho_mix: float,
        stlg: float,
        mum: float,
    ) -> int:
        """
        Метод определения режима потока

        Parameters
        ----------
        :param v_sl: приведенная скорость жидкости, м/с
        :param v_sg: приведенная скорость газа, м/с
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3
        :param rho_mix: плотность смеси, кг/м3
        :param stlg: поверхностное натяжение, Н/м
        :param mum: вязкость смеси, сПз

        :return fp: режим потока
            * 0 - пузырьковый/рассеяный (bybble/dispersed);
            * 1 - пробковый (slug);
        ----------
        """
        v_dt = 1.53 * (9.81 * stlg * (rho_liq - rho_gas) / (rho_liq**2)) ** (1 / 4)
        v_mix = v_sl + v_sg
        r_e = self.fric.calc_n_re(self.d_h, rho_mix, v_mix, mum)
        f = self.fric.calc_norm_ff(r_e, 0.000055, 0)
        if self.check_if_slug(
            v_sl,
            v_sg,
            rho_liq,
            rho_gas,
            stlg,
            v_mix,
            f,
            v_dt,
        ):
            return 1
        else:
            return 0

    def drdh(
        self,
        h: float,
        r: float,
        mul: float,
        r_d: float,
        rho_liq: float,
        rho_gas: float,
        v_sl: float,
    ) -> float:
        """
        Метод определения радиуса сепарации

        Parameters
        ----------
        :param h: задействованная высота щели приема, м
        :param r: радиус сепарации, м
        :param mul: вязкость жидкости, Па*с
        :param r_d: характерный радиус пузырька, м
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3
        :param v_sl: приведеннная скорость жидкости, м/с

        :return: радиус сепарации, м
        --------
        """
        drdh = (
            54
            * mul
            / r_d**2
            * 1
            / ((rho_liq - rho_gas) * 9.81)
            * (self.r_pump + h / self.h_p * (self.r_cas - self.r_pump))
            * v_sl
            * self.tan_beta
            * (
                2
                / 9
                * (r_d**2 * rho_liq)
                / mul
                * 1
                / r**3
                * (self.r_pump + h / self.h_p * (self.r_cas - self.r_pump))
                * v_sl
                * self.tan_beta
                + 1 / r
            )
        )
        return drdh

    def find_r_i(
        self,
        rho_liq: float,
        rho_gas: float,
        stlg: float,
        mul: float,
        v_sl: float,
        v_sg: float,
        fp: int,
    ) -> float:
        """
        Метод определения радиуса пузырька

        Parameters
        ----------
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3
        :param stlg: поверхностное натяжение, Н/м
        :param mul: вязкость жидкости, сПз
        :param v_sl: приведенная скорость жидкости, м/с
        :param v_sg: приведенная скорость газа, м/с
        :param fp: режим потока

        :return: радиус пузырька, м
        --------
        """
        v_slip = MechMarquezCorr.v_slip(rho_liq, rho_gas, stlg, fp)
        r_d = self.calc_r_d(v_sg, v_slip)
        mul = mul / 1000
        r_s = integr.solve_ivp(
            self.drdh,
            [self.h_p, 0],
            (self.r_cas, self.r_pump),
            args=(mul, r_d, rho_liq, rho_gas, v_sl),
            rtol=1e-10,
            atol=1e-10,
        )
        return r_s.y[0][-1]

    def calc_e_sep(self, r_s: float) -> float:
        """
        Метод расчета коэффициента естественной сепарации

        Parameters
        ----------
        :param r_s: радиус пузырька, м

        :return: коэффициент естественной сепарации
        --------
        """

        return (self.r_cas**2 - (r_s**2)) / (self.r_cas**2 - self.r_pump**2)

    def calc_k_sep(
        self,
        ql: float,
        qg: float,
        mul: float,
        mum: float,
        rho_liq: float,
        rho_gas: float,
        rho_mix: float,
        stlg: float,
    ) -> float:
        """
        Метод расчета коэффициента естественной сепарации и необходимых параметров

        Parameters
        ----------
        :param ql: дебит жидкости, м3/с
        :param qg: дебит газа, м3/с
        :param mul: вязкость жидкости, сПз
        :param mum: вязкость смеси, сПз
        :param rho_liq: плотность жидкости, кг/м3
        :param rho_gas: плотность газа, кг/м3
        :param rho_mix: плотность смеси, кг/м3
        :param stlg: поверхностное натяжение, Н/м

        :return: коэффициент естественной сепарации, д.ед.
        --------
        """
        if ql > 0 and qg > 0:
            v_sl = ql / self.s_annular
            v_sg = qg / self.s_annular

            fp = self.define_fp(
                v_sl,
                v_sg,
                rho_liq,
                rho_gas,
                rho_mix,
                stlg,
                mum,
            )

            r_s = self.find_r_i(rho_liq, rho_gas, stlg, mul, v_sl, v_sg, fp)

            e_sep = self.calc_e_sep(r_s)
            e_sep = max(e_sep, 0)
            e_sep = min(e_sep, 1)
        else:
            e_sep = 0
        return e_sep
