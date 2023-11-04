"""
Модуль, описывающий класс однофазного трения по Moody
"""
import math as mt


class Friction:
    """
    Класс для расчета однофазного трения
    """

    @staticmethod
    def calc_n_re(d_m, rho_n_kgm3, vsm_msec, mu_n_cp):
        """
        Вычисление числа Рейнольдса

        Parameters
        ----------
        :param d_m: диаметр трубы, м
        :param rho_n_kgm3: плотность смеси, кг/м3
        :param vsm_msec: скорость смеси, м/с
        :param mu_n_cp: вязкость смеси, сПз

        :return: число Рейнольдса, безразмерн.
        """
        return 1000 * rho_n_kgm3 * vsm_msec * d_m / max(mu_n_cp, 0.000001)

    @staticmethod
    def calc_norm_ff(n_re, eps, rough_pipe):
        """
        Рассчитывает нормирующий коэффициент трения для шероховатых труб,
        используя относительную шероховатость трубы
        и число Рейнольдса.

        Parameters
        ----------
        :param n_re: число Рейнольдса, безразмерн.
        :param eps: относительная шероховатость трубы, безразмерн.
        :param rough_pipe: флаг, указывающий на способ расчета коэффициента
                           трения для шероховатой трубы с
                           использованием корреляции Муди (rough_pipe > 0)
                           или используя корреляцию Дрю для гладких труб
        Eсли Re попадает в переходный режим, то коэф. трения расчитывается через:
        1) турбулентный коэф. трения при Re = 4000 (верхняя граница)
        2) ламинарный коэф. трения при Re = 2000 (нижняя граница)

        :return: нормирующий коэффициент трения, безразмерн.
        """

        if n_re == 0:
            f_n = 0
        elif n_re < 2000:  # ламинарный поток
            f_n = 64 / n_re
        else:
            n_re_save = -1  # флаг для расчета переходного режима
            if n_re <= 4000:
                n_re_save = n_re
                n_re = 4000

            # расcчитываем турбулентный режим
            if rough_pipe > 0:
                f_n = (
                    2
                    * mt.log10(0.5405405405405405 * eps - 5.02 / n_re * mt.log10(0.5405405405405405 * eps + 13 / n_re))
                ) ** -2
                i = 0
                while True:
                    f_n_new = (1.74 - 2 * mt.log10(2 * eps + 18.7 / (n_re * f_n**0.5))) ** -2
                    i = i + 1
                    error = abs(f_n_new - f_n) / f_n_new
                    f_n = f_n_new
                    # stop when error is sufficiently small or max number of iterations exceeded
                    if error <= 0.0001 or i > 19:
                        break
            else:
                f_n = 0.0056 + 0.5 * n_re**-0.32

            if n_re_save > 0:  # переходный режим
                min_re = 2000
                max_re = 4000
                f_turb = f_n
                f_lam = 0.032
                f_n = f_lam + (n_re_save - min_re) * (f_turb - f_lam) / (max_re - min_re)

        norm_friction_factor = f_n

        return norm_friction_factor
