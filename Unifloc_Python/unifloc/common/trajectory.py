"""
Модуль, описывающий класс по работе с траекториями
"""
from typing import Union

import numpy as np
import pandas as pd
import scipy.interpolate as interp

import unifloc.tools.exceptions as exc


class Trajectory:
    """
    Класс для интерполяции траектории для протяженного объекта
    (трубопровод, система трубопроводов, скважина)
    """

    __slots__ = ["inclinometry", "__tube_func"]

    def __init__(self, inclinometry: Union[dict, pd.DataFrame]):
        """

        Parameters
        ----------
        :param inclinometry: pd.DataFrame или dict - инлинометрия

        Examples:
        --------
        >>> import pandas as pd
        >>> from unifloc.common.trajectory import Trajectory
        >>> inclinometry = pd.DataFrame(columns=["MD", "TVD"],
        ...                             data=[[0, 0], [1400, 1400],
        ...                             [1800, 1542.85], [2000, 1700]])
        >>> trajectory = Trajectory(inclinometry)
        >>> tvd = trajectory.calc_tvd(1000)
        """
        if isinstance(inclinometry, pd.DataFrame):
            self.inclinometry = dict(inclinometry)
        elif isinstance(inclinometry, dict):
            self.inclinometry = inclinometry
        else:
            raise exc.InclinometryError(f"Неподдерживаемый тип данных - {type(inclinometry)}")
        self.__check_data()
        self.__tube_func = None

    def __check_data(self):
        """
        Проверка исходной инклинометрии на физичность значений
        """
        md_values = np.asarray(self.inclinometry["MD"])
        tvd_values = np.asarray(self.inclinometry["TVD"])

        # Проверка на дубликаты MD
        if len(md_values) != len(set(md_values)):
            raise exc.InclinometryError("Неправильная инклинометрия. В MD есть дубликаты")

        # Проверка на то, что TVD <= MD:
        if np.any(tvd_values > md_values):
            raise exc.InclinometryError("Неправильная инклинометрия. Значение TVD > MD")

        # Проверка на то, что dTVD <= dMD:
        tvd_d = np.abs(np.round(np.diff(tvd_values), 5))
        md_d = np.round(np.diff(md_values), 5)
        if np.any(tvd_d > md_d):
            raise exc.InclinometryError("Неправильная инклинометрия. Значение dTVD > dMD")

    def __interp_pipe(self):
        """
        Расчет интерполяционной функции для траектории
        """
        self.__tube_func = interp.interp1d(self.inclinometry["MD"], self.inclinometry["TVD"], fill_value="extrapolate")

    def calc_tvd(self, md: float) -> float:
        """
        Расчет TVD по интерполяционной функции траектории по MD

        Parameters
        ----------
        :param md: measured depth, м

        :return: вертикальная глубина, м
        """
        if self.__tube_func is None:
            self.__interp_pipe()

        return self.__tube_func(md).tolist()

    def calc_sin_angle(self, md1: float, md2: float) -> float:
        """
        Расчет синуса угла с горизонталью по интерполяционной функции скважины

        Parameters
        ----------
        :param md1: measured depth 1, м
        :param md2: measured depth 2, м

        :return: синус угла к горизонтали
        """
        return 0 if md2 == md1 else min((self.calc_tvd(md2) - self.calc_tvd(md1)) / (md2 - md1), 1)

    def calc_angle(self, md1: float, md2: float) -> float:
        """
        Расчет угла по интерполяционной функции траектории по MD

        Parameters
        ----------
        :param md1: measured depth 1, м
        :param md2: measured depth 2, м

        :return: угол к горизонтали, град
        """
        return (
            np.degrees(np.arcsin(self.calc_sin_angle(md1, md1 + 0.001)))
            if md2 == md1
            else np.degrees(np.arcsin(self.calc_sin_angle(md1, md2)))
        )

    def calc_ext(self, md: float) -> float:
        """
        Расчет удлинения по интерполяционной функции траектории по MD

        Parameters
        ----------
        :param md: measured depth, м

        :return: удлинение, м
        """
        return md - self.calc_tvd(md)

    @staticmethod
    def calc_tvd_angle(md: float, angle: float) -> float:
        """
        Расчет TVD для заданного угла

        :param md: MD, м
        :param angle: угол, градусы
        :return: TVD для заданного угла
        """
        return md * np.sin(angle * np.pi / 180)
