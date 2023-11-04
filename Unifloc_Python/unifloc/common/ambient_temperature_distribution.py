"""
Модуль, описывающий класс по работе с температурой породы
"""
from functools import cached_property

import scipy.interpolate as interp

import unifloc.tools.exceptions as exc


class AmbientTemperatureDistribution:
    """
    Класс для задания распределения температуры породы
    по глубине для протяженного объекта
    (трубопровод, система трубопроводов, скважина)
    """

    def __init__(self, ambient_temperature_distribution: dict):
        """
        :param ambient_temperature_distribution: распределение температуры в кельвинах по MD, dict
        """
        self.__amb_temp_dist = None
        self.__depth_option = None
        self.amb_temp_dist = ambient_temperature_distribution
        self.__amb_temp_func = None

    @staticmethod
    def __check_data(amb_temp_dist: dict):
        """
        Проверка исходных распределений
        """
        for key in ["MD", "T"]:
            if key not in amb_temp_dist:
                raise exc.UniflocPyError(
                    f"Отсутствует ключ {key} в словаре с температурой породы." f"Поменяйте исходные данные",
                    amb_temp_dist,
                )

        if len(amb_temp_dist["MD"]) != len(amb_temp_dist["T"]):
            raise exc.UniflocPyError(
                "Исходные массивы 'MD' и 'T' разной длины. Поменяйте исходные данные",
                amb_temp_dist,
            )

        if len(amb_temp_dist["MD"]) != len(set(amb_temp_dist["MD"])):
            raise exc.UniflocPyError("Неправильная инклинометрия. В MD есть дубликаты", amb_temp_dist)

    @cached_property
    def amb_temp_func(self):
        """
        Расчет интерполяционной функции для температуры породы
        """
        self.__amb_temp_func = interp.interp1d(
            self.amb_temp_dist["MD"], self.amb_temp_dist["T"], fill_value="extrapolate"
        )
        return self.__amb_temp_func

    @property
    def amb_temp_dist(self):
        """
        Словарь с распределением температуры породы по глубине MD
        """
        return self.__amb_temp_dist

    @amb_temp_dist.setter
    def amb_temp_dist(self, value: float):
        """
        Сеттер распределения температуры породы
        """
        self.__check_data(value)
        self.__amb_temp_dist = value

        if "amb_temp_func" in vars(self):
            del self.amb_temp_func

    def calc_geotemp_grad(self, depth: float) -> float:
        """
        Расчет геотермического градиента по MD

        :param depth: глубина, м
        :return: градиент температуры, К/м
        """
        return (self.amb_temp_func(depth + 0.01) - self.amb_temp_func(depth)) / 0.01

    def calc_temp(self, depth: float) -> float:
        """
        Расчет температуры породы

        :param depth: глубина, м
        :return: температура породы, К
        """
        return self.amb_temp_func(depth)
