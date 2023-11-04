"""
Модуль, для расчета PVT свойств по абстрактной PVT-модели
"""
from abc import ABC, abstractmethod


class PvtModel(ABC):
    """
    Абстрактный класс для расчета PVT-свойств
    """

    @abstractmethod
    def calc_pvt(self, p: float, t: float):
        """

        Parameters
        ----------
        p: давление, Па
        t: температура, К

        """
