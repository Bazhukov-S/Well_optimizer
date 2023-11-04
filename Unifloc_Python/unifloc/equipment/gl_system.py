"""
Модуль, описывающий класс по работе с системой газлифтных клапанов
"""
from unifloc.equipment import gl_valve as gl


class GlSystem:
    """
    Класс для описания работы системы газлифтных клапанов
    """

    __slots__ = ["valves", "valve_working"]

    def __init__(self, valves_data: dict):
        """

        :param valves_data: словарь с данными газлифтных клапанов
        """
        self.valves = {}

        # Создание объектов-клапанов
        for key in valves_data:
            key_lower = key.lower()
            if key_lower.startswith("valve"):
                self.valves[key_lower] = gl.GlValve(**valves_data[key])

        # Определение рабочего клапана, как клапана с максимальной глубиной
        self.valve_working = max(self.valves.values(), key=lambda x: x.h_mes)

        self.valves = sorted(self.valves.values(), key=lambda x: x.h_mes, reverse=True)

    def __repr__(self):
        return "GlSystem"

    @property
    def q_inj(self):
        """
        Расход газа через рабочий клапан
        """
        return self.valve_working.q_inj

    @q_inj.setter
    def q_inj(self, value):
        """
        Расход газа через рабочий клапан
        :param value: Расход газа, м3/c
        """
        self.valve_working.q_inj = value

    @property
    def h_mes_work(self):
        """
        Глубина установки рабочего клапана
        """
        return self.valve_working.h_mes

    @h_mes_work.setter
    def h_mes_work(self, value):
        """
        Глубина установки рабочего клапана
        :param value: глубина рабочего клапана, м
        """
        self.valve_working.h_mes = value
