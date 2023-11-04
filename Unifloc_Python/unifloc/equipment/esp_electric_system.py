"""
Модуль, описывающий класс по работе с электротехнической частью УЭЦН
"""

import scipy.interpolate as interp


class EspElectricSystem:
    """
    Класс с расчетом электрических показателей работы системы УЭЦН
    """

    __slots__ = [
        "gassep_nom_power",
        "protector_nom_power",
        "transform_eff",
        "cs_eff",
        "pump_nom_freq",
        "cable_specific_resistance",
        "cable_length",
        "motor_nom_power",
        "motor_nom_voltage",
        "motor_nom_i",
        "motor_nom_freq",
        "motor_nom_power_fr",
        "gassep_power",
        "protector_power",
        "motor_voltage",
        "motor_amperage_func",
        "motor_cosf_func",
        "motor_eff_func",
        "motor_rpm_func",
    ]

    def __init__(
        self,
        motor_data: dict,
        pump_nom_freq: float,
        cable_length: float,
        gassep_nom_power: float = 500,
        protector_nom_power: float = 500,
        transform_eff: float = 0.97,
        cs_eff: float = 0.97,
        cable_specific_resistance: float = 1.18,
    ):
        """

        Parameters
        ----------
        :param motor_data: словарь с характеристиками ПЭД
        :param pump_nom_freq: номинальная частота вращения насоса, Гц
        :param cable_length: длина электрического кабеля, м
        :param gassep_nom_power: номинальная мощность газосепаратора, Вт -  optional
        :param protector_nom_power: номинальная мощность протектора, Вт - optional
        :param transform_eff: КПД трансформатора, д.ед - optional
        :param cs_eff: КПД станции управления, д.ед - optional
        :param cable_specific_resistance: удельное сопротивление кабеля, Ом/1000 м - optional

        Examples:
        --------
        >>> from unifloc.tools import common_calculations as com
        >>> from unifloc.equipment import esp_electric_system as elsys
        >>> # Исходные данные для ПЭД
        >>> motor_data = {
        ...     "ID": 1,
        ...     "manufacturer": "Centrilift",
        ...     "name": "562Centrilift-KMB-130-2200B",
        ...     "d_motor_mm": 142.7,
        ...     "motor_nom_i": 35,
        ...     "motor_nom_power": 96.98,
        ...     "motor_nom_voltage": 2200,
        ...     "motor_nom_eff": 80,
        ...     "motor_nom_cosf": 0.82,
        ...     "motor_nom_freq": 60,
        ...     "load_points": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        ...     "amperage_points": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        ...     "cosf_points": [0.7, 0.74, 0.77, 0.8, 0.83, 0.85, 0.86, 0.87, 0.88, 0.88],
        ...     "eff_points": [0.78, 0.83, 0.85, 0.88, 0.87, 0.87, 0.87, 0.87, 0.86, 0.86],
        ...     "rpm_points": [
        ...         3568.604,
        ...         3551.63,
        ...         3534.656,
        ...         3517.682,
        ...         3500.708,
        ...         3483.734,
        ...         3466.76,
        ...         3449.786,
        ...         3432.812,
        ...         3415.838,
        ...     ],
        ... }
        >>> # Гидравлическая мощность
        >>> fluid_power = 20000
        >>> # Мощность насоса
        >>> pump_power = 50000
        >>> # Номинальная мощность газосепаратора и протектора
        >>> gassep_nom_power = 1000
        >>> prot_nom_power = 1000
        >>> # Параметры электрического кабеля
        >>> t_cable = 90 + 273.15
        >>> length_cable = 2800
        >>> r_cable = 1.5
        >>> # Номинальная частота вращения насоса
        >>> pump_nom_freq = 50
        >>> # Текущая частота вращения вала
        >>> freq_shaft = 53
        >>> # КПД трансформатора и станции управления
        >>> transform_eff = 0.97
        >>> cs_eff = 0.95
        >>> # Параметры для адаптации
        >>> adaptation_parameters = {
        ...     "cosf_fact": 0.88,
        ...     "motor_i_fact": 45,
        ...     "load_fact": 0.85,
        ...     "transform_voltage_fact": 2500,
        ...     "cs_power_fact": 100000,
        ... }
        >>>
        >>> # Инициализация класса с расчетом электрики
        >>> calc_electric = elsys.EspElectricSystem(
        ...     motor_data,
        ...     pump_nom_freq,
        ...     length_cable,
        ...     gassep_nom_power,
        ...     prot_nom_power,
        ...     transform_eff,
        ...     cs_eff,
        ...     r_cable,
        ... )
        >>> # Расчет адаптационных коэффициентов
        >>> coeff = com.adapt_elsys(
        ...     calc_electric,
        ...     pump_power,
        ...     fluid_power,
        ...     freq_shaft,
        ...     t_cable,
        ...     **adaptation_parameters,
        ... )
        >>>
        >>> c_pump_power = coeff["c_pump_power"]
        >>> c_cosf = coeff["c_cosf"]
        >>> c_amperage = coeff["c_amperage"]
        >>> c_transform_power = coeff["c_transform_power"]
        >>> c_motor_volt = coeff["c_motor_volt"]
        >>>
        >>> # Расчет электрики
        >>> results = calc_electric.calc_electric_esp_system(
        ...     pump_power,
        ...     fluid_power,
        ...     freq_shaft,
        ...     t_cable,
        ...     c_pump_power,
        ...     c_cosf,
        ...     c_amperage,
        ...     c_transform_power,
        ...     c_motor_volt,
        ... )
        >>> # Расчет на другой частоте
        >>> freq_shaft_new = 60
        >>> pump_power_new = pump_power * (freq_shaft_new / freq_shaft) ** 3
        >>> fluid_power_new = fluid_power * (freq_shaft_new / freq_shaft) ** 3
        >>> results_new = calc_electric.calc_electric_esp_system(
        ...     pump_power_new,
        ...     fluid_power_new,
        ...     freq_shaft_new,
        ...     t_cable,
        ...     c_pump_power,
        ...     c_cosf,
        ...     c_amperage,
        ...     c_transform_power,
        ...     c_motor_volt,
        ... )
        """
        # TODO: v 1.5.0: уточнить использование стольких атрибутов

        self.gassep_nom_power = gassep_nom_power
        self.protector_nom_power = protector_nom_power
        self.transform_eff = transform_eff
        self.cs_eff = cs_eff
        self.pump_nom_freq = pump_nom_freq
        self.cable_specific_resistance = cable_specific_resistance
        self.cable_length = cable_length

        self.motor_nom_power = motor_data["motor_nom_power"] * 1000
        self.motor_nom_voltage = motor_data["motor_nom_voltage"]
        self.motor_nom_i = motor_data["motor_nom_i"]
        self.motor_nom_freq = motor_data["motor_nom_freq"]

        load_points = motor_data["load_points"]
        amperage_points = motor_data["amperage_points"]
        cosf_points = motor_data["cosf_points"]
        eff_points = motor_data["eff_points"]

        self.motor_nom_power_fr = None
        self.gassep_power = None
        self.protector_power = None
        self.motor_voltage = None

        # Создание интерполяционных функций для зависимости характеристик ПЭД от загрузки
        self.motor_amperage_func = interp.interp1d(
            load_points,
            amperage_points,
            kind="linear",
            fill_value="extrapolate",
        )
        self.motor_cosf_func = interp.interp1d(load_points, cosf_points, kind="linear", fill_value="extrapolate")
        self.motor_eff_func = interp.interp1d(load_points, eff_points, kind="linear", fill_value="extrapolate")

    def _calc_equip_power(self, freq_shaft) -> None:
        """
        Функция пересчета мощности сепаратора, протектора и номинальной мощности ПЭД с учетом текущей частоты

        Parameters
        ----------
        :param freq_shaft: текущая частота вращения вала, Гц

        Returns
        -------
        Обновляет следующие атрибуты:
        - мощность сепаратора, пересчитанная на текущую частоту, Вт
        - мощность протектора, пересчитанная на текущую частоту, Вт
        - номинальная мощность ПЭД, пересчитанная на текущую частоту, Вт
        """

        self.motor_nom_power_fr = self.motor_nom_power * freq_shaft / self.motor_nom_freq
        self.gassep_power = self.gassep_nom_power * (freq_shaft / self.pump_nom_freq) ** 3
        self.protector_power = self.protector_nom_power * freq_shaft / self.pump_nom_freq

    def calc_electric_esp_system(
        self,
        pump_power: float,
        fluid_power: float,
        freq_shaft: float,
        t_cable: float,
        c_pump_power: float = 1,
        c_load_i: float = 1,
        c_transform_voltage: float = 1,
        c_cs_power: float = 1,
    ) -> dict:
        """
        Функция для расчета электрики сборки УЭЦН

        Parameters
        ----------
        :param pump_power: электрическая мощность насоса, Вт
        :param fluid_power: гидравлическая мощность, Вт
        :param freq_shaft: текущая частота вращения вала, Гц
        :param t_cable: температура на глубине спуска ПЭД, К
        :param c_pump_power: адаптационный коэффициент для мощности насоса - optional
        :param c_load_i: адаптационный коэффициент для загрузки по току - optional
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе - optional
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ - optional

        Returns
        -------
        :return: словарь с рассчитанными электрическими показателями:
            * загрузка по току, д.ед.
            * мощность газосепаратора, Вт
            * мощность протектора, Вт
            * мощность насоса, Вт
            * мощность ПЭД, Вт
            * напряжение на ПЭД, В
            * сила тока ПЭД, А
            * полное сопротивление кабеля, Ом
            * напряжение на отпайке трансформатора, В
            * активная мощность на трансформаторе, Вт
            * активная мощность на станции управления, Вт
            * КПД системы ЭЦН, д.ед.
        """

        # Перевод температуры в Цельсии
        t_cable -= 273.15

        # Если насос не работает, зануляем все электрические параметры
        if pump_power == 0 or freq_shaft == 0:
            return {
                "load": 0,
                "gassep_power": 0,
                "protector_power": 0,
                "pump_power": 0,
                "motor_power": 0,
                "motor_voltage": 0,
                "motor_i": 0,
                "cable_resistance": self.cable_length
                * self.cable_specific_resistance
                / 1000
                * (1 + 0.00214 * (t_cable * 1.8 - 45)),
                "transform_voltage": 0,
                "transform_power": 0,
                "cs_power": 0,
                "pump_efficiency": 0,
                "esp_system_efficiency": 0,
            }

        # Пересчет мощности элементов системы ЭЦН в зависимости от текущей частоты
        self._calc_equip_power(freq_shaft)
        # Мощность насоса
        pump_power *= c_pump_power
        # КПД насоса
        pump_eff = fluid_power / pump_power
        # Расчет механической мощности на валу ПЭД
        motor_shaft_power = self.gassep_power + self.protector_power + pump_power
        # Расчет загрузки ПЭД
        load = motor_shaft_power / self.motor_nom_power_fr
        # Расчет силы тока ПЭД
        motor_i = self.motor_nom_i * self.motor_amperage_func(load)
        # Расчет загрузки по току
        load_i = motor_i / self.motor_nom_i * freq_shaft / self.motor_nom_freq * c_load_i
        # Косф
        motor_cosf = self.motor_cosf_func(load) * 1.0
        # Расчет активной мощности ПЭД
        motor_power = motor_shaft_power / self.motor_eff_func(load)
        # Расчет напряжения на ПЭД
        motor_voltage = motor_power / (1.732 * motor_i * motor_cosf)
        # Расчет полного сопротивления кабеля
        cable_resistance = (
            self.cable_length * self.cable_specific_resistance / 1000 * (1 + 0.00214 * (t_cable * 1.8 - 45))
        )
        # Расчет падения напряжения в кабеле
        voltage_cable_loss = 1.732 * cable_resistance * motor_i

        # Напряжение на отпайке трансформатора
        transform_voltage = (motor_voltage + voltage_cable_loss) * c_transform_voltage

        # Расчет активной мощности на входе в кабель
        cable_power = 1.732 * motor_i * motor_cosf * transform_voltage

        # Расчет активной мощности на трансформаторе
        transform_power = cable_power * (2 - self.transform_eff)
        # Расчет активной мощности на станции управления
        cs_power = transform_power * (2 - self.cs_eff) * c_cs_power
        # Расчет КПД всей системы УЭЦН
        esp_system_efficiency = fluid_power / cs_power
        return {
            "load": load_i,
            "gassep_power": self.gassep_power,
            "protector_power": self.protector_power,
            "pump_power": pump_power,
            "motor_power": motor_power,
            "motor_voltage": motor_voltage,
            "motor_i": motor_i,
            "cable_resistance": cable_resistance,
            "transform_voltage": transform_voltage,
            "transform_power": transform_power,
            "cs_power": cs_power,
            "pump_efficiency": pump_eff,
            "esp_system_efficiency": esp_system_efficiency,
        }
