"""
Модуль, описывающий класс по работе с системой УЭЦН
"""
from typing import Optional

import unifloc.equipment.esp as es
import unifloc.equipment.esp_electric_system as el_sys
import unifloc.equipment.separator as sep
import unifloc.tools.exceptions as exc


class EspSystem:
    """
    Класс, описывающий работу всей установки ЭЦН
    """

    __slots__ = [
        "esp",
        "esp_electric_system",
        "separator",
        "electric_adaptation",
        "distributions",
        "power_fluid",
        "efficiency_esp",
        "nom_eff",
        "nom_rate",
        "nom_head",
        "nom_power",
        "rate_points_corr",
        "head_points_corr",
        "power_points_corr",
        "eff_points_corr",
    ]

    def __init__(
        self,
        esp: dict,
        esp_electric_system: Optional[dict] = None,
        separator: Optional[dict] = None,
    ):
        """
        :param esp: параметры для инициализации объекта Esp, dict
        :param esp_electric_system: параметры для инициализации объекта EspElectricSystem, dict - optional
        :param separator: параметры для инициализации объекта Separator, dict - optional
        """
        # TODO: v 1.5.0: уточнить использование стольких атрибутов
        self.esp = es.Esp(**esp)

        self.esp_electric_system = (
            el_sys.EspElectricSystem(**esp_electric_system) if esp_electric_system is not None else esp_electric_system
        )

        self.separator = sep.Separator(**separator) if separator is not None else separator

        self.power_fluid = None
        self.distributions = None
        self.efficiency_esp = None
        self.nom_eff = None
        self.nom_rate = None
        self.nom_head = None
        self.nom_power = None
        self.rate_points_corr = None
        self.head_points_corr = None
        self.power_points_corr = None
        self.eff_points_corr = None

    @property
    def fluid(self):
        """
        Объект флюида
        """
        return self.esp.fluid

    @fluid.setter
    def fluid(self, new_value):
        self.esp.fluid = new_value

    def calc_general_separation(
        self,
        k_sep_nat: float,
        gf: Optional[float] = None,
        q_liq: Optional[float] = None,
        freq: Optional[float] = None,
    ):
        """
        Расчет общего коэффициента сепарации

        Parameters
        ----------
        :param k_sep_nat: коэффициент естественной сепарации, д.ед.
        :param gf: доля газа на приеме насоса, д.ед. - optional
        :param q_liq: дебит жидкости, м3/с - optional
        :param freq: частота вращения вала ЭЦН, Гц - optional

        :return: коэффициент общей сепарации, д.ед.
        """
        return (
            self.separator.calc_general_separation(k_sep_nat, gf, q_liq, freq)
            if self.separator is not None
            else k_sep_nat
        )

    def calc_esp_system(
        self,
        q_liq: float,
        wct: float,
        p: float,
        t: float,
        freq: float,
        t_cable: Optional[float] = None,
        direction_to: str = "dis",
        head_factor: Optional[float] = None,
        slippage_factor: float = 0.97222,
        extra_output: bool = False,
        c_pump_power: float = 1,
        c_load_i: float = 1,
        c_transform_voltage: float = 1,
        c_cs_power: float = 1,
    ) -> tuple:
        """
        Расчет всей УЭЦН

        Parameters
        ----------
        :param q_liq: дебит жидкости, м3/с
        :param wct: обводненность, д.ед.
        :param p: давление на приеме либо на выкиде (зависит от направления расчета), Па
        :param t: температура на приеме либо на выкиде (зависит от направления расчета), К
        :param freq: частота, Гц
        :param t_cable: температура на глубине спуска ПЭД, К  - optional
        :param direction_to: направление расчета:
                             -dis - от приема к выкиду (по умолчанию);
                             -in - от выкида к приему.
        :param head_factor: коэффициент поправки на напор, д.ед - optional
        :param slippage_factor: коэффициент проскальзывания ЭЦН, д.ед. По умолчанию задан 0.97222
        :param extra_output: флаг сохранения дополнительных атрибутов
        :param c_pump_power: адаптационный коэффициент для мощности насоса
        :param c_load_i: адаптационный коэффициент для загрузки по току
        :param c_transform_voltage: адаптационный коэффицицент для напряжения на трансформаторе
        :param c_cs_power: адаптационный коэффицицент для мощности на СУ

        :return: целевое давление (на приеме или на выкиде), Па
        :return: целевая температура (на приеме или на выкиде), К
        :return: статус расчета насоса
        :return: мощность газосепаратора, Вт
        :return: мощность протектора, Вт
        :return: мощность насоса, Вт
        :return: загрузка ПЭД, д.ед
        :return: мощность ПЭД, Вт
        :return: сила тока ПЭД, А
        :return: напряжение на ПЭД, В
        :return: напряжение на трансформаторе, В
        :return: мощность на трансформаторе, Вт
        :return: активная мощность на станции управления, Вт
        :return: КПД системы УЭЦН
        """

        gassep_power = None
        protector_power = None
        motor_load = None
        motor_power = None
        motor_i = None
        motor_voltage = None
        transform_voltage = None
        transform_power = None
        cs_power = None
        esp_system_efficiency = None

        pt = self.esp.calc_pt(
            p=p,
            t=t,
            freq=freq,
            q_liq=q_liq,
            wct=wct,
            phase_ratio_value=None,
            direction_to=direction_to,
            head_factor=head_factor,
            slippage_factor=slippage_factor,
            extra_output=extra_output,
        )
        esp_power = self.esp.power_esp
        pump_efficiency = self.esp.efficiency
        self.power_fluid = self.esp.power_fluid
        self.nom_eff = self.esp.nom_eff
        self.nom_rate = self.esp.nom_rate
        self.nom_head = self.esp.nom_head
        self.nom_power = self.esp.nom_power
        self.rate_points_corr = self.esp.rate_points_corr
        self.head_points_corr = self.esp.head_points_corr
        self.power_points_corr = self.esp.power_points_corr
        self.eff_points_corr = self.esp.eff_points_corr

        self.distributions = self.esp.distributions

        if self.esp_electric_system is not None:
            if t_cable is not None:
                # Расчет электрики
                results_electric_esp_system = self.esp_electric_system.calc_electric_esp_system(
                    pump_power=esp_power,
                    fluid_power=self.power_fluid,
                    freq_shaft=freq,
                    t_cable=t_cable,
                    c_pump_power=c_pump_power,
                    c_load_i=c_load_i,
                    c_transform_voltage=c_transform_voltage,
                    c_cs_power=c_cs_power,
                )

                gassep_power = results_electric_esp_system["gassep_power"]
                protector_power = results_electric_esp_system["protector_power"]
                esp_power = results_electric_esp_system["pump_power"]
                motor_load = results_electric_esp_system["load"]
                motor_power = results_electric_esp_system["motor_power"]
                motor_i = results_electric_esp_system["motor_i"]
                motor_voltage = results_electric_esp_system["motor_voltage"]
                transform_voltage = results_electric_esp_system["transform_voltage"]
                transform_power = results_electric_esp_system["transform_power"]
                cs_power = results_electric_esp_system["cs_power"]
                pump_efficiency = results_electric_esp_system["pump_efficiency"]
                esp_system_efficiency = results_electric_esp_system["esp_system_efficiency"]
            else:
                raise exc.UniflocPyError("Необходимо задать температуру на глубине спуска ПЭД")

        return (
            pt[0],
            pt[1],
            pt[2],
            gassep_power,
            protector_power,
            esp_power,
            motor_load,
            motor_power,
            motor_i,
            motor_voltage,
            transform_voltage,
            transform_power,
            cs_power,
            pump_efficiency,
            esp_system_efficiency,
        )
