"""
Модуль, описывающий класс по работе с газлифтным клапаном
"""
import enum
import math as mt
from typing import Optional

import numpy as np
import scipy.optimize as opt

import unifloc.equipment.equipment as eq
import unifloc.service._constants as const
import unifloc.tools.exceptions as exc


class GlValve(eq.Equipment):
    """
    Класс для описания работы газлифтного клапана
    """

    __slots__ = [
        "_q_inj",
        "p_valve",
        "d",
        "valve_type",
        "s_bellow",
        "r",
        "p_dome",
        "p_open",
        "p_close",
        "status",
        "p_cas",
        "p_tub",
        "t_cas",
        "t_tub",
        "distributions",
        "distributions_annulus",
    ]

    class ReturnStatus(enum.Enum):
        OPEN = 1
        CLOSED = 2

    RETURN_STATUS = ReturnStatus.OPEN

    def __init__(
        self,
        h_mes: float,
        d: float,
        p_valve: float = None,
        s_bellow: Optional[float] = None,
        valve_type: Optional[str] = None,
    ):
        """
        :param h_mes: глубина установки клапана, м
        :param d: диаметр порта клапана, м
        :param p_valve: давление зарядки клапана, Па абс.
        :param s_bellow: площадь поперечного сечения сильфона клапана, м2
        :param valve_type: тип клапана
        """
        super().__init__(h_mes)
        self._q_inj = None
        self.p_valve = p_valve
        self.d = d
        self.valve_type = valve_type
        self.s_bellow = s_bellow
        if s_bellow is not None:
            self.r = (mt.pi * (d**2) / 4) / s_bellow
        self.p_dome = None
        self.p_open = None
        self.p_close = None
        self.p_cas = None
        self.p_tub = None
        self.t_cas = None
        self.t_tub = None
        self.status = None
        self.distributions = dict()
        self.distributions_annulus = dict()

    def __repr__(self):
        return "GlValve"

    @property
    def q_inj(self):
        """
        Расход газа через клапан
        """
        return self._q_inj

    @q_inj.setter
    def q_inj(self, new_value):
        """
        Расход газа через клапан
        :param new_value: Расход газа, м3/c
        """
        self._q_inj = new_value

    def reinit(self):
        """
        Метод для реинициализации атрибутов класса GlValve
        """
        self.p_dome = None
        self.p_open = None
        self.p_close = None
        self.p_cas = None
        self.p_tub = None
        self.t_cas = None
        self.t_tub = None
        self.status = None
        self.distributions = dict()
        self.distributions_annulus = dict()

    def dome_charge_pressure(self) -> float:
        """
        Метод расчета давления зарядки сильфона в рабочих условиях

        :return: давление зарядки сильфона в рабочих условиях, Па
        -------
        """
        self.p_dome = self.p_valve * (1 - self.r)

        return self.p_dome

    def close_pressure(self, p_d: float) -> float:
        """
        Метод определения давления закрытия клапана

        Parameters
        ----------
        :param p_d: давление зарядки сильфона в рабочих условиях, Па

        :return: давление закрытия клапана, Па
        -------
        """
        self.p_close = p_d

        return self.p_close

    def open_pressure(self, p_close: float, p_out: float, t_out: float) -> float:
        """
        Метод определения давления открытия клапана

        Parameters
        ----------
        :param p_close: давление закрытия клапана, Па
        :param p_out: давление на выходе из клапана
                        (давление в НКТ на глубине спуска клапана), Па
        :param t_out: температура на выходе из клапана
                        (температура в НКТ на глубине спуска клапана), Па

        :return: давление открытия клапана, Па
        -------
        """
        self.p_tub = p_out
        self.t_tub = t_out
        p_open = (p_close - p_out * self.r) / (1 - self.r) if self.r != 1 else (p_close - p_out)
        self.p_open = max(0, float(p_open))

        return self.p_open

    def valve_status(self, p_in: float, t_in: float, p_out: float, p_open: float) -> str:
        """
        Метод определения статуса клапана (открыт/закрыт)

        Parameters
        ----------
        :param p_in: давление на входе в клапан (давление в затрубе на глубине спуска клапана), Па
        :param t_in: температура на входе в клапан (температура в затрубе на глубине спуска
            клапана), Па
        :param p_out: давление на выходе из клапана (давление в НКТ на глубине спуска клапана), Па
        :param p_open: давление открытия клапана, Па

        :return: статус клапана: "open"/"closed"
        -------
        """
        self.p_cas = p_in
        self.t_cas = t_in

        if p_out < p_in and p_in > p_open:
            self.status = GlValve.RETURN_STATUS.OPEN.name
        else:
            self.status = GlValve.RETURN_STATUS.CLOSED.name
        return self.status

    def calc_qgas(
        self,
        p_in: float,
        p_out: float,
        t: float,
        gamma_gas: float,
        cd: Optional[float] = 1,
    ) -> tuple:
        """
        Метод расчета расхода газлифтного газа через клапан

        Parameters
        ----------
        :param p_in: затрубное давление на входе в клапан, Па
        :param p_out: давление в НКТ на выходе из клапана, Па
        :param t: рабочая температура на глубине спуска клапана, K
        :param gamma_gas: относительная плотность газа, д.ед.
        :param cd: калибровочный коэффициент

        :return: расход газлифтного газа через клапан, м3/с
        :return: критическое давление на выходе из клапана, Па
        :return: флаг критического режима течения
        -------
        """
        if not cd:
            cd = 1

        pr = p_out / p_in

        if pr < 1:
            c1 = mt.sqrt(const.CPR ** (2 / const.CP_CV) - const.CPR ** (1 + 1 / const.CP_CV))
            c2 = mt.sqrt(2 * 32.17 * const.CP_CV / (const.CP_CV - 1))
            s = mt.pi * (self.d * 39.37007874015748) ** 2 / 4
            cfr = 155.5 * cd * s * p_in * const.PSI * c1 * c2 / (mt.sqrt(gamma_gas * ((1.8 * (t - 273.15) - 32) + 460)))
            p_out_crit = p_in * const.CPR
            if pr <= const.CPR:
                q_inj = cfr * 0.000327741
                critical_regime = True
            else:
                c0 = mt.sqrt(pr ** (2 / const.CP_CV) - pr ** (1 + 1 / const.CP_CV))
                q_inj = cfr * 0.000327741 * c0 / c1
                critical_regime = False
        else:
            q_inj = 0
            p_out_crit = 0
            critical_regime = False

        return q_inj, p_out_crit, critical_regime

    def calc_pt(
        self,
        p_mes: float,
        t_mes: float,
        flow_direction: int,
        q_gas: float,
        gamma_gas: float,
        cd: Optional[float] = 1,
    ) -> float:
        """
        Метод расчета давления на другом конце клапана

        Parameters
        ----------
        :param p_mes: измеренное давление на входе или выходе клапана
                      (в зависимости от flow_direction), Па
        :param t_mes: измеренная температура на входе или выходе клапана
                      (в зависимости от flow_direction), Па
        :param flow_direction: направление потока
                               1 - к p_mes;
                               -1 - от p_mes
        :param q_gas: расход газлифтного газа, м3/с
        :param gamma_gas: относительная плотность газа, д.ед.
        :param cd: коэффициент калибровки

        :return: давление на другом конце клапана, Па

        Examples:
        --------
        >>> import unifloc.equipment.gl_system as gl_sys
        >>>
        >>> equipment_data = {"gl_system": {
        ...    "valve1": {"h_mes": 1300, "d": 0.003, "s_bellow": 0.000199677,
        ...               "p_valve": 50 * 101325,
        ...               "valve_type": "ЦКсОК"},
        ...    "valve2": {"h_mes": 1100, "d": 0.004, "s_bellow": 0.000195483,
        ...               "p_valve": 60 * 101325,
        ...               "valve_type": "ЦКсОК"},
        ...    "valve3": {"h_mes": 800, "d": 0.005, "s_bellow": 0.000199032,
        ...               "p_valve": 40 * 101325,
        ...               "valve_type": "ЦКсОК"},
        ...    "valve4": {"h_mes": 900, "d": 0.004, "s_bellow": 0.000199032,
        ...               "p_valve": 50 * 101325,
        ...               "valve_type": "ЦКсОК"}}}
        >>> gl_system = gl_sys.GlSystem(equipment_data["gl_system"])
        >>>
        >>> p_in = 70 * 101325
        >>> t = 303.15
        >>> gamma_gas = 0.6
        >>> q_gas = 10000 / 86400
        >>>
        >>> p_out = gl_system.valve_working.calc_pt(p_mes=p_in, t_mes=t, flow_direction=-1, q_gas=q_gas,
        ...                                    gamma_gas=gamma_gas)
        """

        # Проверка корректности направления потока
        if not isinstance(flow_direction, (int, float)) or flow_direction not in [
            1,
            -1,
        ]:
            raise exc.UniflocPyError(f"Неправильно задано направление потока - {flow_direction}")

        if flow_direction == 1:
            down_limit = p_mes
            up_limit = const.P_UP_LIMIT
        else:
            down_limit = 101325
            up_limit = p_mes

        try:
            p_calc = opt.brenth(
                self.__calc_qgas_error,
                a=down_limit,
                b=up_limit,
                args=(p_mes, t_mes, flow_direction, q_gas, gamma_gas, cd),
                xtol=0.000001,
            )
        except ValueError:
            p_calc = opt.minimize_scalar(
                self.__calc_qgas_error_abs,
                args=(p_mes, t_mes, flow_direction, q_gas, gamma_gas, cd),
                bounds=(down_limit, up_limit),
                method="bounded",
                options={"xatol": 0.000001},
            ).x

        if flow_direction == 1:
            self.make_dist(self.h_mes, p_mes, t_mes, p_calc, t_mes)
        else:
            self.make_dist(self.h_mes, p_calc, t_mes, p_mes, t_mes)

        return p_calc

    def make_dist(
        self,
        h_mes: float,
        p_out: Optional[float] = None,
        t_out: Optional[float] = None,
        p_in: Optional[float] = None,
        t_in: Optional[float] = None,
    ):
        """
        Метод записи давлений на входе и выходе клапана в формате распределений для НКТ и затруба

        Parameters
        ----------
        :param h_mes: глубина спуска клапана, м
        :param p_out: давление на выходе из клапана (давление в НКТ на глубине спуска клапана), Па
        :param t_out: температура на выходе из клапана (температура в НКТ на глубине спуска
            клапана), Па
        :param p_in: давление на входе в клапан (давление в затрубе на глубине спуска клапана), Па
        :param t_in: температура на входе в клапан (температура в затрубе на глубине спуска
            клапана), Па

        :return: распределения, dict
        -------

        """
        self.distributions = {"depth": np.array([h_mes])}
        self.distributions_annulus = {"depth": np.array([h_mes])}

        if p_out and t_out:
            self.distributions["p"] = np.array([p_out])
            self.distributions["t"] = np.array([t_out])

        if p_in and t_in:
            self.distributions_annulus["p"] = np.array([p_in])
            self.distributions_annulus["t"] = np.array([t_in])

    def __calc_qgas_error(
        self,
        p_1: float,
        p_2: float,
        t: float,
        flow_direction: int,
        qgas: float,
        gamma_gas: float,
        cd: Optional[float] = None,
    ) -> float:
        """
        Расчет ошибки в расходе газлифтного газа

        Parameters
        ----------
        :param p_1: давление на одном конце газлифтного клапана, Па
        :param p_2: давление на другом конце газлифтного клапана, Па
        :param t: температура на глубине газлифтного клапана, К
        :param flow_direction: направление потока
                               1 - к p_mes;
                               -1 - от p_mes
        :param qgas: расход газлифтного газа, м3/с
        :param gamma_gas: относительная плотность газа, д.ед.
        :param cd: коэффициент калибровки

        :return: ошибка между расчетным и заданным расходами газлифтного газа, м3/с
        -------
        """

        if flow_direction == 1:
            q_calc = self.calc_qgas(p_1, p_2, t, gamma_gas, cd)[0]
        else:
            q_calc = self.calc_qgas(p_2, p_1, t, gamma_gas, cd)[0]

        return qgas - q_calc

    def __calc_qgas_error_abs(
        self,
        p_1: float,
        p_2: float,
        t: float,
        flow_direction: int,
        qgas: float,
        gamma_gas: float,
        cd: Optional[float] = None,
    ) -> float:
        """
        Расчет модуля ошибки в расходе газлифтного газа

        Parameters
        ----------
        :param p_1: давление на одном конце газлифтного клапана, Па
        :param p_2: давление на другом конце газлифтного клапана, Па
        :param t: температура на глубине газлифтного клапана, К
        :param flow_direction: направление потока
                               1 - к p_mes;
                               -1 - от p_mes
        :param qgas: расход газлифтного газа, м3/с
        :param gamma_gas: относительная плотность газа, д.ед.
        :param cd: коэффициент калибровки

        :return: ошибка между расчетным и заданным расходами газлифтного газа, м3/с
        -------
        """

        return abs(self.__calc_qgas_error(p_1, p_2, t, flow_direction, qgas, gamma_gas, cd))
