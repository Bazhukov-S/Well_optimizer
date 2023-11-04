"""
Модуль, для описания класса для расчета давления и температуры в затрубном пространстве
"""
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.interpolate as interp
import shapely.geometry as shp

import ffmt.pvt.adapter as fl
import unifloc.common.ambient_temperature_distribution as amb
import unifloc.common.trajectory as traj
import unifloc.pipe._pipe as pipe
import unifloc.pipe.pipeline as pipel
import unifloc.tools.exceptions as exc


class Annulus(pipel.Pipeline):
    """Класс расчета динамического уровня в затрубном пространстве"""

    __slots__ = [
        "d_annulus",
        "d_tub_out",
        "d_tub_out0",
        "d_cas_in",
        "d_cas_in0",
    ]

    def __init__(
        self,
        fluid: fl.FluidFlow,
        ambient_temperature_distribution: amb.AmbientTemperatureDistribution,
        bottom_depth: float,
        d_casing: Union[float, pd.DataFrame, dict],
        d_tubing: Union[float, pd.DataFrame, dict],
        s_wall: float,
        roughness: float,
        trajectory: traj.Trajectory,
        top_depth: float = 0,
    ):
        """

        Parameters
        ----------
        :param fluid: объект PVT модели флюида
        :param ambient_temperature_distribution: объект распределения температуры, считается,
                                                 что MD в одной системе
                                                 отсчета с top_depth и bottom_depth
        :param bottom_depth: нижняя измеренная глубина трубопровода, м
        :param d_casing: внутренний диаметр ЭК, м. Можно задавать в виде таблицы формата
                         pd.DataFrame или dict или одним числом float, int
        :param d_tubing: внутренний диаметр НКТ, м. Можно задавать в виде таблицы формата
                         pd.DataFrame или dict или одним числом float, int
        :param s_wall: толщина стенки НКТ, м
        :param roughness: шероховатость трубы, м
        :param trajectory: объект с инклинометрией, считается, что MD в одной системе
                           отсчета с top_depth и bottom_depth
        :top_depth: верхняя измеренная глубина трубопровода, м

        Examples:
        --------
        >>> import pandas as pd
        >>> import unifloc.pvt.fluid_flow as fl
        >>> import unifloc.common.trajectory as traj
        >>> import unifloc.common.ambient_temperature_distribution as amb
        >>> import unifloc.pipe.annulus as ann
        >>> # Инициализация исходных данных класса FluidFlow
        >>> q_fluid = 100 / 86400
        >>> wct = 0
        >>> pvt_model_data = {
        ...     "black_oil": {"gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.8,
        ...                   "wct": wct, "phase_ratio": {"type": "GOR", "value": 50},
        ...                   "oil_correlations": {"pb": "Standing", "rs": "Standing",
        ...                                        "rho": "Standing", "b": "Standing",
        ...                                         "mu": "Beggs", "compr": "Vasquez"},
        ...                   "gas_correlations": {"ppc": "Standing", "tpc": "Standing",
        ...                                        "z": "Dranchuk", "mu": "Lee"},
        ...                   "water_correlations": {"b": "McCain", "compr": "Kriel",
        ...                                          "rho": "Standing", "mu": "McCain"},
        ...                   "rsb": {"value": 50, "p": 10000000, "t": 303.15},
        ...                   "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
        ...                   "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
        ...                   "table_model_data": None, "use_table_model": False}}
        >>> # Инициализация исходных данных класса WellTrajectory
        >>> trajectory = pd.DataFrame(columns=["MD", "TVD"], data=[[0, 0],
        ...                                                        [2000, 2000], [2500, 2500]])
        >>> well_trajectory_data = {"inclinometry": trajectory}
        >>> well_trajectory = traj.Trajectory(**well_trajectory_data)
        >>> # Задание параметров сепарации
        >>> p_sep = 4 * (10 ** 6)
        >>> t_sep = 303.15
        >>> k_sep = 0.5
        >>> # Создание объекта флюида
        >>> fluid = fl.FluidFlow(q_fluid, pvt_model_data)
        >>> # Модификация флюида
        >>> fluid.modify(p_sep, t_sep, k_sep, calc_type="annulus")
        >>> bottom_depth = 2000
        >>> d_casing = 0.146
        >>> d_tubing = 0.062
        >>> s_wall = 0.005
        >>> roughness = 0.0001
        >>> # Создание объекта с температурой породы
        >>> ambient_temperature_data = {"MD": [0, 1000], "T": [303.15, 303.15]}
        >>> amb_temp = amb.AmbientTemperatureDistribution(ambient_temperature_data)
        >>> # Создание объекта затрубного пространства
        >>> annulus = ann.Annulus(fluid, amb_temp, bottom_depth, d_casing, d_tubing,
        ...                       s_wall, roughness, well_trajectory)
        >>> # Инициализация исходных данных метода расчета динамического уровня calc_hdyn
        >>> p_esp = 150 * 101325
        >>> t_esp = 303.15
        >>> p_ann = 20 * 101325
        >>> t_ann = 303.15
        >>> # Вызов метода расчета динамического уровня
        >>> h_dyn, p_dyn = annulus.calc_hdyn(p_esp, p_ann, wct)
        """
        self.d_annulus = self.__calc_d_annulus(bottom_depth, d_casing, d_tubing, s_wall)
        super().__init__(
            top_depth=top_depth,
            bottom_depth=bottom_depth,
            d=self.d_annulus,
            roughness=roughness,
            trajectory=trajectory,
            fluid=fluid,
            ambient_temperature_distribution=ambient_temperature_distribution,
        )
        self.d_tub_out, self.d_tub_out0 = self.__calc_d_pipe(d=d_tubing, top_depth=self.top_depth, s_wall=s_wall)
        self.d_cas_in, self.d_cas_in0 = self.__calc_d_pipe(d=d_casing, top_depth=self.top_depth)
        self.pipe_object = pipe.Pipe(
            fluid=self.fluid,
            d=self.d0,
            roughness=roughness,
            hydr_corr_type=self.hydr_corr_type,
            d_tub_out=self.d_tub_out0,
            d_cas_in=self.d_cas_in0,
            s_wall=self.s_wall,
        )

    def calc_hdyn(
        self,
        p_esp: float,
        p_ann: float,
        wct: Optional[float] = None,
        step_length: float = 1,
    ) -> Tuple[float, float]:
        """
        Метод расчета динамического уровня

        Parameters
        ----------
        :param p_esp: давление на приеме ЭЦН, Па
        :param p_ann: давление в затрубном пространстве скважины, Па
        :param wct: обводненность, доли ед.
        :param step_length: шаг расчета кривой распределения давления по стволу, м

        :return: динамический уровень, м
        :return: давление на динамическом уровне, Па
        -------
        """
        q_fluid = 0

        # Расчет распределения давления газа в затрубном пространстве
        # от устья до глубины спуска ЭЦН
        self.fluid.fluid_type = "gas"
        self.fluid.reinit_fluid_type("gas")
        self.calc_pt(
            h_start="top",
            p_mes=p_ann,
            flow_direction=1,
            q_liq=q_fluid,
            wct=wct,
            phase_ratio_value=None,
            t_mes=None,
            hydr_corr_type="static",
            step_len=step_length,
            heat_balance=False,
        )
        pdist_ann_gas = self.distributions["p"]
        gas_pressure_line = shp.LineString(np.column_stack((self.distributions["depth"], pdist_ann_gas)))
        self.fluid.fluid_type = "liquid"
        self.fluid.reinit_fluid_type("liquid")
        # Расчет распределения давления жидкости в затрубном
        # пространстве от глубины спуска ЭЦН до устья
        self.calc_pt(
            h_start="bottom",
            p_mes=p_esp,
            flow_direction=1,
            q_liq=q_fluid,
            wct=wct,
            phase_ratio_value=None,
            t_mes=None,
            hydr_corr_type="static",
            step_len=step_length,
            heat_balance=False,
        )

        pdist_ann_liq = self.distributions["p"]
        liquid_pressure_line = shp.LineString(np.column_stack((self.distributions["depth"], pdist_ann_liq)))

        # Поиск точки пересечения распределений давления жидкости и газа
        intersection = liquid_pressure_line.intersection(gas_pressure_line)

        if intersection.geom_type == "Point":
            return intersection.x, intersection.y
        if pdist_ann_gas[-1] > pdist_ann_liq[0]:
            return self.bottom_depth, pdist_ann_gas[-1]
        if pdist_ann_liq[-1] > pdist_ann_gas[0]:
            return self.top_depth, pdist_ann_liq[0]

        raise exc.UniflocPyError("Кривые давлений жидкости и газа совпадают." "Проверьте исходные данные")

    @staticmethod
    def __calc_d_annulus(bottom_depth, d_casing, d_tubing, s_wall):
        """
        Метод расчета диаметра затрубного пространства

        Parameters
        ----------
        :param bottom_depth: нижняя измеренная глубина трубопровода, м
        :param d_casing: диаметр ЭК, м. Можно задавать в виде таблицы формата pd.DataFrame
        или dict или одним числом float, int
        :param d_tubing: диаметр НКТ, м. Можно задавать в виде таблицы формата pd.DataFrame
        или dict или одним числом float, int
        :param s_wall: толщина стенки НКТ, м

        :return: диаметр затрубного пространства, м.
        Может быть получен в виде dict или числа float, int
        -------
        """

        if isinstance(d_tubing, (float, int)) and isinstance(d_casing, (float, int)):
            # Диаметры НКТ и ЭК заданы числами (постоянные)
            d_annulus = d_casing - (d_tubing + 2 * s_wall)
        elif not isinstance(d_tubing, (float, int)):
            # Преобразование диаметров НКТ и ЭК, заданных в
            # виде pd.DataFrame или dict в массив numpy
            d_tubing_array = np.array(d_tubing["d"])

            if isinstance(d_casing, (float, int)):
                d_annulus_array = d_casing - (d_tubing_array + 2 * s_wall)
                md_annulus = d_tubing["MD"]
            else:
                md_tubing = np.array(d_tubing["MD"])
                d_tubing_func = interp.interp1d(md_tubing, d_tubing_array, fill_value="extrapolate", kind="previous")

                md_casing = np.array(d_casing["MD"])
                d_casing_func = interp.interp1d(md_casing, d_casing["d"], fill_value="extrapolate", kind="previous")

                md_annulus = np.sort(np.unique(np.append(md_tubing, md_casing[np.where(md_casing <= bottom_depth)])))

                d_annulus_array = d_casing_func(md_annulus) - (d_tubing_func(md_annulus) + 2 * s_wall)

            d_annulus = {"MD": md_annulus, "d": d_annulus_array}
        else:
            md_casing = np.array(d_casing["MD"])
            d_casing_func = interp.interp1d(md_casing, d_casing["d"], fill_value="extrapolate", kind="previous")
            md_annulus = np.sort(
                np.unique(
                    np.append(
                        [0, bottom_depth],
                        md_casing[np.where(md_casing <= bottom_depth)],
                    )
                )
            )

            d_annulus_array = d_casing_func(md_annulus) - (d_tubing + 2 * s_wall)
            d_annulus = {"MD": md_annulus, "d": d_annulus_array}

        return d_annulus

    @staticmethod
    def __calc_d_pipe(d, top_depth, s_wall: Optional[float] = None):
        """
        Метод для определения диаметра трубы

        Parameters
        ----------
        :param d: диаметр, м. Можно задавать в виде таблицы формата pd.DataFrame
        или dict или одним числом float, int
        :param top_depth: верхняя измеренная глубина трубопровода, м
        :param s_wall: толщина стенки НКТ, м

        :return: диаметр, м / объект расчет диаметра по глубине скважины
        -------
        """

        if isinstance(d, (float, int)):
            if s_wall:
                d_func = d + 2 * s_wall
                d0 = d + 2 * s_wall
            else:
                d_func = d
                d0 = d
        elif isinstance(d, (pd.DataFrame, dict)):
            if s_wall:
                d_func = interp.interp1d(
                    np.array(d["MD"]),
                    np.array(d["d"]) + 2 * s_wall,
                    fill_value="extrapolate",
                    kind="previous",
                )
            else:
                d_func = interp.interp1d(d["MD"], d["d"], fill_value="extrapolate", kind="previous")
            d0 = d_func(top_depth).item()
        else:
            raise TypeError(f"Неподдерживаемый тип данных для диаметра - {type(d)}")

        return d_func, d0
