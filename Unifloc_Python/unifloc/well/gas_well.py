from typing import Optional, Union

import unifloc.well.gaslift_well as glw


class GasWell(glw.GasLiftWell):
    """
    Класс для расчета газовой скважины.

    Принимает на вход словари с исходными данными.

    Структура словарей исходных данных:

        * fluid_data ("ключ" - определение - тип) - dict
        * "q_fluid" - дебит жидкости в ст.у., ст. м/c - float
        * "wct" - объемная обводненность, доли ед. - float
        * "pvt_model_data" - словарь с флюидами - dict
            * "black_oil" - словарь с свойствами нефти, газа и воды для модели Black Oil - dict
                * "gamma_gas" - относительная плотность газа по воздуху в ст.у.
                 (плотность воздуха = 1.2217 кг/м3)-float
                * "gamma_oil" - относительная плотность нефти по воде в ст.у.
                 (плотность воды = 1000 кг/м3) - float
                * "gamma_wat" - относительная плотность воды по воде в ст.у.
                 (плотность воды = 1000 кг/м3) - float
                * "rp" - газовый фактор, ст. м3 газа/ст. м3 нефти - float
                * "oil_correlations" - словарь с набором корреляций для нефти - dict, optional
                    * "pb" - название корреляции для давления насыщения - string, optional
                        * Возможные значения: "Standing"
                    * "rs" - название корреляции для газосодержания - string, optional
                        * Возможные значения: "Standing"
                    * "rho" - название корреляции для плотности нефти - string, optional
                        * Возможные значения: "Standing"
                    * "mu" - название корреляции для вязкости нефти - string, optional
                        * Возможные значения: "Beggs"
                    * "b" - название корреляции для объемного коэффициента нефти - string, optional
                        * Возможные значения: "Standing"
                    * "compr" - название корреляции для сжимаемости нефти - string, optional
                        * Возможные значения: "Vasquez"
                * "gas_correlations" - словарь с набором корреляций для газа - dict, optional
                    * "ppc" - название корреляции для критического давления - string, optional
                        * Возможные значения: "Standing"
                    * "tpc" - название корреляции для критической температуры - string, optional
                        * Возможные значения: "Standing"
                    * "z" - название корреляции для z-фактора - string, optional
                        * Возможные значения: "Kareem", "Dranchuk"
                    * "mu" - название корреляции для вязкости газа - string, optional
                        * Возможные значения: "Lee"
                * "water_correlations" - словарь с набором корреляций для газа - dict, optional
                    * "b" - название корреляции для объемного коэффициента воды - string, optional
                        * Возможные значения: "McCain"
                    * "rho" - название корреляции для плотности воды - string, optional
                        * Возможные значения: "Standing", "IAPWS"
                    * "mu" - название корреляции для вязкости воды - string, optional
                        * Возможные значения: "McCain", "IAPWS"
                    * "compr" - название корреляции для сжимаемости воды - string, optional
                        * Возможные значения: "Kriel"
                * "salinity" - минерализация воды, ppm - float, optional
                * "rsb" - словарь с калибровочным значением газосодержания при давлении насыщения -
                dict, optional
                    * "value" - калибровочное значение газосодержания при давлении насыщения,
                    ст. м3 газа/ст. м3 нефти - float
                    * "p" - давление калибровки, Па абс. - float
                    * "t" - температура калибровки газосодержания, К - float
                * "bob" - словарь с калибровочным значением объемного коэффициента нефти при
                давлении насыщения - dict, optional
                    * "value" - калибровочное значение объемного коэффициента нефти при давлении
                    насыщения, ст. м3/ст. м3 - float
                    * "p" - давление калибровки, Па абс. - float
                    * "t" - температура калибровки объемного коэффициента нефти, К - float
                * "muob" - словарь с калибровочным значением вязкости нефти при давлении насыщения
                - dict, optional
                    * "value" - калибровочное значение вязкости нефти при давлении насыщения, сПз - float
                    * "p" - давление калибровки, Па абс. - float
                    * "t" - температура калибровки вязкости нефти, К - float
                * "table_model_data" - словарь с исходными данными табличной модели - dict, optional
                    * "pvt_dataframes_dict" - словарь с таблицами с исходными данными
                    - dict of DataFrames
                    * "interp_type" -  тип интерполяции (по умолчанию - линейный) - string, optional
                * "use_table_model" - флаг использования табличной модели - boolean, optional
    * pipe_data ("ключ" - определение - тип) - dict
        * "casing" - словарь с исходными данными для создания ЭК - dict
            * "bottom_depth" - измеренная глубина верхних дыр перфорации, м - float
            * "d" - внутренний диаметр ЭК, м - float, pd.DataFrame("MD", "d")
            * ! можно задавать как числом, так и таблицей с распределением по глубине или словарем,
             см. пример
            * "roughness" - шероховатость, м - float
        * "tubing" - словарь с исходными данными для создания колонны НКТ - dict
            * "bottom_depth" - измеренная глубина спуска колонны НКТ, м - float
            * "d" - внутренний диаметр колонны НКТ, м - float, pd.DataFrame("MD", "d")
            * ! можно задавать как числом, так и таблицей с распределением по глубине или словарем,
             см. пример
            * "roughness" - шероховатость, м - float
            * "s_wall" - толщина стенки, м - float
    * well_trajectory_data ("ключ" - определение - тип) - dict
        * "inclinometry" - таблица с инклинометрией, две колонки: "MD","TVD", индекс по умолчанию,
         см.пример - DataFrame
        * или возможно с помощью dict с ключами "MD", "TVD"
    * ambient_temperature_data ("ключ" - определение - тип) - словарь с распределением температуры
     породы по MD - dict
        * обязательные ключи MD, T - list
    * equipment_data ("ключ" - определение - тип) - dict, optional
        * "choke" - словарь с исходными данными для создания объекта штуцера - dict, optional
            * "d" - диаметр штуцера, м - float
    """

    def __init__(
        self,
        fluid_data: dict,
        pipe_data: dict,
        well_trajectory_data: dict,
        ambient_temperature_data: dict,
        equipment_data: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        :param fluid_data: словарь с исходными данными для создания флюида
        :param pipe_data: словарь с исходными данными для создания колонн труб
        :param well_trajectory_data: словарь с исходными данными для создания инклинометрии скважины
        :param ambient_temperature_data: словарь с распределением температуры породы по MD
        :param equipment_data: словарь с исходными данными для создания различного оборудования,
            optional

        Examples:
        --------
        >>> import pandas as pd
        >>> import unifloc.tools.units_converter as uc
        >>> from unifloc.well.gas_well import GasWell
        >>> # Инициализация исходных данных
        >>> df = pd.DataFrame(columns=["MD", "TVD"], data=[[0, 0], [1400, 1400], [1800, 1800]])
        >>> # Возможный способ задания инклинометрии через dict
        >>> # df = {"MD": [0, 1000],
        >>> #       "TVD": [0, 1000]}
        >>> # В словари с калибровками подается давление и температура калибровки.
        >>> # Зачастую - это давление насыщения и пластовая температура
        >>> fluid_data = {"q_fluid": uc.convert_rate(100, "m3/day", "m3/s"),
        ...               "pvt_model_data": {
        ...                   "black_oil": {
        ...                       "gamma_gas": 0.78, "gamma_wat": 1, "gamma_oil": 0.8,
        ...                       "wct": 0,
        ...                   }
        ...               },
        ...               "fluid_type": "gas"}
        >>> # Диаметр можно задавать как числом так и таблицей с распределением по глубине
        >>> d = pd.DataFrame(columns=["MD", "d"], data=[[0, 0.062], [1000, 0.082]])
        >>> # Так тоже возможно: d = {"MD": [0, 1000], "d": [0.06, 0.08]}
        >>> pipe_data = {"casing": {"bottom_depth": 1800, "d": 0.146, "roughness": 0.0001},
        ...              "tubing": {"bottom_depth": 1400, "d": d, "roughness": 0.0001}}
        >>> well_trajectory_data = {"inclinometry": df}
        >>> equipment_data = None
        >>> ambient_temperature_data = {"MD": [0, 1800], "T": [303.15, 303.15]}
        >>> # Инициализация объекта скважины
        >>> well = GasWell(fluid_data, pipe_data, well_trajectory_data,
        ...                    ambient_temperature_data, equipment_data)
        >>> # Расчет забойного давления
        >>> p_fl = 10 * 101325
        >>> q_gas = 500000 / 86400
        >>> p_wh = 26 * 101325
        >>> d_ch = 0.032
        >>> friction_factor = None
        >>> c_choke = None
        >>> output_params = True
        >>> step_length = 100
        >>> # Расчет с сохранением доп. атрибутов распределений свойств
        >>> p_wf = well.calc_pwf_pfl(p_fl, q_gas, d_ch, p_wh, friction_factor,
        ...                          c_choke, step_length,
        ...                          output_params)
        >>> # Запрос всех значений доп. свойств в виде словаря
        >>> result = well.extra_output
        """
        fluid_data["fluid_type"] = "gas"
        super().__init__(
            fluid_data,
            pipe_data,
            well_trajectory_data,
            ambient_temperature_data,
            equipment_data,
        )

    def calc_pwf_pfl(
        self,
        p_fl: float,
        q_gas: float,
        d_ch: Optional[float] = None,
        p_wh: Optional[float] = None,
        friction_factor: Optional[float] = None,
        c_choke: Optional[Union[float, dict]] = None,
        step_length: Optional[float] = None,
        output_params: bool = False,
        heat_balance: bool = False,
        **kwargs
    ) -> float:
        """
        Расчет забойного давления по линейному с учетом всех гидравлических элементов

        Parameters
        ----------
        :param p_fl: линейное давление, Па абс.
        :param q_gas: дебит газа, ст. м3/с
        :param d_ch: диаметр штуцера, м
        :param p_wh: буферное давление, Па абс.
        :param friction_factor: к-т адаптации КРД на трение, если не задан берется из атрибутов трубы
        :param c_choke: адаптационный коэффициент штуцера \
            Задается в виде числа как коэффициент калибровки, либо как словарь {"const": value}, \
            где value - постоянный перепад, который будет использоваться как перепад между буферным и линейным давлением
        :param step_length: длина шага интегрирования, м
        :param output_params: флаг для расчета дополнительных распределений параметров
        :param heat_balance: опция учета теплопотерь

        :return: забойное давление, Па абс.
        """
        if self.choke and d_ch:
            self.choke.d = d_ch
        p_wf = super().calc_pwf_pfl(
            p_fl=p_fl,
            q_liq=q_gas,
            wct=0,
            p_wh=p_wh,
            hydr_corr_type="gray",
            q_gas_inj=0,
            friction_factor=friction_factor,
            grav_holdup_factor=None,
            c_choke=c_choke,
            step_length=step_length,
            output_params=output_params,
            heat_balance=heat_balance,
        )
        return p_wf
