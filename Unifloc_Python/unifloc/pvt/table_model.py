"""
Модуль, для расчета PVT свойств по табличной PVT-модели
"""
import scipy.interpolate as sc

import unifloc.tools.exceptions as exc


class TableModel:
    """
    Класс, включающий функции для расчета свойств с помощью табличной функции
    """

    def __init__(self, pvt_dataframes_dict: dict, interp_type: str = "linear"):
        """
        Parameters
        ----------
        pvt_dataframes_dict - словарь с таблицами с исходными данными (dict of DataFrames)
        interp_type - тип интерполяции (по умолчанию - линейный)
        ----------
        """
        self.pvt_dataframes_dict = pvt_dataframes_dict
        self.__interp_type = interp_type
        self.__init_funcs()

    @property
    def interp_type(self):
        return self.__interp_type

    @interp_type.setter
    def interp_type(self, interp_kind):
        """
        Изменение текущего типа интерполяции

        Parameters
        ----------
        :param interp_kind: тип интерполяции

        Specifies the kind of interpolation as a string
        (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
        ‘previous’, ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’
        refer to a spline interpolation of zeroth,
        first, second or third order; ‘previous’ and ‘next’ simply return
        the previous or next value of the point) o
         as an integer specifying the order of the spline interpolator to use. Default is ‘linear’.

        :return: тип интерполяции
        -------
        """
        if interp_kind not in ["linear", "cubic"]:
            raise exc.UniflocPyError(f"Неподходящий тип интерполяции - {interp_kind}")
        else:
            self.__interp_type = interp_kind
            self.__init_funcs()

    def __init_funcs(self):
        """
        Инициализация функции для расчета свойств в зависимости от подаваемой таблицы

        :return: создает атрибуты для всех свойств в исходном словаре с PVT-свойствами
        -------
        """
        for pvt_key, df in self.pvt_dataframes_dict.items():
            if df.shape[1] == 1:
                self.__make_interp_func(df.index.values, df.iloc[:, 0].values, pvt_key)
            else:
                self.__make_interp_func(df.index.values, df.columns.values, pvt_key, df.values)

    def __make_interp_func(self, x_array, y_array, pvt_property, z_array=None):
        """
        Общая функция для создания интерполяционного полинома

        Parameters
        ----------
        x_array: массив иксов
        y_array: массив игреков
        pvt_property: название свойства
        z_array: массив z

        :return: создает атрибут с интерполяционным полиномом pvt_property_func: z = f(x,y),
        если z не задано, то y = f(x)
        -------

        """
        if z_array is None:
            interp_func = sc.interp1d(x_array, y_array, kind=self.__interp_type, fill_value="extrapolate")
        else:
            interp_func = sc.interp2d(x_array, y_array, z_array.T, kind=self.__interp_type)
        setattr(self, pvt_property + "_func", interp_func)

    def calc_property(self, p: float, t: float, pvt_property: str, **kwargs):
        """
        Расчет свойства для данного давления и температуры

        Parameters
        ----------
        :param p: давление в тех же единицах измерения что и в исходной таблице
        :param t: температура в тех же единицах измерения  что и в исходной таблице
        :param pvt_property: название свойства (как в таблице)

        :return: значение свойства, рассчитанное по таблице pvt-свойств
        -------
        """
        # TODO: v 1.5.0 - избавиться от рефлексии
        if hasattr(self, pvt_property + "_func"):
            if isinstance(getattr(self, pvt_property + "_func"), sc.interpolate.interp1d):
                return getattr(self, pvt_property + "_func")(p).item()
            elif isinstance(self.__dict__[pvt_property + "_func"], sc.interpolate.interp2d):
                return getattr(self, pvt_property + "_func")(p, t).item()
            else:
                raise exc.UniflocPyError(
                    f"Неизвестный тип у интерполяционного полинома -" f"{type(getattr(self, pvt_property + '_func'))}"
                )
        else:
            raise exc.UniflocPyError(
                f"{pvt_property} нет в атрибутах." f"Проверьте исходные данные и формат задания таблицы"
            )
