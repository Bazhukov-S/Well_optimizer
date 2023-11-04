"""
Модуль с exceptions, которые возбуждаются в расчетных модулях
"""


class UniflocPyError(Exception):
    """
    Общий класс для описания ошибок в UniflocPy
    """

    def __init__(self, message, detail=None):
        super().__init__(message)

        self.detail = detail


class InclinometryError(UniflocPyError):
    """
    Класс для описания ошибок инклинометрии в UniflocPy
    """


class GlValveError(UniflocPyError):
    """
    Класс для описания ошибок газлифтных клапанов в UniflocPy
    """


class NotImplementedHydrCorrError(UniflocPyError, NotImplementedError):
    """
    Класс для описания ошибок не заданной гидравлической корреляции в UniflocPy
    """


class NotImplementedPvtCorrError(UniflocPyError, NotImplementedError):
    """
    Класс для описания ошибок не заданной PVT-корреляции в UniflocPy
    """


class NotImplementedChokeCorrError(UniflocPyError, NotImplementedError):
    """
    Класс для описания ошибок не заданной корреляции для штуцера в UniflocPy
    """


class NotImplementedPhaseRatioError(UniflocPyError, NotImplementedError):
    """
    Класс для описания ошибок не заданного фазового соотношения UniflocPy
    """


class OptimizationStatusError(UniflocPyError):
    """Класс для описания неудачного статуса расчета при оптимизации."""

    def __init__(self):
        super().__init__(message="Неудачная попытка оптимизации.")
