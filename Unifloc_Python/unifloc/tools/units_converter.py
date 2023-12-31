"""
Модуль для конвертации единиц измерения

"""

import numpy as np
import scipy.constants as const

# constants, which don"t have in scipy.constants

g = const.g  # gravity
pi = const.pi
psc_bar = 1  # pressure standard condition
tsc_c = 15  # temperature standard condition
const_at = 98066.5  # техническая атмосфера в Па
air_density_sckgm3 = 1.225  # definition from https://en.wikipedia.org/wiki/Density_of_air
z_default = 0.9
gamma_gas_default = 0.8
rsb_default_m3m3 = 100
rho_w_kgm3_sc = 1000
foot = const.foot
bbl = const.bbl


# complex unit conversion functions


def convert_pressure(val, old_scale, new_scale):
    """
    Convert from a pressure scale to another one among psi, bar,
    atm scales.

    Parameters
    ----------
    val : array_like
        Value(s) of the pressure(s) to be converted expressed in the
        original scale.
    old_scale: str
        Specifies as a string the original scale from which the pressure
        value(s) will be converted. Supported scales are atm ("atm",
        "Atm", "C" or "c"), psi ("psi"), bar ("bar") and Pa
        ("Pa").
    new_scale: str
        Specifies as a string the new scale to which the temperature
        value(s) will be converted. Supported scales are atm ("atm",
        "Atm", "C" or "c"), psi ("psi"), bar ("bar") and Pa
        ("Pa").
    Returns
    -------
    res : float or array of floats
        Value(s) of the converted pressure(s) expressed in the new scale.
    Notes
    -----
    .. versionadded:: 0.1
    """
    # Convert from `old_scale` to Pa
    if old_scale.lower() in ["bar", "бар"]:
        tempo = __bar2Pa(np.asanyarray(val))
    elif old_scale.lower() in ["atm", "атм"]:
        tempo = __atm2Pa(np.asanyarray(val))
    elif old_scale.lower() in ["psi"]:
        tempo = __psi2Pa(np.asanyarray(val))
    elif old_scale.lower() in ["at", "ат", "kgssm2", "кгссм2"]:
        tempo = __atm2Pa(np.asanyarray(val))
    elif old_scale.lower() in ["mpa", "мпа"]:
        tempo = np.asanyarray(val) * const.mega
    elif old_scale.lower() in ["pa", "па"]:
        tempo = np.asanyarray(val)
    else:
        raise NotImplementedError(
            "%s scale is unsupported: supported scales " "are bar, atm, at and, MPa, Pa" % old_scale
        )
    # and from Kelvin to `new_scale`.
    if new_scale.lower() in ["bar", "бар"]:
        res = __Pa2bar(tempo)
    elif new_scale.lower() in ["atm", "атм"]:
        res = __Pa2atm(tempo)
    elif new_scale.lower() in ["at", "ат", "kgssm2", "кгссм2"]:
        res = __Pa2at(tempo)
    elif new_scale.lower() in ["psi"]:
        res = __Pa2psi(tempo)
    elif new_scale.lower() in ["mpa", "мпа"]:
        res = tempo / const.mega
    elif new_scale.lower() in ["pa", "па"]:
        res = tempo
    else:
        raise NotImplementedError(
            "'%s' scale is unsupported: supported scales " "are bar, atm, at and, MPa, Pa" % new_scale
        )

    return res


def convert_temperature(val, old_scale, new_scale):
    """
    Convert a temperature from old_scale to new_scale

    Parameters
    ----------
    val
    old_scale
    new_scale

    Returns
    -------

    """
    return const.convert_temperature(val, old_scale, new_scale)


def convert_length(val, old_scale, new_scale):
    # TODO надо сделать конвертер для расстояний. Общая функция и набор быстрых функций
    pass


def convert_rate(val, old_scale, new_scale):
    """
    Convert from a rate scale to another one

    based on scipy.constants.convert_temperature
    Parameters
    ----------
    val : array_like
        Value(s) of the pressure(s) to be converted expressed in the
        original scale.
    old_scale: str
        Specifies as a string the original scale from which the pressure
        value(s) will be converted. Supported scales are "m3/s", "m3/day"
    new_scale: str
        Specifies as a string the new scale to which the temperature
        value(s) will be converted. Supported scales are "m3/s", "m3/day".
    Returns
    -------
    res : float or array of floats
        Value(s) of the converted pressure(s) expressed in the new scale.
    Notes
    -----
    .. versionadded:: 0.1
    """
    # Convert from `old_scale` to m3/s
    if old_scale.lower() in ["m3/day", "м3/сут"]:
        tempo = np.asanyarray(val) / 86400
    elif old_scale.lower() in ["m3/s", "м3/с"]:
        tempo = np.asanyarray(val)
    else:
        raise NotImplementedError("%s scale is unsupported: supported scales " "are m3/day, m3/s" % old_scale)
    # and from m3/s to `new_scale`.
    if new_scale.lower() in ["m3/day", "м3/сут"]:
        res = tempo * 86400
    elif new_scale.lower() in ["m3/s", "м3/с"]:
        res = tempo
    else:
        raise NotImplementedError("%s scale is unsupported: supported scales " "are m3/day, m3/s" % old_scale)
    return res


def convert_GOR(val, old_scale, new_scale, gamma_oil=0.86):
    # TODO надо сделать конвертер для газового фактора, газосодержания. Общая функция и набор быстрых функций
    # в том числе надо чтобы поддерживалась конвертация между м3/м3 и м3/т
    pass


def convert_density(val, old_scale, new_scale):
    # TODO надо сделать конвертер для плотности. Общая функция и набор быстрых функций
    pass


# simple unit conversion functions
# pressure
def __psi2Pa(value):
    """
    converts pressure in psi to Pa
    :param value: pressure value in psi
    :return: pressure value in Pa
    """
    return value * const.psi


def __Pa2psi(value):
    """
    converts pressure in psi to Pa
    :param value: pressure value in psi
    :return: pressure value in Pa
    """
    return value / const.psi


def __MPa2psi(value):
    """
    converts pressure in psi to MPa
    :param value: pressure value in psi
    :return: pressure value in Pa
    """
    return value * const.mega / const.psi


def __bar2Pa(value):
    """
    converts pressure in bar to Pa
    :param value: pressure value in bar
    :return: pressure value in Pa
    """
    return value * const.bar


def __bar2MPa(value):
    """
    converts pressure in bar to Pa
    :param value: pressure value in bar
    :return: pressure value in Pa
    """
    return value * const.bar / const.mega


def __Pa2bar(value):
    """
    converts pressure in Pa to bar
    :param value: pressure value in Pa
    :return: pressure value in bar
    """
    return value / const.bar


def __MPa2bar(value):
    """
    converts pressure in MPa to bar
    :param value: pressure value in Pa
    :return: pressure value in bar
    """
    return value * const.mega / const.bar


def __atm2Pa(value):
    """
    converts pressure in atm (standard atmosphere) to Pa
    :param value: pressure value in atm
    :return: pressure value in Pa
    """
    return value * const.atm


def __Pa2atm(value):
    """
    converts pressure in Pa to atm (standard atmosphere)
    :param value: pressure value in Pa
    :return: pressure value in atm (standard atmosphere)
    """
    return value / const.atm


def __at2Pa(value):
    """
    converts pressure in at (technical atmosphere) to Pa
    :param value: pressure value in atm
    :return: pressure value in Pa
    """
    return value * const_at


def __Pa2at(value):
    """
    converts pressure in Pa (Pascal) to at (technical atmosphere)
    :param value: pressure value in Pa
    :return: pressure value in at (technical atmosphere)
    """
    return value * const_at


def __bar2psi(value):
    """
    converts pressure in bar to psi
    :param value: pressure value in bar
    :return: pressure value in psi
    """
    return value * const.bar / const.psi


def __psi2bar(value):
    """
    converts pressure in psi to bar
    :param value: pressure value in psi
    :return: pressure value in bar
    """
    return value / const.bar * const.psi


def __bar2atm(value=1):
    """
    converts pressure in bar to atm(standard atmosphere)
    :param value: pressure value in bar
    :return: pressure value in atm(standard atmosphere)
    """
    return value * const.bar / const.atm


def __atm2bar(value=1):
    """
    converts pressure in atm(standard atmosphere) to bar
    :param value: pressure value in atm(standard atmosphere)
    :return: pressure value in bar
    """
    return value * const.atm / const.bar


# temperature


def __c2f(value):
    """
    converts temperature in C(degrees Celsius) to F(degrees Fahrenheit)
    :param value: temperature in C(degrees Celsius)
    :return: temperature in F(degrees Fahrenheit)
    """
    return const.convert_temperature(value, "C", "F")


def __f2c(value):
    """
    converts temperature in F(degrees Fahrenheit) to C(degrees Celsius)
    :param value: temperature in F(degrees Fahrenheit)
    :return: temperature in C(degrees Celsius)
    """
    return const.convert_temperature(value, "F", "K")


def __c2k(value):
    """
    converts temperature in C(degrees Celsius) to K(Kelvins)
    :param value: temperature in C(degrees Celsius)
    :return: temperature in K(Kelvins)
    """
    return const.convert_temperature(value, "C", "K")


def __k2c(value):
    """
    converts temperature in K(Kelvins) to C(degrees Celsius)
    :param value: temperature in K(Kelvins)
    :return: temperature in C(degrees Celsius)
    """
    return const.convert_temperature(value, "K", "C")


def _k2f(value):
    """
    converts temperature in F(degrees Fahrenheit) to K(Kelvins)
    :param value: temperature in F(degrees Fahrenheit)
    :return: temperature in K(Kelvins)
    """
    return const.convert_temperature(value, "K", "F")


def __f2k(value):
    """
    converts temperature in F(degrees Fahrenheit) to K(Kelvins)
    :param value: temperature in F(degrees Fahrenheit)
    :return: temperature in K(Kelvins)
    """
    return const.convert_temperature(value, "F", "K")


def __f2r(value):
    """
    converts temperature in F(degrees Fahrenheit) to R(degrees Rankine)
    :param value: temperature in F(degrees Fahrenheit)
    :return: temperature in R(degrees Rankine)
    """
    return const.convert_temperature(value, "F", "R")


def __r2f(value):
    """
    converts temperature in R(degrees Rankine) to F(degrees Fahrenheit)
    :param value: temperature in R(degrees Rankine)
    :return: temperature in F(degrees Fahrenheit)
    """
    return const.convert_temperature(value, "R", "F")


def __c2r(value):
    """
    converts temperature in C(degrees Celsius) to R(degrees Rankine)
    :param value: temperature in C(degrees Celsius)
    :return: temperature in R(degrees Rankine)
    """
    return const.convert_temperature(value, "C", "R")


def __r2c(value):
    """
    converts temperature in R(degrees Rankine) to C(degrees Celsius)
    :param value: temperature in C(degrees Celsius)
    :return: temperature in R(degrees Rankine)
    """
    return const.convert_temperature(value, "R", "C")


def __k2r(value):
    """
    converts temperature in K(Kelvins) to R(degrees Rankine)
    :param value: temperature in K(Kelvins)
    :return: temperature in R(degrees Rankine)
    """
    return const.convert_temperature(value, "K", "R")


def __r2k(value):
    """
    converts temperature in R(degrees Rankine) to K(Kelvins)
    :param value: temperature in R(degrees Rankine)
    :return: temperature in K(Kelvins)
    """
    return const.convert_temperature(value, "R", "K")


# length


def __m2in(value):
    """
    converts length in m(meters) to in(inches)
    :param value: length in m(meters)
    :return: length in in(inches)
    """
    return value / const.inch


def __in2m(value):
    """
    converts length in in(inches) to m(meters)
    :param value: length in in(inches)
    :return: length in m(meters)
    """
    return value * const.inch


def __m2ft(value):
    """
    converts length in m(meters) to ft(feet)
    :param value: length in m(meters)
    :return: length in ft(feet)
    """
    return value / const.foot


def __ft2m(value):
    """
    converts length in ft(feet) to m(meters)
    :param value: length in ft(feet)
    :return: length in m(meters)
    """
    return value * const.foot


# rate


def __m3_2_bbl(value=1):
    """
    converts rate in m3(cubic metres) to bbl(barrels)
    :param value: rate in m3(cubic metres)
    :return: rate in bbl(barrels)
    """
    return value / const.barrel


def __bbl2m3(value=1):
    """
    converts rate in m3(cubic metres) to bbl(barrels)
    :param value: rate in m3(cubic metres)
    :return: rate in bbl(barrels)
    """
    return value / const.barrel


# GOR


def __m3m3_2_m3t(value, gamma=1):
    """
    converts Gas-Oil Ratio in m3/m3(cubic metres/cubic meter) to m3/t(cubic metres/ton)
    :param value: Gas-Oil Ratio in m3/m3(cubic metres/cubic meter);
           gamma=1: oil density(by air)
    :return: Gas-Oil Ratio in m3/t(cubic metres/ton)
    """
    return value * gamma


def __m3t2m3m3(value, gamma=1):
    """
    converts Gas-Oil Ratio in m3/t(cubic metres/ton) to m3/m3(cubic metres/cubic meter)
    :param value: Gas-Oil Ratio in m3/t(cubic metres/ton);
           gamma=1: oil density(by air)
    :return: Gas-Oil Ratio in m3/m3(cubic metres/cubic meter)
    """
    return value / gamma


def __scfstb2m3m3(value):
    """
    converts Gas-Oil Ratio in scf/stb(standard cubic feet/standard barrel) to m3/m3(cubic metres/cubic meter)
    :param value: Gas-Oil Ratio in scf/stb(standard cubic feet/standard barrel)
    :return: Gas-Oil Ratio in m3/m3(cubic metres/cubic meter)
    """
    return value * (const.foot**3 / const.bbl)


def _m3m3_2_scfstb(value):
    """
    converts Gas-Oil Ratio in m3/m3(cubic metres/cubic meter) to scf/stb(standard cubic feet/standard barrel)
    :param value: Gas-Oil Ratio in m3/m3(cubic metres/cubic meter)
    :return: Gas-Oil Ratio in scf/stb(standard cubic feet/standard barrel)
    """
    return value / (const.foot**3 / const.bbl)


# density


def __api2gamma_oil(value):
    """
    converts density in API(American Petroleum Institute gravity) to gamma_oil (oil relative density by water)
    :param value: density in API(American Petroleum Institute gravity)
    :return: oil relative density by water
    """
    return (value + 131.5) / 141.5


def __gamma_oil2api(value):
    """
    converts density in API(American Petroleum Institute gravity) to gamma_oil (oil relative density by water)
    :param value: oil relative density by water
    :return: density in API(American Petroleum Institute gravity)
    """
    return 141.5 / value - 131.5


def __kgm3_2_lbft3(value):
    """
    converts density in kg/m3(kilogrammes/cubic meter) to lb/ft3(pounds/cubic feet)
    :param value: density in kgm3(kilogrammes/cubic meter)
    :return: density in lb/ft3(pounds/cubic feet)
    """
    return value * (const.foot**3 / const.lb)


def __lbft3_2_kgm3(value):
    """
    converts density in lb/ft3(pounds/cubic feet) to kgm3(kilogrammes/cubic meter)
    :param value: density in lb/ft3(pounds/cubic feet)
    :return: density in kgm3(kilogrammes/cubic meter)
    """
    return value / (const.foot**3 / const.lb)


# compressibility


def __compr_1pa_2_1psi(value):
    """
    converts compressibility in 1/pa to 1/psi
    :param value: compressibility in 1/pa
    :return: compressibility in 1/psi
    """
    return value * const.psi


def __compr_1psi_2_1pa(value):
    """
    converts compressibility in 1/psi to 1/pa
    :param value: compressibility in 1/psi
    :return: compressibility in 1/pa
    """
    return value / const.psi


def __compr_1pa_2_1bar(value):
    """
    converts compressibility in 1/pa to 1/bar
    :param value: compressibility in 1/pa
    :return: compressibility in 1/bar
    """
    return value * const.bar


def __compr_1bar_2_1pa(value):
    """
    converts compressibility in 1/bar to 1/pa
    :param value: compressibility in 1/bar
    :return: compressibility in 1/pa
    """
    return value / const.bar


def __compr_1mpa_2_1bar(value):
    """
    converts compressibility in 1/mpa to 1/bar
    :param value: compressibility in 1/mpa
    :return: compressibility in 1/bar
    """
    return value * const.bar / const.mega


def __compr_1bar_2_1mpa(value):
    """
    converts compressibility in 1/bar to 1/mpa
    :param value: compressibility in 1/bar
    :return: compressibility in 1/mpa
    """
    return value * const.mega / const.bar


def __btulbmF2kJkgK(value):
    """
    converts heat capacity from FIELD to SI
    :param value: heat capacity in Btu/lbm F
    :return: heat capacity in kJ/kgK equal to kJ/kgC
    """
    return value * 4.186


def __dyncm2nm(value):
    """
    converts surface tension
    :param value: surface tension in dyn / cm
    :return: surface tension in N / m
    """
    return value / 1000


def m3day2m3sec(value):
    """
    comverts flow rate
    :param value: flow rate in m3  / day
    :return: flow rate in m3  / sec
    """
    return value / 86400


def __cP2pasec(value):
    """
    converts viscosity
    :param value: viscosity in cP
    :return: viscosity in pa * sec
    """
    return value / 1000


def __grad2rad(value):
    """
    converts angle
    :param value: angle in grad
    :return: angle in rad
    """
    return value * pi / 180
