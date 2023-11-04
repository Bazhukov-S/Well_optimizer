"""
Модуль с полезными функциями, используемыми в расчетных модулях
"""
import itertools as iter
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

import ffmt.pvt.adapter as fl
if TYPE_CHECKING:
    import unifloc.equipment.choke as ch
    import unifloc.equipment.esp_system as esp_sys
    import unifloc.pipe.pipeline as pipe

import unifloc.equipment.gl_valve as gl_vl
import unifloc.service._constants as const


def extract_output_fluid(fluid: fl.FluidFlow) -> np.ndarray:
    """
    Вывод необходимых параметров флюида

    Parameters
    ----------
    :param fluid: объект флюида для вывода
    :return: массив значений необходимых параметров
    """
    output_array = np.empty(len(const.DISTRS_PVT))
    output_array[0] = fluid.rs
    output_array[1] = fluid.pb
    output_array[2] = fluid.muo
    output_array[3] = fluid.mug
    output_array[4] = fluid.muw
    output_array[5] = fluid.mul
    output_array[6] = fluid.mum
    output_array[7] = fluid.z
    output_array[8] = fluid.bo
    output_array[9] = fluid.bg
    output_array[10] = fluid.bw
    output_array[11] = fluid.ro
    output_array[12] = fluid.rg
    output_array[13] = fluid.rw
    output_array[14] = fluid.rl
    output_array[15] = fluid.rm
    output_array[16] = fluid.co
    output_array[17] = fluid.qo
    output_array[18] = fluid.qg
    output_array[19] = fluid.qw
    output_array[20] = fluid.ql
    output_array[21] = fluid.qm
    output_array[22] = fluid.gf
    output_array[23] = fluid.stog
    output_array[24] = fluid.stwg
    output_array[25] = fluid.stlg
    return output_array


def make_unified_distributions(
        casing: Optional[List["pipe.Pipeline"]] = None,
        tubings: Optional[List["pipe.Pipeline"]] = None,
        annulus: Optional[List["pipe.Pipeline"]] = None,
        choke: Optional["ch.Choke"] = None,
        ann_choke: Optional["ch.Choke"] = None,
        gl_valves: Optional[List["gl_vl.GlValve"]] = None,
        esp_sys: Optional["esp_sys.EspSystem"] = None,
        params: Optional[list] = None,
        flag_ann: Optional[bool] = False,
) -> dict:
    """
    Функция, создающая сборные распределения

    :param casing: объект ЭК
    :param tubings: объекты НКТ
    :param annulus: объект затрубного пространства
    :param choke: объект штуцера
    :param ann_choke: объект штуцера на линии затруба
    :param gl_valves: объекты газлифтных клапанов
    :param esp_sys: объект УЭЦН
    :param params: список распределений для сохранения
    :param flag_ann: флаг, необходимый для корректного построения распределения в затрубе
    :return: словарь с сборными распределениями
    """
    result = dict()
    distr_objects = []

    if not params:
        params = const.DISTRS

    for par in params:
        if casing is not None:
            distr_objects = [casing]

        if tubings is not None:
            distr_objects[0:0] = [tubing for tubing in tubings]

        if annulus is not None:
            distr_objects = [annulus]

        if gl_valves is not None:
            if isinstance(gl_valves, list):
                distr_objects[0:0] = [gl_valve for gl_valve in gl_valves]
            else:
                distr_objects.insert(-2, gl_valves)

        if choke:
            distr_objects.insert(0, choke)

        if ann_choke:
            for el in ann_choke.distributions:
                if ann_choke.distributions[el] is not None:
                    ann_choke.distributions[el][0], ann_choke.distributions[el][1] = (
                        ann_choke.distributions[el][1],
                        ann_choke.distributions[el][0],
                    )
            distr_objects.insert(0, ann_choke)

        if esp_sys:
            distr_objects.insert(-1, esp_sys)

        distrs = []
        for obj in distr_objects:
            if isinstance(obj, gl_vl.GlValve) and casing is None:
                distrs.append(obj.distributions_annulus.get(par))
            else:
                distrs.append(obj.distributions.get(par))

        non_existence_flags = [v is None for v in distrs]

        if all(non_existence_flags):
            result[par] = None
            continue
        else:
            for i, _ in iter.compress(enumerate(distrs), non_existence_flags):
                distrs[i] = np.full(len(distr_objects[i].distributions["depth"]), np.NAN)

        unified_distr = np.concatenate(distrs)

        if all(np.isnan(unified_distr)) or (all(unified_distr) == 0.0 and par in const.DISTRS_NONE and flag_ann):
            result[par] = None
        else:
            if par == "depth":
                unified_distr *= -1

            unified_distr = np.where(np.isnan(unified_distr), None, unified_distr)
            result[par] = unified_distr.tolist()

    if isinstance(gl_valves, list):
        result_df = pd.DataFrame(result)
        if choke or ann_choke:
            result_df_up = result_df[:2]
            result_df_dwn = result_df[2:]
            result_df_dwn = result_df_dwn.sort_values(by="depth", ascending=False)
            result_df = pd.concat([result_df_up, result_df_dwn])
        else:
            result_df = result_df.sort_values(by="depth", ascending=False)
        result_df = result_df.replace({np.nan: None})
        result = result_df.to_dict("list")

    return result


def check_nan(distr_dict: dict) -> dict:
    """
    Проверка распределений на существование

    :param distr_dict: словарь с распределениями
    :return: словарь с распределениям без NaN распределений
    """
    for k in distr_dict:
        if all(np.isnan(distr_dict[k])):
            distr_dict[k] = None

    return distr_dict


def make_output_attrs(
        fluid: fl.FluidFlow, p_array: np.ndarray, t_array: np.ndarray
) -> dict:
    """
    Функция для сохранения экстра-выводных параметров в словарь

    Parameters
    ----------
    :param fluid: объект флюида
    :param p_array: массив давлений, Па
    :param t_array: массив температур, К
    """
    result_data = np.empty([len(const.DISTRS_PVT), len(p_array)])

    for i, p in enumerate(p_array):
        fluid.calc_flow(p, t_array[i])
        result_data[:, i] = extract_output_fluid(fluid)

    result = {k: result_data[i] for i, k in enumerate(const.DISTRS_PVT)}

    # проверка распределений на существование
    result = check_nan(result)

    return result
