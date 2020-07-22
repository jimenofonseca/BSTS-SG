import numpy as np
import pandas as pd


def calc_humidity_ratio(rh_percent, dry_bulb_C, patm_mbar):
    """
    convert relative humidity to moisture content
    Based on https://www.vaisala.com/sites/default/files/documents/Humidity_Conversion_Formulas_B210973EN.pdf
    """
    patm_hPa = patm_mbar

    A, m, Tn = get_phycometric_constants(dry_bulb_C)
    T_dry = dry_bulb_C

    p_ws_hPa = A * 10 ** ((m * T_dry) / (T_dry + Tn))
    p_w_hPa = p_ws_hPa * rh_percent / 100
    B_kgperkg = 0.6219907
    x_kgperkg = B_kgperkg * p_w_hPa / (patm_hPa - p_w_hPa)
    return x_kgperkg


def calc_h_sen(dry_bulb_C):
    """
    Calc specific temperature of moist air (sensible)
    """
    CPA_kJ_kgC = 1.006
    h_kJ_kg = dry_bulb_C * CPA_kJ_kgC

    return h_kJ_kg


def calc_h_lat(dry_bulb_C, humidity_ratio_out_kgperkg):
    """
    Calc specific temperature of moist air (latent)

    :param temperatures_out_C:
    :param CPA:
    :return:
    """
    CPW_kJ_kgC = 1.84
    h_we_kJ_kg = 2501

    h_kJ_kg = humidity_ratio_out_kgperkg * (dry_bulb_C * CPW_kJ_kgC + h_we_kJ_kg)

    return h_kJ_kg


def get_phycometric_constants(T_C):
    if -20 <= T_C <= 50:
        m = 7.591386
        Tn = 240.7263
        A = 6.116441
    elif -70 <= T_C <= 0:
        m = 9.778707
        Tn = 273.1466
        A = 6.114742

    return A, m, Tn


def calc_enthalpy_gradient_sensible(dry_bulb_C, dry_bulb_C_base_cooling):
    H_sen_outdoor_kjperkg = calc_h_sen(dry_bulb_C)

    # Cooling case
    AH_sensible_kJperKg = H_sen_outdoor_kjperkg - calc_h_sen(dry_bulb_C_base_cooling)
    if AH_sensible_kJperKg > 0.0:
        return abs(AH_sensible_kJperKg)
    else:
        return 0.0



def calc_enthalpy_gradient_latent(dry_bulb_C,
                                  x_kgperkg,
                                  dry_bulb_C_base_cooling,
                                  x_kgperkg_base_cooling):
    H_latent_outdoor_kjperkg = calc_h_lat(dry_bulb_C, x_kgperkg)

    # Cooling case
    AH_latent_kJperKg = H_latent_outdoor_kjperkg - calc_h_lat(dry_bulb_C_base_cooling, x_kgperkg_base_cooling)
    if AH_latent_kJperKg > 0.0:
        return abs(AH_latent_kJperKg)
    else:
        return 0.0


def daily_enthalpy_gradients_daily_data(dry_bulb_C: np.array,
                                        rh_percent: np.array):
    # Enhtalpy gradients
    dry_bulb_C_base_heating = 18.3
    dry_bulb_C_base_cooling = 21.3
    rh_percent_base_heating = 40
    rh_percent_base_cooling = 70
    patm_mbar = 1013.25
    patm_mbar_base = 1013.25

    # FEATURE 3 and 4: Enthalpy Sensible heat for heating and cooling seasons kg/kJ:
    DEG_C_kJperKg, \
    DEG_H_kJperKg = np.vectorize(calc_enthalpy_gradient_sensible, otypes=[np.float32, np.float32])(dry_bulb_C,
                                                                                                   dry_bulb_C_base_heating,
                                                                                                   dry_bulb_C_base_cooling)

    # FEATURE 5 and 6: Enthalpy Sensible heat for heating and cooling seasons kg/kJ:
    x_kgperkg_base_heating = calc_humidity_ratio(rh_percent_base_heating, dry_bulb_C_base_heating, patm_mbar_base)
    x_kgperkg_base_cooling = calc_humidity_ratio(rh_percent_base_cooling, dry_bulb_C_base_cooling, patm_mbar_base)
    x_kgperkg = np.vectorize(calc_humidity_ratio, otypes=[np.float32])(rh_percent, dry_bulb_C, patm_mbar)

    DEG_DEHUM_kJperKg, \
    DEG_HUM_kJperKg = np.vectorize(calc_enthalpy_gradient_latent, otypes=[np.float32, np.float32])(dry_bulb_C,
                                                                                                   x_kgperkg,
                                                                                                   dry_bulb_C_base_heating,
                                                                                                   dry_bulb_C_base_cooling,
                                                                                                   x_kgperkg_base_heating,
                                                                                                   x_kgperkg_base_cooling)
    return DEG_C_kJperKg, DEG_H_kJperKg, DEG_DEHUM_kJperKg, DEG_HUM_kJperKg


def daily_enthalpy_gradients_hourly_data(timestamp,
                                         dry_bulb_C,
                                         rh_percent):
    # Enhtalpy gradients
    dry_bulb_C_base_cooling = 21
    rh_percent_base_cooling = 50
    patm_mbar = 1013.25
    patm_mbar_base = 1013.25

    # FEATURE 3 and 4: Enthalpy Sensible heat for cooling seasons kg/kJ:
    DEG_C_kJperKg = np.vectorize(calc_enthalpy_gradient_sensible)(dry_bulb_C, dry_bulb_C_base_cooling)

    # FEATURE 5 and 6: Enthalpy Sensible heat for heating and cooling seasons kg/kJ:
    x_kgperkg_base_cooling = calc_humidity_ratio(rh_percent_base_cooling, dry_bulb_C_base_cooling, patm_mbar_base)
    x_kgperkg = np.vectorize(calc_humidity_ratio)(rh_percent, dry_bulb_C, patm_mbar)

    DEG_DEHUM_kJperKg = np.vectorize(calc_enthalpy_gradient_latent)(dry_bulb_C,
                                                                    x_kgperkg,
                                                                    dry_bulb_C_base_cooling,
                                                                    x_kgperkg_base_cooling)

    result = pd.DataFrame({'DEG_C_kJperKg': DEG_C_kJperKg, 'DEG_DEHUM_kJperKg': DEG_DEHUM_kJperKg}, index=timestamp)
    result = result.resample('D').sum() / 24

    return result
