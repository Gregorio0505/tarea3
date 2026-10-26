# Uso de IA: para que los modelos de regresion compartan una misma funcion que arma
# la matriz de diseno (asi se usa igual en la validacion cruzada y en el origen movil) y para
# la interfaz de SARIMAX de statsmodels (forecast, mle_retvals). La definicion de cada modelo y
# la regla del rezago en dias cerrados son las del enunciado.
#
# Modelos de pronostico de la parte 5: B1 (media historica), B2 (naive estacional), M1
# (tendencia + dia de la semana), M2 (dia de la semana + rezagos 7 y 14), la version con la
# media movil centrada (la trampa de 5.4) y los SARIMA S1 y S2 de la parte 4.

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import serie_diaria as sd

MAXITER = 500   # con el valor por defecto (50) S1 no converge; ver inciso 4.3
SARIMA = {
    "S1": dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 6), trend="c"),
    "S2": dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 6), trend="n"),
}


def rezago_mismo_dia(serie, fecha, rezago):
    # y_{t-rezago}; si ese dia estuvo cerrado se usa el mismo dia de la semana abierto mas
    # reciente (para el lunes 9 de mayo de 2011, y_{t-7} es el lunes 18 de abril)
    d = fecha - pd.Timedelta(days=rezago)
    while d >= serie.index[0]:
        valor = serie.get(d, np.nan)   # serie va sobre el calendario completo
        if not np.isnan(valor):
            return valor
        d -= pd.Timedelta(days=7)
    return np.nan   # se salio del inicio de la serie: esa fila no se usa


def indicadoras_dia(fechas):
    # una columna por dia abierto, sin intercepto
    return np.column_stack([(fechas.dayofweek == d).astype(float) for d in sd.DIAS_ABIERTOS])


def matriz_diseno(modelo, serie, fechas, media_movil=None):
    # serie es la historia de la que salen los rezagos; fechas son las filas que se quieren
    if modelo == "B1":
        return np.ones((len(fechas), 1))
    dias = indicadoras_dia(fechas)
    if modelo == "M1":
        t = np.asarray((fechas - sd.PRIMER_DIA).days, dtype=float)   # dias desde el 1/12/2010
        return np.column_stack((dias, t))
    rezagos = np.array([[rezago_mismo_dia(serie, f, 7), rezago_mismo_dia(serie, f, 14)]
                        for f in fechas])
    if modelo == "M2":
        return np.column_stack((dias, rezagos))
    if modelo == "M2+MA":
        return np.column_stack((dias, rezagos, media_movil.reindex(fechas).values))
    raise ValueError(modelo)


def ols(x, y):
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def ajustar_y_predecir(modelo, serie, fechas_ajuste, fechas_objetivo, media_movil=None):
    # OLS con las filas que tienen todos sus predictores; predice las fechas objetivo
    x = matriz_diseno(modelo, serie, fechas_ajuste, media_movil)
    y = serie.reindex(fechas_ajuste).values
    completas = ~np.isnan(x).any(axis=1)
    beta = ols(x[completas], y[completas])
    return matriz_diseno(modelo, serie, fechas_objetivo, media_movil) @ beta


def naive_estacional(serie, fechas_objetivo):
    # B2: el valor del mismo dia de la semana abierto mas reciente
    return np.array([rezago_mismo_dia(serie, f, 7) for f in fechas_objetivo])


def ajustar_sarima(nombre, y_log, maxiter=MAXITER):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # la convergencia se revisa con mle_retvals
        return SARIMAX(y_log, **SARIMA[nombre]).fit(disp=False, maxiter=maxiter)


def pronostico_sarima(nombre, historia, pasos):
    # se ajusta sobre log de la venta de los dias abiertos (los .values, sin fechas) y el
    # pronostico de h pasos es el del h-esimo dia abierto despues del origen
    res = ajustar_sarima(nombre, np.log(historia.dropna().values))
    return np.exp(res.forecast(pasos)), res.mle_retvals["converged"]
