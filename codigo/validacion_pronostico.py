# Uso de IA: para escribir el ciclo del origen movil a partir de las reglas del
# enunciado y la validacion cruzada partiendo los indices a mano con numpy (permutacion y
# array_split) en lugar de scikit-learn. Que dias se evaluan y las semillas son las del
# enunciado y del trabajo.
#
# Las tres formas de medir el error de la parte 5: validacion cruzada aleatoria de 5 folds, origen
# movil semanal dentro del entrenamiento y origen movil semanal en la prueba. El MAE y el RMSE se
# calculan juntando todos los dias pronosticados.

import numpy as np
import pandas as pd

import modelos_pronostico as mp


def mae(real, pred):
    return float(np.mean(np.abs(np.asarray(real) - np.asarray(pred))))


def rmse(real, pred):
    return float(np.sqrt(np.mean((np.asarray(real) - np.asarray(pred)) ** 2)))


def domingos(desde, hasta):
    return pd.date_range(desde, hasta, freq="W-SUN")


def objetivos_de_la_semana(serie, origen, limite):
    # dias abiertos del lunes al domingo siguientes al origen, sin pasar del limite
    semana = serie.loc[origen + pd.Timedelta(days=1):min(origen + pd.Timedelta(days=7), limite)]
    return semana.dropna().index


def origen_movil(serie, modelos, origenes, limite, media_movil=None):
    # devuelve, por modelo, una Serie con el pronostico de cada dia evaluado y los avisos de
    # convergencia de los SARIMA
    pronosticos = {m: [] for m in modelos}
    no_convergio = []
    for origen in origenes:
        historia = serie.loc[:origen]   # el domingo del origen ya esta observado
        objetivos = objetivos_de_la_semana(serie, origen, limite)
        if len(objetivos) == 0:
            continue
        fechas_hist = historia.dropna().index
        for m in modelos:
            if m in mp.SARIMA:
                p, convergio = mp.pronostico_sarima(m, historia, len(objetivos))
                if not convergio:
                    no_convergio.append((m, origen))
            elif m == "B2":
                p = mp.naive_estacional(historia, objetivos)
            else:
                # los rezagos de los objetivos caen en la historia, porque estan a 7 o mas dias
                p = mp.ajustar_y_predecir(m, historia, fechas_hist, objetivos, media_movil)
            pronosticos[m].append(pd.Series(p, index=objetivos))
    return {m: pd.concat(v) for m, v in pronosticos.items()}, no_convergio


def validacion_cruzada(serie_ent, modelo, semilla, folds=5, media_movil=None):
    # 5 folds al azar sobre las filas del entrenamiento; cada fila se predice una vez con el
    # modelo ajustado en los otros cuatro folds
    fechas = serie_ent.dropna().index
    if modelo != "B1" and modelo != "M1":
        # las filas sin rezago no se pueden ajustar ni predecir con M2
        x = mp.matriz_diseno(modelo, serie_ent, fechas, media_movil)
        fechas = fechas[~np.isnan(x).any(axis=1)]
    orden = np.random.default_rng(semilla).permutation(len(fechas))
    pred = pd.Series(np.nan, index=fechas)
    for prueba in np.array_split(orden, folds):
        ajuste = np.setdiff1d(orden, prueba)
        pred.iloc[prueba] = mp.ajustar_y_predecir(modelo, serie_ent, fechas[ajuste], fechas[prueba],
                                                  media_movil)
    real = serie_ent.loc[fechas]
    return mae(real, pred), rmse(real, pred)
