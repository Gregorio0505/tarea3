# Uso de IA: para ordenar el script siguiendo los incisos 3.1 a 5.5, para
# imprimir las tablas en LaTeX y para dudas de statsmodels (nombres de los parametros,
# mle_retvals y forecast de varios pasos). Que se mide en cada inciso, las fechas de los
# origenes y la interpretacion de los resultados son mias.
#
# Script de las partes 3, 4 y 5: descomposicion (tendencia y estacionalidad), autocorrelacion,
# del AR(1) al SARIMA, y la comparacion de los seis modelos con las tres formas de validar.
# Tarda unos minutos porque reajusta S1 y S2 en cada origen semanal.

import re
from pathlib import Path

import numpy as np
import pandas as pd

import autocorrelacion as ac
import graficas
import limpieza_datos as ld
import modelos_pronostico as mp
import procesos_simulados as ps
import serie_diaria as sd
import validacion_pronostico as vp

SEMILLA = 552003
SEMILLAS_CV = [SEMILLA * k for k in range(1, 11)]   # 10 repeticiones de la validacion cruzada
K_MAX = 21
MODELOS = ["B1", "B2", "M1", "M2", "S1", "S2"]
ORIGENES_ENTRENAMIENTO = vp.domingos("2011-05-01", "2011-09-25")   # mayo a septiembre
ORIGENES_PRUEBA = vp.domingos(sd.ORIGEN_PRUEBA, "2011-12-04")
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def escribir(nombre, texto):
    print(f"\n--- tabla LaTeX: {nombre} ---\n{texto}--- fin de la tabla ---\n")


def tabla(formato, encabezado, filas):
    return ("\\begin{tabular}{|" + "|".join(re.findall(r"p\{[^}]*\}|\S", formato)) + "|}\n\\hline\n" + encabezado +
            " \\\\\n\\hline\n" + "\n".join(filas) + "\n\\hline\n\\end{tabular}\n")


def miles(x):
    return f"{x:,.0f}".replace(",", "\\,")


def resumen_acf(nombre, r, n):
    fuera = np.where(np.abs(r) > ac.banda(n))[0] + 1
    print(f"  {nombre}: n = {n}, banda = {ac.banda(n):.4f}, rezagos fuera = {len(fuera)} "
          f"{fuera.tolist()}, pico en k = {np.argmax(r) + 1} (r = {r.max():.3f}), "
          f"r_1 = {r[0]:.3f}")


def imprimir_ljung_box(nombre, y, parametros=0):
    for m in (6, 12):
        q, gl, p = ac.ljung_box(y, m, parametros)
        print(f"  {nombre}: Q({m}) = {q:.2f}, gl = {gl}, p = {p:.3g}")


def main():
    limpio = ld.limpiar(ld.cargar_datos())
    serie = sd.serie_calendario(limpio, hasta=sd.FIN_PRUEBA)   # sin el 9 de diciembre
    ent = sd.entrenamiento(serie)                             # una fila por dia abierto
    ent_cal = serie.loc[:sd.FIN_ENTRENAMIENTO]                # el mismo tramo sobre el calendario

    # 3.1 tendencia
    mm = sd.media_movil_centrada(serie)
    graficas.serie_con_media_movil(serie, mm, FIGURAS / "serie_diaria_media_movil.png",
                                   sd.FIN_ENTRENAMIENTO)
    print("3.1 media movil promedio por mes (miles de libras):")
    print((mm.groupby(mm.index.to_period("M")).mean() / 1000).round(1).to_dict())

    # 3.2 estacionalidad
    indice = sd.indice_estacional(serie, mm)
    nombres = sd.nombres_dias(indice.index)
    graficas.barras_indice(nombres, indice.values, FIGURAS / "indice_estacional.png")
    filas = [" & ".join(nombres) + " & Promedio \\\\",
             " & ".join(f"{v:.3f}" for v in indice.values) + f" & {indice.mean():.3f} \\\\"]
    escribir("tabla_indice_estacional (3.2)",
             "\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n" + "\n\\hline\n".join(filas) +
             "\n\\hline\n\\end{tabular}\n")
    print(f"  7/6 = {7 / 6:.3f}")

    # 3.3 autocorrelacion de la venta, una fila por dia abierto
    print("3.3")
    r_venta = ac.acf(ent.values, K_MAX)
    resumen_acf("venta", r_venta, len(ent))
    print(f"  prueba de retardo 1 de la Tarea 2: rechaza = {ac.prueba_retardo_1(ent.values)}")
    imprimir_ljung_box("venta", ent.values)
    print(f"  log(n) = {np.log(len(ent)):.2f}")
    graficas.acf_paneles([(r_venta, len(ent), "Venta neta diaria (días abiertos del entrenamiento)",
                           None)], FIGURAS / "acf_venta_entrenamiento.png")

    # 4.1 OLS contra indicadoras de dia de la semana, sin intercepto
    print("4.1")
    x = mp.indicadoras_dia(ent.index)
    beta = mp.ols(x, ent.values)
    e = ent.values - x @ beta
    print("  medias por dia:", dict(zip(nombres, np.round(beta, 0))))
    r_e = ac.acf(e, K_MAX)
    resumen_acf("residuos dia de la semana", r_e, len(e))
    imprimir_ljung_box("residuos dia de la semana", e)

    # 4.2 AR(1) sobre esos residuos
    print("4.2")
    c, phi, u = ps.ols_ar1(e)
    print(f"  c = {c:.2f}, phi = {phi:.4f}")
    r_u = ac.acf(u, K_MAX)
    resumen_acf("residuos AR(1)", r_u, len(u))
    imprimir_ljung_box("residuos AR(1)", u, parametros=1)
    graficas.acf_paneles([(r_e, len(e), "(a) Residuos de la regresión contra el día de la semana",
                           None),
                          (r_u, len(u), f"(b) Residuos del AR(1) ajustado a (a), "
                                        f"$\\hat\\phi = {phi:.3f}$", None)],
                         FIGURAS / "acf_residuos_dia_semana_ar1.png")

    # 4.3 SARIMA sobre el logaritmo, con el arreglo de valores
    print("4.3")
    y_log = np.log(ent.values)
    ajustes = {}
    filas = []
    for nombre in ("S1", "S2"):
        for maxiter in (50, mp.MAXITER):
            res = mp.ajustar_sarima(nombre, y_log, maxiter)
            rv = res.mle_retvals
            print(f"  {nombre} maxiter={maxiter}: convergio = {rv['converged']}, "
                  f"iteraciones = {rv.get('iterations')}, log-veros. = {res.llf:.2f}, "
                  f"parametros = {dict(zip(res.param_names, np.round(res.params, 4)))}")
            parametros = ", ".join(f"{n} = {v:.4f}" for n, v in zip(res.param_names, res.params))
            filas.append(f"{nombre} & {maxiter} & {'Sí' if rv['converged'] else 'No'} & "
                         f"{rv.get('iterations')} & {res.llf:.2f} & "
                         f"\\texttt{{{parametros.replace('_', chr(92) + '_')}}} \\\\")
            ajustes[nombre] = res
    escribir("tabla_sarima (4.3)", tabla("ccccc p{7cm}", "Modelo & \\texttt{maxiter} & Convergió & "
                                         "Iteraciones & Log-verosimilitud & Parámetros", filas))
    # cuantas filas tienen en t-6 un dia de la semana distinto (s = 6 se rompe)
    dia = ent.index.dayofweek.values
    rotas = dia[6:] != dia[:-6]
    print(f"  filas cuyo rezago 6 no es el mismo dia de la semana: {rotas.sum()} de {len(rotas)}")
    s1 = ajustes["S1"].params
    mu = s1[0] / ((1 - s1[1]) * (1 - s1[2]))
    print(f"  S1: media de largo plazo en log = {mu:.4f}, exp = {np.exp(mu):.2f}; "
          f"media geometrica del entrenamiento = {np.exp(y_log.mean()):.2f}")

    # 4.4 diagnostico: Ljung-Box con m - 2 grados de libertad
    print("4.4")
    res_s1 = ajustes["S1"].resid
    res_s2 = ajustes["S2"].resid[7:]   # las diferencias (1-B)(1-B^6) se comen 7 observaciones
    imprimir_ljung_box("residuos S1", res_s1, parametros=2)
    imprimir_ljung_box("residuos S2", res_s2, parametros=2)
    graficas.acf_paneles([(ac.acf(res_s1, K_MAX), len(res_s1),
                           "(a) Residuos de S1: SARIMA$(1,0,0)(1,0,0)_6$", None),
                          (ac.acf(res_s2, K_MAX), len(res_s2),
                           "(b) Residuos de S2: SARIMA$(0,1,1)(0,1,1)_6$", None)],
                         FIGURAS / "acf_residuos_sarima.png")
    resumen_acf("residuos S1", ac.acf(res_s1, K_MAX), len(res_s1))
    resumen_acf("residuos S2", ac.acf(res_s2, K_MAX), len(res_s2))

    # 4.5 hacia donde van los pronosticos de largo plazo
    pasos = 120
    x_hist = np.arange(len(ent) - 60, len(ent))
    x_fut = np.arange(len(ent), len(ent) + pasos)
    paneles = []
    for nombre, titulo in (("S1", "S1 (estacionario)"), ("S2", "S2 (no estacionario)")):
        p = np.exp(ajustes[nombre].forecast(pasos))
        paneles.append((x_fut, p, titulo))
        print(f"  {nombre}: pronostico a 1-6 pasos {np.round(p[:6], 0).tolist()}, "
              f"a 115-120 pasos {np.round(p[-6:], 0).tolist()}")
    graficas.pronostico_largo((x_hist, ent.values[-60:]), paneles,
                              FIGURAS / "pronostico_largo_s1_s2.png")

    # 5.2 (a) validacion cruzada aleatoria, solo B1, M1 y M2
    cv = {}
    for m in ("B1", "M1", "M2"):
        resultados = np.array([vp.validacion_cruzada(ent_cal, m, s) for s in SEMILLAS_CV])
        cv[m] = resultados
    # (b) y (c) origen movil semanal
    pron_b, nc_b = vp.origen_movil(serie, MODELOS, ORIGENES_ENTRENAMIENTO, sd.FIN_ENTRENAMIENTO)
    pron_c, nc_c = vp.origen_movil(serie, MODELOS, ORIGENES_PRUEBA, sd.FIN_PRUEBA)
    print(f"5.1 origenes (b): {len(ORIGENES_ENTRENAMIENTO)}, de {ORIGENES_ENTRENAMIENTO[0]:%d/%m} a "
          f"{ORIGENES_ENTRENAMIENTO[-1]:%d/%m}; dias evaluados {len(pron_b['B1'])}")
    print(f"    origenes (c): {len(ORIGENES_PRUEBA)}; dias evaluados {len(pron_c['B1'])}")
    print(f"    ajustes SARIMA que no convergieron: (b) {nc_b}, (c) {nc_c}")

    filas = []
    for m in MODELOS:
        if m in cv:
            a = (f"{miles(cv[m][:, 0].mean())} ({miles(cv[m][:, 0].std(ddof=1))}) & "
                 f"{miles(cv[m][:, 1].mean())} ({miles(cv[m][:, 1].std(ddof=1))})")
        else:
            a = "-- & --"
        rb, rc = serie.loc[pron_b[m].index], serie.loc[pron_c[m].index]
        filas.append(f"{m} & {a} & {miles(vp.mae(rb, pron_b[m]))} & {miles(vp.rmse(rb, pron_b[m]))}"
                     f" & {miles(vp.mae(rc, pron_c[m]))} & {miles(vp.rmse(rc, pron_c[m]))} \\\\")
    escribir("tabla_validacion (5.2)", tabla(
        "lcccccc", "& \\multicolumn{2}{c|}{(a) CV aleatoria} & \\multicolumn{2}{c|}{(b) Origen móvil "
        "(entrenamiento)} & \\multicolumn{2}{c|}{(c) Prueba} \\\\\n\\cline{2-3}"
        "\\cline{4-5}\\cline{6-7}\nModelo & MAE & RMSE & MAE & RMSE & MAE & RMSE",
        filas))

    # 5.3 grafica de la prueba y sesgo de cada modelo
    real = serie.loc[pron_c["B1"].index]
    graficas.pronosticos_prueba(real, pron_c, FIGURAS / "pronosticos_prueba.png")
    print("5.3 sesgo medio en prueba (pronostico - real) y MAE por mes:")
    for m in MODELOS:
        err = pron_c[m] - real
        por_mes = err.abs().groupby(err.index.month).mean().round(0).to_dict()
        print(f"  {m}: sesgo = {err.mean():.0f}, MAE por mes = {por_mes}")
    print(f"  venta media: entrenamiento {ent.mean():.0f}, mayo-sep {ent.loc['2011-05':].mean():.0f},"
          f" prueba {real.mean():.0f}")

    # 5.4 la trampa: la media movil centrada calculada una vez sobre toda la serie
    print("5.4")
    cv_ma = np.array([vp.validacion_cruzada(ent_cal, "M2+MA", s, media_movil=mm) for s in SEMILLAS_CV])
    pron_ma, _ = vp.origen_movil(serie, ["M2+MA"], ORIGENES_ENTRENAMIENTO, sd.FIN_ENTRENAMIENTO,
                                 media_movil=mm)
    rb = serie.loc[pron_ma["M2+MA"].index]
    filas = []
    for m, a, pb in (("M2", cv["M2"], pron_b["M2"]), ("M2 + media móvil", cv_ma, pron_ma["M2+MA"])):
        rb = serie.loc[pb.index]
        filas.append(f"{m} & {miles(a[:, 0].mean())} ({miles(a[:, 0].std(ddof=1))}) & "
                     f"{miles(a[:, 1].mean())} ({miles(a[:, 1].std(ddof=1))}) & "
                     f"{miles(vp.mae(rb, pb))} & {miles(vp.rmse(rb, pb))} \\\\")
    escribir("tabla_trampa (5.4)", tabla(
        "lcccc", "& \\multicolumn{2}{c|}{(a) CV aleatoria} & \\multicolumn{2}{c|}{(b) Origen móvil} "
        "\\\\\n\\cline{2-3}\\cline{4-5}\nModelo & MAE & RMSE & MAE & RMSE", filas))
    print(f"  la media movil de los ultimos dias de la prueba: {mm.loc['2011-12-04':].round(0).to_dict()}")
    # lo que daria (c) haciendo trampa, solo en los dias de prueba que tienen media movil
    pron_ma_c, _ = vp.origen_movil(serie, ["M2+MA"], ORIGENES_PRUEBA, sd.FIN_PRUEBA, media_movil=mm)
    p = pron_ma_c["M2+MA"].dropna()
    print(f"  (c) con trampa en {len(p)} de {len(real)} dias: MAE = {vp.mae(serie.loc[p.index], p):.0f}"
          f", RMSE = {vp.rmse(serie.loc[p.index], p):.0f}; M2 en esos dias: MAE = "
          f"{vp.mae(serie.loc[p.index], pron_c['M2'].loc[p.index]):.0f}")

    # 5.5 cuanto cambio el nivel
    print("5.5 venta media por mes (dias abiertos):")
    abiertos = serie.dropna()
    print((abiertos.groupby(abiertos.index.to_period("M")).mean() / 1000).round(1).to_dict())


if __name__ == "__main__":
    main()
