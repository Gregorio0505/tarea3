# Uso de IA: para ordenar el script siguiendo los incisos 0.2 a 0.5 y para el
# formato de las tablas en LaTeX (promedio con su error estandar entre parentesis). Los
# escenarios, el numero de replicas y las semillas salen del enunciado y de la semilla del
# trabajo; la lectura de cada tabla la hice yo.
#
# Script de la parte 0 (procesos simulados): grafica una realizacion de cada proceso, corre
# 5000 replicas por escenario e imprime en LaTeX las tablas de los incisos 0.2, 0.4 y 0.5.
# Tambien comprueba que la Ljung-Box propia coincide con la de statsmodels.

import re
from pathlib import Path

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

import autocorrelacion as ac
import graficas
import procesos_simulados as ps

# Semilla base del trabajo: 552003 (5 de mayo de 2003), la misma de la Tarea 2. Cada escenario
# usa un multiplo distinto, para que ninguno comparta numeros aleatorios con otro.
SEMILLA = 552003
REPLICAS = 5000
K_MAX = 21
PHIS = [0.0, 0.5, 0.9, 1.0]
SEMILLAS_AR1 = {phi: SEMILLA * (i + 1) for i, phi in enumerate(PHIS)}   # x1 .. x4
SEMILLA_TENDENCIA = SEMILLA * 5
SEMILLA_SEMANAL = SEMILLA * 6
M_FPP2 = 10                            # m que recomienda FPP2 sin estacionalidad
M_TSAY = int(round(np.log(ps.N)))      # m ~ log(n) que cita Tsay: log(300) = 5.7 -> 6

FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def escribir(nombre, texto):
    # las tablas van dentro de main.tex, asi que el script las imprime para copiarlas de aqui
    print(f"\n--- tabla LaTeX: {nombre} ---\n{texto}--- fin de la tabla ---\n")


def tabla(formato, encabezado, filas):
    return ("\\begin{tabular}{|" + "|".join(re.findall(r"p\{[^}]*\}|\S", formato)) + "|}\n\\hline\n" + encabezado +
            " \\\\\n\\hline\n" + "\n".join(filas) + "\n\\hline\n\\end{tabular}\n")


def media_ee(x, cifras=4):
    # promedio de las replicas y su error estandar s/sqrt(R)
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{cifras}f} ({x.std(ddof=1) / np.sqrt(len(x)):.{cifras}f})"


def tasa_ee(rechazos):
    # una fraccion es el promedio de indicadoras: su error estandar es sqrt(p(1-p)/R)
    p = np.mean(rechazos)
    return f"{p:.4f} ({np.sqrt(p * (1 - p) / len(rechazos)):.4f})"


def resumen(matriz):
    # para cada replica: r_1..r_21, la decision de la prueba de retardo 1 y el phi de OLS
    r = np.array([ac.acf(y, K_MAX) for y in matriz])
    rechazo = np.abs(r[:, 0]) > ac.banda(matriz.shape[1])
    phi = np.array([ps.ols_ar1(y)[1] for y in matriz])
    return {"r": r, "rechazo": rechazo, "phi": phi}


def tasa_ljung_box(matriz, m):
    return np.array([ac.ljung_box(y, m)[2] < ac.ALFA for y in matriz])


def main():
    FIGURAS.mkdir(exist_ok=True)

    # 0.2: una realizacion de cada phi (la primera replica de cada escenario)
    sim = {phi: ps.simular_ar1(phi, REPLICAS, np.random.default_rng(SEMILLAS_AR1[phi]))
           for phi in PHIS}
    nombres = {0.0: "$\\phi=0$ (ruido blanco)", 0.5: "$\\phi=0.5$", 0.9: "$\\phi=0.9$",
               1.0: "$\\phi=1$ (caminata aleatoria)"}
    graficas.realizaciones([(sim[phi][0], nombres[phi], None) for phi in PHIS],
                           FIGURAS / "realizaciones_ar1.png")

    res = {phi: resumen(sim[phi]) for phi in PHIS}
    filas = []
    for phi in PHIS:
        s = res[phi]
        filas.append(f"{phi:g} & {media_ee(s['r'][:, 0])} & {media_ee(s['r'][:, 4])} & "
                     f"{phi ** 5:.4f} & {tasa_ee(s['rechazo'])} & {media_ee(s['phi'])} \\\\")
    escribir("tabla_ar1 (0.2)", tabla(
        "cccccc", "$\\phi$ & $\\bar r_1$ & $\\bar r_5$ & $\\phi^5$ & "
        "Rechazo $|r_1|>1.96/\\sqrt{n}$ & $\\bar{\\hat\\phi}_{\\text{OLS}}$", filas))

    # 0.3: sesgo del OLS y del r_1 contra lo que dice la teoria
    for phi in PHIS:
        s = res[phi]
        print(f"phi={phi}: sesgo OLS = {s['phi'].mean() - phi:+.4f}, "
              f"aprox. -(1+3phi)/n = {-(1 + 3 * phi) / ps.N:+.4f}, "
              f"r1 - phi = {s['r'][:, 0].mean() - phi:+.4f}")

    # 0.4: procesos sin memoria con media que cambia
    m_tend = ps.media_tendencia()
    m_sem = ps.media_semanal()
    tend = ps.simular_sin_memoria(m_tend, REPLICAS, np.random.default_rng(SEMILLA_TENDENCIA))
    sem = ps.simular_sin_memoria(m_sem, REPLICAS, np.random.default_rng(SEMILLA_SEMANAL))
    graficas.realizaciones([(tend[0], "(a) Tendencia lineal, $m_t = 2t/300$", m_tend),
                            (sem[0], "(b) Patrón semanal, $m_t=-1.5$ si $t\\ \\mathrm{mod}\\ 7 = 6$", m_sem)],
                           FIGURAS / "realizaciones_sin_memoria.png")

    procesos = [("Ruido blanco ($\\phi=0$)", sim[0.0], res[0.0]),
                ("(a) Tendencia lineal", tend, resumen(tend)),
                ("(b) Patrón semanal", sem, resumen(sem))]
    filas = []
    for nombre, _, s in procesos:
        filas.append(f"{nombre} & {media_ee(s['r'][:, 0])} & {media_ee(s['r'][:, 4])} & "
                     f"{media_ee(s['r'][:, 6])} & {tasa_ee(s['rechazo'])} & "
                     f"{media_ee(s['phi'])} \\\\")
    escribir("tabla_sin_memoria (0.4)", tabla(
        "lccccc", "Proceso & $\\bar r_1$ & $\\bar r_5$ & $\\bar r_7$ & "
        "Rechazo $|r_1|>1.96/\\sqrt{n}$ & $\\bar{\\hat\\phi}_{\\text{OLS}}$", filas))

    # ACF promedio de los seis procesos, para comparar formas en 0.4 y en la parte 4
    paneles = [(res[phi]["r"].mean(axis=0), ps.N, nombres[phi], phi ** np.arange(1, K_MAX + 1))
               for phi in PHIS]
    paneles += [(procesos[1][2]["r"].mean(axis=0), ps.N, "(a) Tendencia lineal", None),
                (procesos[2][2]["r"].mean(axis=0), ps.N, "(b) Patrón semanal", None)]
    graficas.acf_paneles(paneles, FIGURAS / "acf_promedio_procesos_simulados.png", columnas=2,
                         alto=2.5)

    # 0.5: Ljung-Box con m = 10 (FPP2) y m = 6 (log n, Tsay), contra el retardo 1
    filas = []
    for nombre, matriz, s in procesos:
        filas.append(f"{nombre} & {tasa_ee(s['rechazo'])} & "
                     f"{tasa_ee(tasa_ljung_box(matriz, M_TSAY))} & "
                     f"{tasa_ee(tasa_ljung_box(matriz, M_FPP2))} \\\\")
    escribir("tabla_ljung_box (0.5)", tabla(
        "lccc", f"Proceso & Retardo 1 & Ljung--Box $m={M_TSAY}$ & Ljung--Box $m={M_FPP2}$", filas))

    # comprobacion contra statsmodels sobre todas las replicas de los tres procesos
    dif = 0.0
    for _, matriz, _ in procesos:
        for y in matriz:
            q_sm = acorr_ljungbox(y, lags=[M_FPP2])["lb_stat"].iloc[0]
            dif = max(dif, abs(q_sm - ac.ljung_box(y, M_FPP2)[0]))
    print(f"Maxima diferencia |Q propio - Q statsmodels| en {3 * REPLICAS} series: {dif:.2e}")


if __name__ == "__main__":
    main()
