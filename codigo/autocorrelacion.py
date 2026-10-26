# Uso de IA: para pasar a numpy las formulas de r_k y de Ljung-Box que vienen en la
# seccion 2 del enunciado (FPP2 2.8-2.9 y Tsay 2.2) y para comprobar contra statsmodels que la
# Q calculada a mano da lo mismo. Las formulas y los grados de libertad son los del enunciado.
#
# Autocorrelacion muestral r_k, bandas de ruido blanco y prueba de Ljung-Box, escritas a mano.
# Es la misma funcion para los procesos simulados de la parte 0 y para la venta diaria de las
# partes 3 y 4.

import numpy as np
from scipy import stats

ALFA = 0.05
Z = 1.959964   # cuantil 0.975 de la normal estandar


def acf(y, k_max):
    # r_k = sum_{t=k+1}^n (y_t - ybar)(y_{t-k} - ybar) / sum_t (y_t - ybar)^2, para k = 1..k_max
    y = np.asarray(y, dtype=float)
    d = y - y.mean()
    denominador = np.sum(d * d)
    return np.array([np.sum(d[k:] * d[:-k]) / denominador for k in range(1, k_max + 1)])


def banda(n):
    # bajo ruido blanco r_k ~ N(0, 1/n), asi que el 95% cae dentro de +/- 1.96/sqrt(n)
    return Z / np.sqrt(n)


def prueba_retardo_1(y):
    # la prueba de independencia de la Tarea 2: rechaza si |r_1| sale de la banda
    return abs(acf(y, 1)[0]) > banda(len(y))


def ljung_box(y, m, parametros=0):
    # Q(m) = n(n+2) sum_{k=1}^m r_k^2 / (n-k); bajo H0 es chi-cuadrado con m - K grados de
    # libertad, donde K es el numero de parametros estimados del modelo que dejo los residuos
    y = np.asarray(y, dtype=float)
    n = len(y)
    r = acf(y, m)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, m + 1)))
    gl = m - parametros
    return q, gl, stats.chi2.sf(q, gl)
