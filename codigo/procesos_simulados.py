# Uso de IA: para que la simulacion del AR(1) avance las 5000 replicas a la vez con
# numpy (una fila por replica) en lugar de un ciclo por replica, para que la parte 0 corra en
# segundos. La recursion, el calentamiento de 200 pasos y las dos medias m_t son las del
# enunciado; las derivaciones de 0.1 las hice a mano.
#
# Simulacion de los procesos de la parte 0: el AR(1) Y_t = c + phi Y_{t-1} + e_t y los
# procesos sin memoria Y_t = m_t + e_t con tendencia lineal o con patron semanal.
# Tambien el estimado OLS de phi, que es una regresion de y_t contra y_{t-1}.

import numpy as np

N = 300            # pasos que se conservan
CALENTAMIENTO = 200   # pasos que se descartan para que Y_0 = 0 deje de importar


def simular_ar1(phi, replicas, rng, n=N, calentamiento=CALENTAMIENTO, c=0.0, sigma=1.0):
    # devuelve una matriz replicas x n; cada fila es una realizacion que arranca en Y_0 = 0
    e = rng.normal(0.0, sigma, size=(replicas, calentamiento + n))
    y = np.zeros(replicas)
    salida = np.empty((replicas, n))
    for t in range(calentamiento + n):
        y = c + phi * y + e[:, t]
        if t >= calentamiento:
            salida[:, t - calentamiento] = y
    return salida


def media_tendencia(n=N):
    # (a) sube en linea recta de 0 a 2 a lo largo de los n pasos: m_t = 2t/300
    t = np.arange(1, n + 1)
    return 2 * t / n


def media_semanal(n=N):
    # (b) un dia de cada siete vende mucho menos: m_t = -1.5 si t mod 7 = 6, 0.25 si no
    t = np.arange(1, n + 1)
    return np.where(t % 7 == 6, -1.5, 0.25)


def simular_sin_memoria(media, replicas, rng):
    # Y_t = m_t + e_t con e_t iid N(0,1): no hay ninguna dependencia entre dias
    return media + rng.normal(0.0, 1.0, size=(replicas, len(media)))


def ols_ar1(y):
    # OLS de y_t contra (1, y_{t-1}); devuelve c, phi y los residuos
    y = np.asarray(y, dtype=float)
    x = np.column_stack((np.ones(len(y) - 1), y[:-1]))
    (c, phi), *_ = np.linalg.lstsq(x, y[1:], rcond=None)
    return c, phi, y[1:] - c - phi * y[:-1]
