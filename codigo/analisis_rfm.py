# Uso de IA: para hacer las agregaciones por cliente en pandas (R, F y M al 1 de
# octubre) y para escribir el AUC a mano con rangos (forma de Mann-Whitney), sin usar
# scikit-learn. El desempate de los puntajes con default_rng(2026) y qcut sobre
# rank(method='first') es el que pide el enunciado.
#
# Analisis RFM de la parte 6: recencia, frecuencia y monto de cada cliente al 1 de octubre de
# 2011, puntajes por quintiles, quien compro despues y que parte de la venta vino de cada grupo.

import numpy as np
import pandas as pd
from scipy import stats

import limpieza_datos as ld

CORTE = pd.Timestamp("2011-10-01")
FIN = pd.Timestamp("2011-12-09")      # se excluye: el periodo de compra va hasta el 8 de diciembre
SEMILLA_DESEMPATE = 2026             # la fija el enunciado


def con_cliente(limpio):
    # para la parte 6 si hace falta saber quien compro
    return limpio[limpio["CustomerID"].notna() & (limpio["InvoiceDate"] < FIN)]


def calcular_rfm(datos, corte=CORTE):
    # solo lo anterior al corte; R y F salen de las facturas de venta, M de la venta neta
    previo = datos[datos["InvoiceDate"] < corte]
    ventas = previo[~ld.es_cancelacion(previo)]
    ultima = ventas.groupby("CustomerID")["InvoiceDate"].max().dt.normalize()
    rfm = pd.DataFrame({
        "R": (corte - ultima).dt.days,
        "F": ventas.groupby("CustomerID")["InvoiceNo"].nunique(),
        "M": previo.groupby("CustomerID")["Monto"].sum().round(2),   # en centavos: quita el ruido
                                                                     # de punto flotante (-5e-15)
    })
    solo_cancelaciones = rfm["R"].isna().sum()   # clientes sin ninguna venta antes del corte
    return rfm.dropna(subset=["R"]).sort_index(), int(solo_cancelaciones)


def puntaje(valores):
    # quintiles sobre el rango con desempate por orden de aparicion (que ya viene barajado)
    return pd.qcut(valores.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


def puntajes(rfm, semilla=SEMILLA_DESEMPATE):
    # clientes por CustomerID ascendente, barajados con la semilla, y despues los quintiles;
    # en recencia 5 es el mas reciente, por eso se puntua -R
    barajado = rfm.sort_index().iloc[np.random.default_rng(semilla).permutation(len(rfm))]
    p = pd.DataFrame({"R": puntaje(-barajado["R"]), "F": puntaje(barajado["F"]),
                      "M": puntaje(barajado["M"])})
    return p.loc[rfm.index]


def compradores(datos, desde=CORTE, hasta=FIN):
    # clientes con al menos una factura de venta entre el corte y el 8 de diciembre
    tramo = datos[(datos["InvoiceDate"] >= desde) & (datos["InvoiceDate"] < hasta)]
    return set(tramo.loc[~ld.es_cancelacion(tramo), "CustomerID"])


def venta_neta_por_cliente(datos, desde=CORTE, hasta=FIN):
    tramo = datos[(datos["InvoiceDate"] >= desde) & (datos["InvoiceDate"] < hasta)]
    return tramo.groupby("CustomerID")["Monto"].sum()


def auc(puntuacion, y):
    # AUC = P(puntuacion de un comprador > la de un no comprador), con empates a la mitad:
    # (suma de rangos de los positivos - n1(n1+1)/2) / (n1 n0)
    rangos = stats.rankdata(puntuacion)
    y = np.asarray(y, dtype=bool)
    n1, n0 = y.sum(), (~y).sum()
    return (rangos[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
