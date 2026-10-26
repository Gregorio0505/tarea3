# Uso de IA: para escribir las reglas como una lista de (nombre, mascara), de modo que
# la misma definicion sirva para limpiar y para armar la bitacora, y con las expresiones
# regulares de InvoiceNo y StockCode. Que lineas cuentan como venta, el orden de las reglas y
# la lista de codigos que no son productos lo decidi yo despues de revisar las descripciones.
#
# Carga y limpieza de Online Retail (incisos 1.1 a 1.5).
# limpiar(df) devuelve las lineas que cuentan como venta; bitacora(df) dice cuantas filas y
# cuanto dinero quita cada regla, en el mismo orden en que limpiar las aplica.
# Sirve para la serie diaria (partes 2 a 5) y para el analisis RFM de la parte 6.

from pathlib import Path

import pandas as pd

RUTA_DATOS = Path(__file__).resolve().parent.parent / "datos" / "Online Retail.xlsx"

# codigos que no son mercancia: envios, comisiones, muestras y la deuda incobrable.
# M (ajuste manual) y D (descuento) se quedan porque corrigen el precio de ventas reales.
CODIGOS_NO_VENTA = ["POST", "DOT", "C2", "AMAZONFEE", "BANK CHARGES", "CRUK", "S", "B"]


def cargar_datos(ruta=RUTA_DATOS):
    # InvoiceNo, StockCode y CustomerID son etiquetas, no numeros: se leen como texto para
    # no perder la C de las cancelaciones, las letras de 85123A ni convertir 17850 en 17850.0
    df = pd.read_excel(ruta, dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str})
    df["Monto"] = df["Quantity"] * df["UnitPrice"]
    return df


def es_cancelacion(df):
    return df["InvoiceNo"].str.startswith("C")


# cada regla devuelve True en las filas que se van; se aplican en este orden
REGLAS = [
    ("Duplicados exactos", lambda d: d.drop(columns="Monto").duplicated()),
    ("Facturas con prefijo A (ajuste de deuda)", lambda d: d["InvoiceNo"].str.startswith("A")),
    ("Cantidad negativa sin ser cancelación",
     lambda d: (d["Quantity"] < 0) & ~es_cancelacion(d)),
    ("Códigos que no son mercancía", lambda d: d["StockCode"].isin(CODIGOS_NO_VENTA)),
    ("Precio igual a cero o negativo", lambda d: d["UnitPrice"] <= 0),
]


def limpiar(df):
    limpio = df
    for _, regla in REGLAS:
        limpio = limpio[~regla(limpio)]
    return limpio   # conserva el indice original para poder rastrear cada fila


def bitacora(df):
    # mismo recorrido que limpiar, pero anotando lo que quita cada paso
    filas = []
    actual = df
    for nombre, regla in REGLAS:
        fuera = regla(actual)
        filas.append({"regla": nombre, "filas": int(fuera.sum()),
                      "monto": actual.loc[fuera, "Monto"].sum(),
                      "monto_abs": actual.loc[fuera, "Monto"].abs().sum()})
        actual = actual[~fuera]
    return pd.DataFrame(filas)
