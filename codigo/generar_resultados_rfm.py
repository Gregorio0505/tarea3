# Uso de IA: para ordenar el script siguiendo los incisos 6.1 a 6.5 y para imprimir
# en LaTeX las tablas de tasa de compra, AUC y reparto de la venta. La lectura de los resultados
# y la conexion con la parte 5 la hice yo.
#
# Script de la parte 6 (analisis RFM): calcula R, F y M al 1 de octubre de 2011, los puntajes,
# la tasa de compra por puntaje, el AUC de cada dimension y que parte de la venta de octubre a
# diciembre vino de los campeones y de los clientes nuevos.

import re
from pathlib import Path

import numpy as np
import pandas as pd

import analisis_rfm as rfm_mod
import graficas
import limpieza_datos as ld

FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def escribir(nombre, texto):
    print(f"\n--- tabla LaTeX: {nombre} ---\n{texto}--- fin de la tabla ---\n")


def tabla(formato, encabezado, filas):
    return ("\\begin{tabular}{|" + "|".join(re.findall(r"p\{[^}]*\}|\S", formato)) + "|}\n\\hline\n" + encabezado +
            " \\\\\n\\hline\n" + "\n".join(filas) + "\n\\hline\n\\end{tabular}\n")


def miles(x):
    return f"{x:,.0f}".replace(",", "\\,")


def main():
    limpio = ld.limpiar(ld.cargar_datos())
    datos = rfm_mod.con_cliente(limpio)

    # 6.1
    rfm, solo_cancel = rfm_mod.calcular_rfm(datos)
    print("6.1")
    print(f"  clientes con alguna venta antes del 1/10/2011: {len(rfm)}; "
          f"clientes con solo cancelaciones (sin R ni F): {solo_cancel}")
    no_positivo = rfm[rfm["M"] <= 0]
    print(f"  clientes con M <= 0: {len(no_positivo)} (M = 0: {(rfm['M'] == 0).sum()}, "
          f"M < 0: {(rfm['M'] < 0).sum()}); suma de su M = {no_positivo['M'].sum():.2f}")
    print(no_positivo.sort_values("M").head(8).to_string())
    print(rfm.describe().round(2).to_string())

    # 6.2 puntajes; primero lo que pasa con F sin desempatar
    print("6.2")
    print(f"  fraccion de clientes con F = 1: {(rfm['F'] == 1).mean():.4f}; F = 2: "
          f"{(rfm['F'] == 2).mean():.4f}; cuantiles de F al 20/40/60/80%: "
          f"{rfm['F'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()}")
    try:
        pd.qcut(rfm["F"], 5)
    except ValueError as error:
        print(f"  pd.qcut(F, 5) sin desempatar falla: {error}")
    p = rfm_mod.puntajes(rfm)
    f1 = rfm["F"] == 1
    print(f"  clientes con F = 1 repartidos en puntajes: {p.loc[f1, 'F'].value_counts().sort_index().to_dict()}")
    print(f"  clientes por puntaje: {p['F'].value_counts().sort_index().to_dict()}")

    # 6.3 tasa de compra por puntaje y AUC
    compro = rfm.index.isin(rfm_mod.compradores(datos))
    print(f"6.3 fraccion que compro entre el 1/10 y el 8/12: {compro.mean():.4f} ({compro.sum()})")
    tasas = {}
    filas = []
    for dim, nombre in (("R", "Recencia"), ("F", "Frecuencia"), ("M", "Monto")):
        t = pd.Series(compro, index=rfm.index).groupby(p[dim]).mean()
        tasas[nombre] = t.values
        filas.append(f"{nombre} & " + " & ".join(f"{v:.3f}" for v in t.values) +
                     f" & {t.values[-1] - t.values[0]:.3f} \\\\")
    escribir("tabla_tasa_compra (6.3)", tabla(
        "lcccccc", "Dimensión & 1 & 2 & 3 & 4 & 5 & Diferencia 5 $-$ 1", filas))
    graficas.tasa_compra_por_puntaje(tasas, FIGURAS / "rfm_tasa_compra_por_puntaje.png")
    suma = p["R"] + p["F"] + p["M"]
    candidatos = [("Recencia ($-R$, crudo)", -rfm["R"]), ("Frecuencia ($F$, crudo)", rfm["F"]),
                  ("Monto ($M$, crudo)", rfm["M"]), ("Suma de los tres puntajes", suma)]
    filas = [f"{n} & {rfm_mod.auc(v.values, compro):.4f} \\\\" for n, v in candidatos]
    escribir("tabla_auc (6.3)", tabla("lc", "Clasificador & AUC", filas))

    # 6.4 campeones y 6.5 clientes nuevos
    venta = rfm_mod.venta_neta_por_cliente(datos)
    total = venta.sum()
    campeones = p.index[(p >= 4).all(axis=1)]
    nuevos = venta.index.difference(rfm.index)
    # los nuevos son quienes no tienen ninguna factura de venta antes del corte
    previo = datos[datos["InvoiceDate"] < rfm_mod.CORTE]
    con_venta_previa = set(previo.loc[~ld.es_cancelacion(previo), "CustomerID"])
    assert not set(nuevos) & con_venta_previa
    viejos_sin_campeones = venta.index.intersection(rfm.index).difference(campeones)
    filas = [
        f"Campeones (4 o 5 en R, F y M) & {len(campeones)} & {len(campeones) / len(rfm):.3f} & "
        f"{miles(venta.reindex(campeones).fillna(0).sum())} & {venta.reindex(campeones).fillna(0).sum() / total:.3f} \\\\",
        f"Resto de clientes con historia & {len(rfm) - len(campeones)} & "
        f"{(len(rfm) - len(campeones)) / len(rfm):.3f} & {miles(venta.loc[viejos_sin_campeones].sum())} & "
        f"{venta.loc[viejos_sin_campeones].sum() / total:.3f} \\\\",
        f"Clientes nuevos & {len(nuevos)} & -- & {miles(venta.loc[nuevos].sum())} & "
        f"{venta.loc[nuevos].sum() / total:.3f} \\\\",
        "\\hline",
        f"Total con \\texttt{{CustomerID}} & & & {miles(total)} & 1.000 \\\\",
    ]
    escribir("tabla_reparto_venta (6.4 y 6.5)", tabla(
        "lcccc", "Grupo & Clientes & Fracción de los clientes de 6.1 & Venta neta 1/10--8/12 (£) & "
        "Fracción de la venta", filas))
    print(f"  campeones que compraron: {np.mean(pd.Index(campeones).isin(rfm_mod.compradores(datos))):.3f}")
    print(f"  clientes nuevos que compraron: {len(set(nuevos) & rfm_mod.compradores(datos))}")
    # la serie de la parte 5 tambien incluye las ventas sin CustomerID
    periodo = limpio[(limpio["InvoiceDate"] >= rfm_mod.CORTE) & (limpio["InvoiceDate"] < rfm_mod.FIN)]
    sin_id = periodo.loc[periodo["CustomerID"].isna(), "Monto"].sum()
    print(f"6.5 venta neta total del periodo (con y sin CustomerID): {periodo['Monto'].sum():.0f}; "
          f"sin CustomerID: {sin_id:.0f} ({sin_id / periodo['Monto'].sum():.3f}); nuevos sobre el "
          f"total: {venta.loc[nuevos].sum() / periodo['Monto'].sum():.3f}")


if __name__ == "__main__":
    main()
