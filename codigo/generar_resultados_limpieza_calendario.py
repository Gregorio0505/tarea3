# Uso de IA: para ordenar el script siguiendo los incisos 1.1 a 2.4 y para
# imprimir en LaTeX la bitacora, la tabla de codigos, las seis lineas mas grandes y los dias
# cerrados. Que revisar en cada caso y lo que se decidio con cada uno salio de revisar los
# datos, y la interpretacion la hice yo.
#
# Script de las partes 1 y 2 (limpieza y calendario): describe los datos crudos, arma la
# bitacora de limpieza, investiga las seis lineas mas grandes y clasifica los dias cerrados.

import re
import pandas as pd

import limpieza_datos as ld
import serie_diaria as sd


def escribir(nombre, texto):
    print(f"\n--- tabla LaTeX: {nombre} ---\n{texto}--- fin de la tabla ---\n")


def tabla(formato, encabezado, filas):
    return ("\\begin{tabular}{|" + "|".join(re.findall(r"p\{[^}]*\}|\S", formato)) + "|}\n\\hline\n" + encabezado +
            " \\\\\n\\hline\n" + "\n".join(filas) + "\n\\hline\n\\end{tabular}\n")


def libras(x):
    return f"{x:,.2f}".replace(",", "\\,")


def entero(x):
    return f"{x:,d}".replace(",", "\\,")


def tex(texto):
    return str(texto).replace("&", "\\&").replace("_", "\\_").replace("#", "\\#")


def main():
    crudo = ld.cargar_datos()

    # 1.1 descripcion de los datos crudos
    print("1.1")
    print(f"  filas: {len(crudo)}, facturas: {crudo['InvoiceNo'].nunique()}, "
          f"productos: {crudo['StockCode'].nunique()}, clientes: {crudo['CustomerID'].nunique()}")
    print(f"  fraccion sin CustomerID: {crudo['CustomerID'].isna().mean():.4f} "
          f"({crudo['CustomerID'].isna().sum()} filas)")
    print(f"  monto bruto total: {crudo['Monto'].sum():.2f}")

    # 1.2 (b) prefijos de factura
    prefijo = crudo["InvoiceNo"].str.extract(r"^(\D*)")[0]
    print("\n1.2 (b) prefijos de InvoiceNo:", prefijo.value_counts().to_dict())
    cancel = crudo[prefijo == "C"]
    print(f"  cancelaciones: {len(cancel)} filas, {cancel['InvoiceNo'].nunique()} facturas, "
          f"monto {cancel['Monto'].sum():.2f}; todas con cantidad negativa: "
          f"{(cancel['Quantity'] < 0).all()}")
    print(crudo[prefijo == "A"][["InvoiceNo", "StockCode", "Description", "Quantity",
                                 "InvoiceDate", "UnitPrice"]].to_string())

    # 1.2 (c) codigos que no empiezan con cinco digitos
    raros = crudo[~crudo["StockCode"].str.match(r"^\d{5}")]
    resumen = raros.groupby("StockCode").agg(
        filas=("Monto", "size"), monto=("Monto", "sum"),
        descripcion=("Description", lambda s: s.dropna().iloc[0] if s.notna().any() else "(vacía)"))
    resumen = resumen.sort_values("filas", ascending=False)
    filas = []
    for codigo, f in resumen.iterrows():
        if codigo.startswith(("DCGS", "gift_")) or codigo == "PADS":
            continue
        decision = "Se va" if codigo in ld.CODIGOS_NO_VENTA else "Se queda"
        filas.append(f"\\texttt{{{tex(codigo)}}} & {tex(f['descripcion'])} & {entero(f['filas'])}"
                     f" & {libras(f['monto'])} & {decision} \\\\")
    for prefijo_prod, nombre in [("DCGS", "DCGS*"), ("gift_", "gift\\_0001\\_*"), ("PADS", "PADS")]:
        g = raros[raros["StockCode"].str.startswith(prefijo_prod)]
        filas.append(f"\\texttt{{{nombre}}} & productos con código especial & "
                     f"{entero(len(g))} & {libras(g['Monto'].sum())} & Se queda \\\\")
    escribir("tabla_codigos (1.2c)", tabla(
        "llrrl", "Código & Descripción & Filas & Monto (£) & Decisión", filas))

    # 1.2 (d) y (e)
    print(f"1.2 (d) precio = 0: {(crudo['UnitPrice'] == 0).sum()}, "
          f"precio < 0: {(crudo['UnitPrice'] < 0).sum()}")
    neg = crudo[(crudo["Quantity"] < 0) & (prefijo != "C")]
    print(f"1.2 (e) cantidad negativa sin C: {len(neg)} filas, precio maximo {neg['UnitPrice'].max()}, "
          f"con CustomerID: {neg['CustomerID'].notna().sum()}")
    print("  descripciones mas comunes:",
          neg["Description"].fillna("(vacía)").value_counts().head(8).to_dict())

    # 1.3 bitacora
    b = ld.bitacora(crudo)
    limpio = ld.limpiar(crudo)
    filas = [f"{i + 1} & {f['regla']} & {entero(f['filas'])} & {libras(f['monto'])} & "
             f"{libras(f['monto_abs'])} \\\\" for i, f in b.iterrows()]
    sin_id = limpio["CustomerID"].isna()
    filas.append("\\hline")
    filas.append(f"-- & Sin \\texttt{{CustomerID}} (se quedan en la serie) & {entero(int(sin_id.sum()))}"
                 f" & {libras(limpio.loc[sin_id, 'Monto'].sum())} & "
                 f"{libras(limpio.loc[sin_id, 'Monto'].abs().sum())} \\\\")
    filas.append("\\hline")
    filas.append(f"& Datos crudos & {entero(len(crudo))} & {libras(crudo['Monto'].sum())} & \\\\")
    filas.append(f"& Datos limpios & {entero(len(limpio))} & {libras(limpio['Monto'].sum())} & \\\\")
    escribir("tabla_bitacora (1.3)", tabla(
        "clrrr", "Orden & Regla & Filas que quita & Monto neto (£) & Monto absoluto (£)", filas))
    print(f"  limpias sin CustomerID: {sin_id.sum()} ({sin_id.mean():.4f})")

    # 1.4 las seis lineas mas grandes antes de limpiar
    top = crudo.loc[crudo["Monto"].abs().sort_values(ascending=False).index[:6]]
    filas = []
    for _, f in top.iterrows():
        filas.append(f"{f['InvoiceNo']} & \\texttt{{{f['StockCode']}}} & {tex(f['Description'])} & "
                     f"{entero(int(f['Quantity']))} & {f['UnitPrice']:.2f} & "
                     f"{f['InvoiceDate']:%d/%m/%Y %H:%M} & {f['CustomerID']} & "
                     f"{libras(f['Monto'])} \\\\")
    escribir("tabla_lineas_grandes (1.4)", tabla(
        "lllrrllr", "Factura & Código & Descripción & Cantidad & Precio & Fecha & Cliente & "
        "Monto (£)", filas))
    for cliente in top["CustomerID"].unique():
        c = crudo[crudo["CustomerID"] == cliente]
        print(f"  cliente {cliente}: {c['InvoiceNo'].nunique()} facturas, monto neto "
              f"{c['Monto'].sum():.2f}, de {c['InvoiceDate'].min():%d/%m/%Y} a "
              f"{c['InvoiceDate'].max():%d/%m/%Y}")
        for dia in top.loc[top["CustomerID"] == cliente, "InvoiceDate"].dt.normalize().unique():
            del_dia = crudo[crudo["InvoiceDate"].dt.normalize() == dia]
            mismo = del_dia[del_dia["CustomerID"] == cliente]
            print(f"    {dia:%d/%m/%Y}: lineas del cliente ese dia = {len(mismo)}, neto del "
                  f"cliente = {mismo['Monto'].sum():.2f}; venta neta del dia (crudo) = "
                  f"{del_dia['Monto'].sum():.2f}")
    print("  cuantas de las seis siguen en los datos limpios:", top.index.isin(limpio.index).sum())

    # 2.1 calendario
    serie_completa = sd.serie_calendario(limpio)
    print("\n2.1")
    print(f"  dias calendario entre {sd.PRIMER_DIA:%d/%m/%Y} y {sd.ULTIMO_DIA:%d/%m/%Y}: "
          f"{len(serie_completa)}; con transacciones: {serie_completa.notna().sum()}")

    # 2.2 dias cerrados
    cerrados = sd.clasificar_cerrados(serie_completa)
    print("  cerrados por motivo:", cerrados["motivo"].value_counts().to_dict())
    print("  sabados con alguna transaccion:",
          (serie_completa[serie_completa.index.dayofweek == 5].notna()).sum())
    otros = cerrados[cerrados["motivo"] != "Sábado"]
    filas = [f"{f['fecha']:%d/%m/%Y} & {f['dia']} & {tex(f['motivo'])} & "
             f"{'Sí' if f['movil'] else 'No'} \\\\" for _, f in otros.iterrows()]
    escribir("tabla_dias_cerrados (2.2)", tabla(
        "llll", "Fecha & Día & Motivo & Móvil", filas))

    # 2.4 el ultimo dia
    ultimo = crudo[crudo["InvoiceDate"].dt.normalize() == sd.ULTIMO_DIA]
    viernes = serie_completa[(serie_completa.index.dayofweek == 4)].dropna()
    print("2.4")
    print(f"  ultimo dia: primera {ultimo['InvoiceDate'].min():%H:%M}, ultima "
          f"{ultimo['InvoiceDate'].max():%H:%M}, facturas {ultimo['InvoiceNo'].nunique()}, "
          f"venta neta limpia {serie_completa.iloc[-1]:.2f}")
    ultimas = crudo.groupby(crudo["InvoiceDate"].dt.normalize())["InvoiceDate"].max()
    minutos = (ultimas - ultimas.index).dt.total_seconds() / 60   # minutos desde la medianoche
    mediana = minutos.iloc[:-1].median()
    print(f"  hora de la ultima factura en un dia tipico (mediana): "
          f"{int(mediana // 60)}:{int(mediana % 60):02d}; "
          f"venta media de los viernes: {viernes.iloc[:-1].mean():.2f}; "
          f"ultimos cuatro viernes: {viernes.iloc[-5:-1].round(2).tolist()}")


if __name__ == "__main__":
    main()
