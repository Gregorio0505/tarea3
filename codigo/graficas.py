# Uso de IA: para que todas las figuras tengan el mismo estilo de la Tarea 2 y se
# guarden como PNG sin abrir ventanas, y para el manejo de fechas en los ejes (mdates) y las
# bandas de la ACF. Que se grafica en cada figura lo decidi yo segun lo que pide cada inciso.
#
# Graficas de la tarea: realizaciones simuladas, ACF con bandas, serie diaria con su media
# movil, indice estacional, pronosticos y la tasa de compra por puntaje RFM.
# Sirve para no repetir el codigo de dibujo en cada inciso.

import matplotlib
matplotlib.use("Agg")   # backend sin ventana: el script solo escribe archivos

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

AZUL = "#2a78d6"
NARANJA = "#eb6834"
AQUA = "#1baf7a"
AMARILLO = "#eda100"
MAGENTA = "#e87ba4"
VERDE = "#008300"
GRIS = "#52514e"
COLORES = [AZUL, NARANJA, AQUA, AMARILLO, MAGENTA, VERDE]

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def eje_fechas(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def realizaciones(paneles, ruta):
    # paneles = lista de (serie, titulo, media o None); una fila por proceso
    fig, ejes = plt.subplots(len(paneles), 1, figsize=(7.5, 1.8 * len(paneles)), sharex=True)
    for ax, (y, titulo, media) in zip(ejes, paneles):
        ax.plot(np.arange(1, len(y) + 1), y, color=AZUL, linewidth=1)
        if media is not None:
            ax.plot(np.arange(1, len(y) + 1), media, color=NARANJA, linewidth=2, label="$m_t$")
            ax.legend(loc="upper left", fontsize=8)
        ax.set_title(titulo, fontsize=10, loc="left")
        ax.set_ylabel("$y_t$")
    ejes[-1].set_xlabel("$t$")
    guardar(fig, ruta)


def dibujar_acf(ax, r, n, titulo, teoria=None):
    k = np.arange(1, len(r) + 1)
    ax.vlines(k, 0, r, color=AZUL, linewidth=2)
    ax.plot(k, r, "o", color=AZUL, markersize=4)
    b = 1.959964 / np.sqrt(n)
    ax.axhline(b, color=GRIS, linestyle="--", linewidth=1)
    ax.axhline(-b, color=GRIS, linestyle="--", linewidth=1)
    ax.axhline(0, color=GRIS, linewidth=0.8)
    if teoria is not None:
        ax.plot(k, teoria, "_", color=NARANJA, markersize=10, markeredgewidth=2, label="Teoría")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xticks(k[::2] if len(k) > 12 else k)
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.set_xlabel("Rezago $k$")
    ax.set_ylabel("$r_k$")


def acf_paneles(paneles, ruta, columnas=1, alto=2.6):
    # paneles = lista de (r, n, titulo, teoria o None)
    filas = int(np.ceil(len(paneles) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(4.2 * columnas if columnas > 1 else 7.5,
                                                       alto * filas), squeeze=False)
    for ax, panel in zip(ejes.flat, paneles):
        dibujar_acf(ax, *panel)
    for ax in list(ejes.flat)[len(paneles):]:
        ax.set_visible(False)
    guardar(fig, ruta)


def serie_con_media_movil(serie, mm, ruta, fin_entrenamiento):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(serie.index, serie.values / 1000, ".", color=AZUL, markersize=3.5,
            label="Venta neta diaria (días abiertos)")
    ax.plot(mm.index, mm.values / 1000, color=NARANJA, linewidth=2,
            label="Media móvil centrada de 7 días")
    ax.axvline(fin_entrenamiento, color=GRIS, linestyle="--", linewidth=1)
    ax.text(fin_entrenamiento, 2, "  fin del entrenamiento", fontsize=8,
            color=GRIS, va="bottom")
    eje_fechas(ax)
    ax.set_ylabel("Miles de £")
    ax.legend(loc="upper left", fontsize=8)
    guardar(fig, ruta)


def barras_indice(nombres, valores, ruta):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    barras = ax.bar(nombres, valores, color=AZUL, width=0.6)
    ax.axhline(1, color=GRIS, linestyle="--", linewidth=1)
    ax.axhline(np.mean(valores), color=NARANJA, linestyle=":", linewidth=1.5,
               label=f"Promedio de los seis = {np.mean(valores):.3f}")
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Índice estacional")
    ax.legend(loc="lower right", fontsize=8)
    guardar(fig, ruta)


def pronostico_largo(historia, paneles, ruta):
    # historia = (x, y) de los ultimos dias observados; paneles = lista de (x, y, titulo)
    fig, ejes = plt.subplots(len(paneles), 1, figsize=(7.5, 2.6 * len(paneles)), sharex=True)
    for ax, (x, y, titulo) in zip(ejes, paneles):
        ax.plot(historia[0], historia[1] / 1000, color=GRIS, linewidth=1, label="Observado")
        ax.plot(x, y / 1000, color=NARANJA, linewidth=1.5, label="Pronóstico")
        ax.set_title(titulo, fontsize=10, loc="left")
        ax.set_ylabel("Miles de £")
        ax.legend(loc="upper left", fontsize=8)
    ejes[-1].set_xlabel("Fila (día abierto) del entrenamiento")
    guardar(fig, ruta)


def pronosticos_prueba(real, pronosticos, ruta):
    # real = Serie de la prueba; pronosticos = dict nombre -> Serie con el mismo indice
    fig, ejes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    grupos = [["B1", "B2", "M1"], ["M2", "S1", "S2"]]
    for ax, grupo in zip(ejes, grupos):
        ax.plot(real.index, real.values / 1000, "o-", color=GRIS, markersize=3, linewidth=1,
                label="Venta real")
        for nombre in grupo:
            color = COLORES[["B1", "B2", "M1", "M2", "S1", "S2"].index(nombre)]
            p = pronosticos[nombre]
            ax.plot(p.index, p.values / 1000, ".-", color=color, markersize=3, linewidth=1.2,
                    label=nombre)
        ax.set_ylabel("Miles de £")
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", labelsize=7)
    guardar(fig, ruta)


def tasa_compra_por_puntaje(tasas, ruta):
    # tasas = dict dimension -> arreglo de 5 fracciones (puntajes 1 a 5)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ancho = 0.26
    x = np.arange(1, 6)
    for i, (nombre, valores) in enumerate(tasas.items()):
        ax.bar(x + (i - 1) * ancho, valores, width=ancho - 0.02, color=COLORES[i], label=nombre)
    ax.set_xticks(x)
    ax.set_xlabel("Puntaje (5 = mejor)")
    ax.set_ylabel("Fracción que compró\n1 oct – 8 dic")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    guardar(fig, ruta)
