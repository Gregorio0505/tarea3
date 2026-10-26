# Uso de IA: ayuda con el manejo de fechas en pandas (date_range, reindex y rolling
# centrado) y para armar la lista de bank holidays de 2010-2011, que despues revise contra
# gov.uk. Las fechas de corte de entrenamiento y prueba son las del enunciado.
#
# Serie de venta neta diaria sobre el calendario completo (incisos 2.1 a 2.4 y 3.1 a 3.2):
# dias cerrados como faltantes, clasificacion de esos dias, media movil centrada de 7 dias e
# indice estacional por dia de la semana.

import numpy as np
import pandas as pd

PRIMER_DIA = pd.Timestamp("2010-12-01")
ULTIMO_DIA = pd.Timestamp("2011-12-09")          # incompleto: se descarta (inciso 2.4)
FIN_ENTRENAMIENTO = pd.Timestamp("2011-09-30")
ORIGEN_PRUEBA = pd.Timestamp("2011-10-02")       # domingo: historia, no se evalua
FIN_PRUEBA = pd.Timestamp("2011-12-08")

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
DIAS_ABIERTOS = [0, 1, 2, 3, 4, 6]   # dayofweek de pandas: 0 = lunes; el sabado (5) no abre

# bank holidays de Inglaterra y Gales que caen dentro de los datos (gov.uk/bank-holidays)
BANK_HOLIDAYS = {
    "2010-12-27": "Christmas Day (sustituto)",
    "2010-12-28": "Boxing Day (sustituto)",
    "2011-01-03": "New Year's Day (sustituto)",
    "2011-04-22": "Good Friday",
    "2011-04-25": "Easter Monday",
    "2011-04-29": "Boda real (Guillermo y Catalina)",
    "2011-05-02": "Early May bank holiday",
    "2011-05-30": "Spring bank holiday",
    "2011-08-29": "Summer bank holiday",
}
MOVILES = {"2011-04-22", "2011-04-25", "2011-04-24"}   # dependen de la fecha de Pascua


def venta_diaria(limpio):
    # suma de Quantity x UnitPrice por dia; las cancelaciones ya traen cantidad negativa
    return limpio.groupby(limpio["InvoiceDate"].dt.normalize())["Monto"].sum()


def serie_calendario(limpio, hasta=ULTIMO_DIA):
    # un renglon por dia calendario; NaN donde no hubo ninguna transaccion
    calendario = pd.date_range(PRIMER_DIA, hasta, freq="D")
    return venta_diaria(limpio).reindex(calendario)


def clasificar_cerrados(serie):
    filas = []
    for dia in serie.index[serie.isna()]:
        clave = dia.strftime("%Y-%m-%d")
        if dia.dayofweek == 5:
            motivo = "Sábado"
        elif clave in BANK_HOLIDAYS:
            motivo = "Bank holiday: " + BANK_HOLIDAYS[clave]
        elif clave == "2011-04-24":
            motivo = "Domingo de Pascua"
        else:
            motivo = "Cierre de Navidad y Año Nuevo"
        filas.append({"fecha": dia, "dia": DIAS[dia.dayofweek], "motivo": motivo,
                      "movil": clave in MOVILES})
    return pd.DataFrame(filas)


def media_movil_centrada(serie, ventana=7):
    # aqui los dias cerrados cuentan como cero: la media movil resume la venta de la semana
    return serie.fillna(0).rolling(ventana, center=True).mean()


def indice_estacional(serie, mm):
    # promedio de y_t / MA_t por dia de la semana, solo dias abiertos del entrenamiento
    datos = pd.DataFrame({"y": serie, "mm": mm})
    datos = datos.loc[:FIN_ENTRENAMIENTO].dropna()
    cociente = datos["y"] / datos["mm"]
    indice = cociente.groupby(cociente.index.dayofweek).mean()
    return indice.reindex(DIAS_ABIERTOS)


def dias_abiertos(serie, desde=None, hasta=None):
    # una fila por dia abierto, sin calendario
    return serie.loc[desde:hasta].dropna()


def nombres_dias(indices):
    return [DIAS[i] for i in indices]


def entrenamiento(serie):
    return dias_abiertos(serie, hasta=FIN_ENTRENAMIENTO)


def prueba(serie):
    return dias_abiertos(serie, desde=ORIGEN_PRUEBA + pd.Timedelta(days=1), hasta=FIN_PRUEBA)


def es_abierto(serie, fecha):
    return fecha in serie.index and not np.isnan(serie.loc[fecha])
