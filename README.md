# Tarea 3 — Introducción a series de tiempo

Curso Libre de Configuración 2: Análisis de Datos — Universidad del Istmo de Guatemala.

Jose Gregorio Coronel Colombo — jcoronel@unis.edu.gt

## Uso de inteligencia artificial

En este trabajo se usó inteligencia artificial como apoyo para escribir varios de los códigos
en Python a partir de las fórmulas y reglas del enunciado, para hacer algunas de las gráficas y
dar formato a las tablas, para resolver dudas de pandas y statsmodels, y para investigar temas
que no quedaban claros con los libros. También se usó para ordenar la redacción del documento a
partir de los resultados. La semilla, qué líneas cuentan como venta, el orden de las reglas de
limpieza y las derivaciones del inciso 0.1 se hicieron sin IA, y los números del texto se
revisaron contra la salida de los scripts. Al inicio de cada archivo `.py` se indica para qué se
usó la IA en ese archivo.

## Cómo correrlo

```bash
pip install -r requirements.txt
cd codigo
python3 generar_resultados_procesos_simulados.py    # parte 0, ~15 segundos
python3 generar_resultados_limpieza_calendario.py   # partes 1 y 2, ~30 segundos
python3 generar_resultados_series_pronostico.py     # partes 3, 4 y 5, ~1 minuto
python3 generar_resultados_rfm.py                   # parte 6, ~30 segundos
```

Los tres últimos parten del archivo original `datos/Online Retail.xlsx` (UCI, conjunto 352),
que va incluido en el repositorio; leerlo toma unos 25 segundos cada vez. Cada script guarda
sus figuras en `figuras/` e imprime en pantalla, en LaTeX, las tablas que lleva `main.tex`.
Las semillas están fijas al inicio de cada script, así que el resultado es siempre el mismo.

## Semillas

La semilla base es 552003, la misma de la Tarea 2.

- Parte 0: cada escenario usa un múltiplo distinto (`552003·1` a `552003·4` para
  φ = 0, 0.5, 0.9, 1; `552003·5` para la tendencia; `552003·6` para el patrón semanal).
- Inciso 5.2 (a): las 10 repeticiones de la validación cruzada usan `552003·1` a `552003·10`.
- Inciso 6.2: el desempate usa 2026, que es la semilla que fija el enunciado.

## Archivos

| Archivo | Qué contiene |
|---|---|
| `codigo/limpieza_datos.py` | `cargar_datos`, la función `limpiar(df)` y la bitácora de limpieza |
| `codigo/autocorrelacion.py` | r_k a mano, bandas de ruido blanco y Ljung–Box |
| `codigo/procesos_simulados.py` | AR(1), procesos sin memoria y el OLS de φ |
| `codigo/serie_diaria.py` | Serie diaria sobre el calendario, días cerrados, media móvil e índice estacional |
| `codigo/modelos_pronostico.py` | B1, B2, M1, M2, M2 con media móvil, S1 y S2 |
| `codigo/validacion_pronostico.py` | Validación cruzada aleatoria y origen móvil semanal |
| `codigo/analisis_rfm.py` | Recencia, frecuencia, monto, puntajes y AUC |
| `codigo/graficas.py` | Todas las figuras |
| `codigo/generar_resultados_*.py` | Los cuatro scripts que corren cada parte |
| `main.tex` / `main.pdf` | El informe con todas las partes |
| `figuras/` | Las once figuras del informe, en PNG |
| `datos/Online Retail.xlsx` | Los datos originales |
| `BI_Tarea4.pdf` | El enunciado de la tarea |

## Compilar el documento

`main.tex` es autocontenido: las tablas están escritas dentro del archivo y lo único
externo son las imágenes de `figuras/`. Compila con pdfLaTeX sin bibliografía aparte.
