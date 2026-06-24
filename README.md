# TFG — Clasificación de diabetes y detección de infradiagnóstico con datos NHANES

Código del Trabajo de Fin de Grado *«Revisión de conceptos y técnicas de aprendizaje automático y aplicación a un problema práctico»*, del Grado en Ingeniería Matemática de la Universidad Complutense de Madrid, tutorizado por Juan Tinguaro Rodríguez González.

## Descripción

El proyecto aplica métodos de aprendizaje automático supervisado sobre los datos de la encuesta *National Health and Nutrition Examination Survey* (NHANES), combinando seis ciclos bienales (E–P), para resolver dos problemas de clasificación binaria:

- **Flujo DIABETES**: detección de diabetes en población general a partir de una variable respuesta compuesta, que integra criterios clínicos y autorreporte.
- **Flujo INFRADIAG**: identificación del subgrupo de individuos con diabetes no diagnosticada (infradiagnóstico), en el que la clase positiva es la minoritaria.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `sincorrecion.py` | Pipeline sin corrección del desbalanceo de clases; selección del umbral de decisión *out-of-fold*. |
| `submuestreo.py` | Pipeline con submuestreo (*undersampling*) de la clase mayoritaria; selección del umbral sobre un conjunto de *holdout*. |

Ambos scripts comparten el mismo preprocesamiento, las mismas familias de modelos y la misma partición *train/test*; difieren únicamente en la estrategia de tratamiento del desbalanceo y en el procedimiento de selección del umbral.

## Metodología (resumen)

- **Modelos**: regresión logística (con variantes de regularización), *Random Forest* y *Gradient Boosting*.
- **Ajuste de hiperparámetros**: `GridSearchCV` con validación cruzada para todas las familias, empleando ROC-AUC como métrica de evaluación, comparable entre estrategias.
- **Umbral de decisión**: optimización de la métrica F<sub>β</sub> con β = 2, que prioriza la sensibilidad (*recall*) por motivos clínicos, al ser el coste de un falso negativo superior al de un falso positivo.
- **Preprocesamiento**: unión de ciclos mediante `SEQN`, filtrado de variables muy correlacionadas (Spearman, umbral 0,80), *winsorización* en los percentiles 1 y 99 y matrices de características diferenciadas para modelos lineales y basados en árboles.

## Datos

Los datos proceden de NHANES y son de acceso público a través de los CDC. Por su tamaño, **los ficheros `.XPT` no se incluyen** en este repositorio: deben descargarse de la web oficial y situarse en el directorio de entrada que esperan los scripts.

Fuente: <https://wwwn.cdc.gov/nchs/nhanes/>

## Requisitos

- Python 3.x
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`

La lectura de los ficheros `.XPT` se realiza con `pandas.read_sas` (alternativamente, `pyreadstat`).

## Ejecución

Los scripts se desarrollaron y ejecutaron en el entorno de desarrollo Spyder. Pueden ejecutarse ajustando las rutas de entrada a la ubicación de los ficheros NHANES:

```bash
python sincorreciones.py
python submuestreo.py
```

## Autoría

Vanessa Carnero — Grado en Ingeniería Matemática, Universidad Complutense de Madrid (2026).
