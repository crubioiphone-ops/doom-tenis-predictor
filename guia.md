# Guia de interpretacion — Tennis Predictor

---

## 1. Probabilidad de victoria

La probabilidad indica cuantas veces de cada 100 ganaría el jugador si este partido se disputase en las mismas condiciones repetidamente.

| Probabilidad | Interpretacion | Precision historica del modelo |
|---|---|---|
| > 70% | Alta confianza en el favorito | ~75% de acierto (ATP) / ~76% (WTA) |
| 60–70% | Confianza media | ~69–72% |
| 50–60% | Partido muy igualado | ~65–67% |

> El modelo nunca tiene certeza absoluta. Una probabilidad del 80% significa que el rival gana 1 de cada 5 veces.

---

## 2. Cuota justa

La cuota justa es la cuota que debería ofrecer una casa sin margen de beneficio. Se calcula como **1 / probabilidad**.

**Ejemplo:** si el modelo da 70% al jugador A, su cuota justa es 1 / 0.70 = **1.43**.

- Si la casa ofrece **mas de 1.43** → existe ventaja matematica potencial.
- Si la casa ofrece **menos de 1.43** → la casa ya se ha quedado con el valor.

Compara siempre la cuota justa con la cuota real antes de apostar.

---

## 3. Expected Value (EV)

El EV mide cuanto ganarías o perderías de media por cada euro apostado si repitieras esa apuesta infinitas veces.

**Formula:** `EV = P_modelo × (cuota - 1) - (1 - P_modelo)`

| EV | Significado |
|---|---|
| **EV > 0** | Apuesta con valor esperado positivo. Rentable a largo plazo. |
| **EV = 0** | Sin ventaja ni desventaja. |
| **EV < 0** | La casa tiene ventaja matematica. |

**Ejemplo:** P = 0.65, cuota = 2.10
- EV = 0.65 × (2.10 - 1) - 0.35 = 0.65 × 1.10 - 0.35 = **+0.365 → +36.5% por euro**

> EV positivo NO garantiza ganar el partido. Solo garantiza rentabilidad si repites la estrategia cientos de veces con disciplina.

---

## 4. ELO Global

Sistema de puntuacion que mide el nivel general de un jugador basandose en su historial completo. Se actualiza tras cada partido: ganar a un rival fuerte sube mas el ELO que ganar a uno debil.

| Rango ELO | Nivel aproximado |
|---|---|
| > 2100 | Elite mundial (top 3–5) |
| 1900–2100 | Top 10–30 |
| 1750–1900 | Top 30–100 |
| 1600–1750 | Top 100–200 |
| < 1600 | Circuito menor / jugador joven |

La diferencia de ELO es el predictor mas importante del modelo (41% de importancia en las decisiones del algoritmo).

---

## 5. ELO de Superficie

ELO calculado exclusivamente con partidos en esa superficie (arcilla, hierba, pista dura). Captura la especializacion del jugador en ese terreno.

Un jugador puede tener ELO global alto pero ELO de arcilla bajo. Ejemplo tipico: jugadores potentes en pista dura que pierden velocidad y eficacia en tierra batida.

**Cuando importa mas:** en torneos de arcilla (Roland Garros, Madrid, Roma) o hierba (Wimbledon), el ELO de superficie puede ser mas revelador que el ELO global.

---

## 6. Forma reciente (form10, form20)

Porcentaje de victorias en los ultimos 10 o 20 partidos jugados, sin filtrar por superficie.

| Valor | Interpretacion |
|---|---|
| 1.00 | Racha perfecta (10 de 10) |
| 0.70–0.90 | Excelente forma |
| 0.50–0.70 | Forma normal |
| < 0.40 | Mala racha |
| 0.50 (exacto) | Insuficientes datos — se usa como valor neutro |

> Si ves 0.50 exacto en un jugador joven o recien llegado, es probable que sea el valor por defecto, no el real.

---

## 7. Forma en superficie (surface_form)

Porcentaje de victorias en los ultimos 30 partidos jugados en esa superficie especifica. Mas relevante que form10 cuando el torneo es en una superficie especializada.

Ejemplo: un jugador con form10 = 0.90 pero surface_form = 0.40 en arcilla → lleva buena racha general pero mal rendimiento reciente en tierra.

---

## 8. H2H (Head-to-head)

Numero de veces que el jugador ha ganado al rival en el historial registrado desde el año 2000.

- Un H2H 3-0 a favor puede inclinar la prediccion, especialmente en la misma superficie.
- H2H 0-0 es frecuente entre jugadores que nunca se han enfrentado o que han jugado muy poco entre si.
- El peso del H2H en el modelo es moderado (~2.4% de importancia) porque muchos emparejamientos tienen pocos datos.

---

## 9. Estadisticas de servicio

Medias moviles calculadas sobre los ultimos 15 partidos con estadisticas disponibles:

| Estadistica | Descripcion | Referencia de buen nivel |
|---|---|---|
| 1er Servicio % | Porcentaje de primeros servicios que entran | > 60% |
| Ganados con 1er serv. | % de puntos ganados cuando entra el 1er servicio | > 70% |
| Ganados con 2do serv. | % de puntos ganados con el 2do servicio | > 50% |
| Break points salvados | % de break points salvados al resto | > 60% |

Si aparece un guion (—) en estas estadisticas, significa que el dataset no tiene suficientes datos de servicio para ese jugador (frecuente en torneos menores o datos pre-2010).

---

## 10. Niveles de confianza y precision historica

Esta tabla muestra la precision real del modelo medida sobre partidos de 2023–2026 que NO fueron usados en el entrenamiento:

| Confianza del modelo | Precision ATP | Precision WTA | Partidos evaluados |
|---|---|---|---|
| Todos los partidos (>50%) | 65.5% | 65.5% | ~8,600 (ATP) / ~8,000 (WTA) |
| Confianza >= 60% | 69.2% | 69.5% | ~6,600 / ~5,900 |
| Confianza >= 65% | 71.6% | 72.9% | ~5,500 / ~4,700 |
| Confianza >= 70% | 74.6% | 76.0% | ~4,300 / ~3,600 |

Interpretacion practica: si el modelo predice con 70% de probabilidad a un jugador, historicamente acierta el 74–76% de las veces. Solo te equivocas 1 de cada 4.

---

## 11. Cuando ser mas cauteloso

El modelo funciona peor en estos contextos. Tener en cuenta para ajustar el tamano de la apuesta:

- **Jugadores jovenes con pocos partidos** — el modelo usa valores neutros (0.50) por falta de datos.
- **Lesiones o baja reciente** — no reflejadas en los datos. Consultar noticias antes de apostar.
- **Condiciones inusuales** — viento extremo, lluvia interrumpida, calor extremo.
- **Primeras rondas de torneos menores** — menos historial de los rivales en esa superficie.
- **Largo periodo sin jugar** — la forma reciente puede estar desactualizada.
- **Cambios de entrenador o preparacion especial** — no capturados por el modelo.

---

## 12. Gestion del capital (Kelly Criterion)

El criterio de Kelly calcula que fraccion del capital apostar para maximizar el crecimiento a largo plazo:

**Formula:** `f = (P × b - (1 - P)) / b`

Donde:
- P = probabilidad del modelo
- b = cuota decimal - 1

**Ejemplo:** P = 0.65, cuota = 1.90 → b = 0.90
→ f = (0.65 × 0.90 - 0.35) / 0.90 = **9.4% del capital**

**Recomendacion practica:** usar Kelly fraccional (apostar entre el 25% y el 50% del valor Kelly) para reducir la varianza y proteger el capital en rachas negativas.

| Kelly calculado | Apuesta recomendada (Kelly 1/4) |
|---|---|
| 10% | 2.5% del capital |
| 8% | 2.0% |
| 5% | 1.25% |

---

## 13. Como usar el predictor paso a paso

1. Selecciona el **tour** (ATP o WTA) y la **superficie** del partido.
2. Elige los dos jugadores usando la barra de busqueda (escribe para filtrar).
3. Pulsa **PREDECIR**.
4. Lee la probabilidad y la cuota justa.
5. Si tienes las cuotas de la casa, ingrésalas en el campo opcional para ver el EV y el Kelly.
6. Consulta el radar de perfil para entender por que el modelo favorece a uno u otro.
7. Usa el nivel de confianza como referencia para decidir cuanto apostar.

---

## 14. Actualizacion diaria de datos

Los datos se actualizan automaticamente cada dia a las 7:00 si configuraste el Task Scheduler (`python setup_scheduler.py`). También puedes forzar la actualizacion manualmente desde la barra lateral de la aplicacion.

La actualizacion descarga los partidos mas recientes del año en curso y recalcula el ELO y la forma de todos los jugadores. El modelo en si no se reentrena diariamente — solo se actualiza el estado de los jugadores.
