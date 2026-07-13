# Agente autónomo para Overcooked-AI — Informe técnico

> **Grupo:** DeepN — Integrantes: Barzola Estrella, Johar Jared · Acervo Correa, Renzo Alfonso
> Curso de Deep Learning — Proyecto final (competencia Overcooked-AI)

## 1. Problema

Diseñar un agente autónomo que colabore con un compañero desconocido para preparar y entregar la mayor cantidad de sopas en episodios de 250 timesteps, evaluado con la fórmula oficial `10000·sopas + 10·(250−t_última) + (250−t_primera) − min(100·timeouts, 5000)`, en 6 escenarios con compañeros crecientes en dificultad (greedy_full_task, greedy con sticky/random actions, random_motion y agentes de otros grupos) y layouts parcialmente desconocidos.

## 2. Datos

- **Dataset colectivo del curso**: 152,857 transiciones de los 23 grupos (137,857 humanas + 15,000 de bots), consolidadas de 564 archivos con estructuras heterogéneas en un formato unificado (`data/processed/unified.npz`), con control de calidad: validación de esquema (obs 96-dim float32), detección de duplicados (0), episodios truncados y episodios sin entregas (28%).
- **Observación**: vector de 96 features egocéntricas (`featurize_state` del repo oficial) — orientación, objeto en mano, deltas a ingredientes/ollas/entrega más cercanos, estado de 2 ollas, features del compañero, posiciones relativa y absoluta. Es independiente del tamaño del layout, lo que habilita generalización a layouts nuevos.
- **59 layouts** presentes en los datos; pool de entrenamiento de 58 layouts válidos (14 oficiales + 44 customs de los grupos, validados y deduplicados).

## 3. Método (mapeo con el sílabo)

### 3.1 Fase 1 — Behavioral Cloning (Imitation Learning) [sílabo 5.1]

Clasificador de acciones π(a|s): **MLP 98→256→256→6** (obs z-scoreada + one-hot del índice de agente) con **LayerNorm + tanh + Dropout 0.15** [5.1.5 regularización], entrenado con **AdamW + cosine schedule con warmup** [5.1.3 optimizers], **label smoothing** y **early stopping** [5.1.6 generalización] sobre las demostraciones humanas.

Decisiones clave contra el sobreajuste y el desbalance [5.1.4 bias-variance]:
- **Ponderación de clases** (46% de las acciones humanas son "stay"; peso ∝ frecuencia^-0.4) — sin ella el modelo colapsa a quedarse quieto.
- **Ponderación por calidad**: episodios sin ninguna sopa entregada (28%) pesan 0.3.
- **Validación leave-groups-out** (los grupos de validación no aparecen en train): mide generalización a demostradores nuevos, no memorización.

**Resultado BC**: val_acc = 51.6%, macro-F1 = 0.50. En evaluación oficial: 4.0 sopas/ep con greedy en coordination_ring, pero 0 sopas en solitario → motiva la Fase 2.

### 3.2 Fase 2 — Destilación de un experto planificador (teacher-student + DAgger) [sílabo 5.6]

El BC humano (51.6% acc) entrega sopas con un buen compañero pero **no sabe completar el ciclo solo** (0 sopas con compañero pasivo) — fatal para el escenario 4. Para superar ese techo aplicamos **knowledge distillation** de un experto planificador:

1. **Auditamos al `GreedyFullTaskPolicy` del curso** (el compañero oficial de los escenarios 1-3) y encontramos un bug: elige objetivos por distancia Manhattan sin verificar alcanzabilidad. En layouts con zonas separadas (p.ej. `asymmetric_advantages`) apunta a la ventanilla del lado contrario, su BFS no encuentra camino y **se congela para siempre sosteniendo la sopa** (0 sopas en toda configuración — verificado empíricamente).
2. **Construimos un experto mejorado** (`experts/improved_greedy.py`): selección de objetivos por distancia real de camino (BFS) con conciencia de alcanzabilidad, plan B ignorando el bloqueo del compañero, y sidestep determinista anti-deadlock. Resultado: 0→5 sopas en solitario en asymmetric_advantages; 5.5-8.0 sopas en equipo con el greedy original.
3. **Destilación con DAgger**: el experto genera ~870k transiciones etiquetadas en 58 layouts × ambos roles × 5 tipos de compañero; se entrena un clon MLP; el clon juega y el experto **re-etiqueta los estados que el clon visita** (segunda iteración, ~870k transiciones más) — esto corrige el *compounding error* del BC puro (verificado: un clon con 99% de accuracy pero sin DAgger hace 0 sopas solo).
4. **Lecciones empíricas documentadas** (sección 4): las etiquetas del desatasco aleatorio contaminan el dataset (acc 99%→86% y peor rollout); la corrección es que el experto etiquete siempre acciones deterministas con propósito. La observación featurizada sí codifica alcanzabilidad (verificamos que el delta al "serving más cercano" usa distancias del motion planner), por lo que el clon puede aprender el enrutamiento correcto.

### 3.3 Fase 3 — Fine-tuning con Reinforcement Learning (PPO) [sílabo 5.10]

Actor-Critic con **redes separadas** para política y valor (el value loss de Overcooked es ~2 órdenes mayor que el policy loss y con torso compartido aplasta la señal de política — verificado empíricamente). PPO clip (γ=0.99, λ_GAE=0.98, clip 0.1, entropy annealing 0.02→0.003, lr annealing), **inicializado desde los pesos de BC** (pipeline BC→PPO de Carroll et al. 2019, "On the Utility of Learning about Humans for Human-AI Coordination").

- **Reward shaping annealado** [5.10.1]: recompensa densa (+3 ingrediente en olla, +3 recoger plato, +5 recoger sopa) que decae linealmente a 0 en 4M steps, dejando solo la recompensa real (+20 por sopa).
- **Población de compañeros (Fictitious Co-Play simplificado)**: cada episodio se samplea un compañero de {greedy_full_task exacto del curso, greedy+sticky p~U[0,0.5], greedy+ε-random ε~U[0,0.3], random_motion, stay, clon BC congelado (proxy de los agentes de otros grupos)} y un **rol aleatorio (jugador 0 o 1)** — cubre exactamente las condiciones de los 6 escenarios, incluido el cambio de rol.
- **Randomización de layouts**: cada episodio se samplea uno de los 58 layouts → un **generalista** robusto para layouts no vistos (escenarios 5-6).
- **Especialistas por layout**: al revelarse los layouts de los escenarios 1-4, se re-entrena un especialista por layout (~4M steps, ~1h) contra la población del escenario correspondiente.

### 3.4 Agente entregable

Clase `StudentAgent` según la plantilla oficial, **autocontenida (solo numpy)**, inferencia < 1ms (límite: 100ms):
- **Bundle de pesos** con generalista + especialistas y **enrutamiento por fingerprint**: la observación de t=0 es determinista por (layout, rol) — verificado bit a bit entre seeds — así el agente detecta el layout y activa el especialista; ante un layout desconocido usa el generalista.
- **Anti-atasco**: si la posición absoluta (dims 94/95) lleva ≥3 steps congelada con acciones de movimiento, emite un movimiento aleatorio distinto (técnica estándar del benchmark).
- **Robustez total**: sin excepciones hacia fuera, acción siempre válida, 0 timeouts medidos.

## 4. Resultados en los escenarios oficiales de la competencia

Bundle final: generalista (PPO fine-tune multi-layout) + especialistas por layout (PPO enfocado con el compañero exacto del escenario), con enrutamiento automático por fingerprint de la observación inicial. Evaluado en las condiciones exactas reveladas (3 seeds × 2 roles):

| Escenario | Layout + compañero | Sopas/ep | Score oficial |
|---|---|---|---|
| 1 | asymmetric_advantages + greedy | **12.50** | 125,366 |
| 2 | coordination_ring + greedy sticky | **4.67** | 47,225 |
| 3 | counter_circuit + greedy sticky+random | **1.83** | 19,002 |

Hallazgos específicos por escenario: (1) en asymmetric_advantages el compañero oficial sufre un deadlock por objetivos inalcanzables — nuestro agente completa el ciclo solo; (3) en counter_circuit las órdenes son mixtas y una sopa de 3 ingredientes iguales vale 0 — el compañero (configurado con cebollas) produce sopas sin valor, así que el agente aprendió por RL a completar recetas insertando tomates (regla emergente: ninguna olla debe terminar con 3 iguales).

## 5. Protocolo de evaluación

Harness que replica exactamente el runner oficial del curso (mismo loop, seeds, wrappers de seguridad de 100ms y cambio de rol) y calcula la fórmula de score oficial. Protocolo: 3 seeds × 2 roles × {greedy, greedy_sticky, greedy_eps, random_motion, stay} × layouts.

### Resultados del modelo final (destilación DAgger + fine-tune PPO, 3 seeds × 2 roles, horizon 250)

Selección de modelo por torneo: 5 checkpoints candidatos evaluados con la matriz completa (15 celdas × 6 rollouts); gana el fine-tune PPO temprano por total ponderado **y** por mejor peor-celda (ninguna celda bajo 2.0 sopas):

| Layout (sopas/ep) | greedy | greedy+sticky 0.3 | greedy+ε 0.15 | random_motion | stay |
|---|---|---|---|---|---|
| cramped_room | **7.00** | 4.33 | 4.50 | 5.00 | 2.50 |
| asymmetric_advantages | **6.50** | 6.33 | **7.00** | 5.00 | 5.00 |
| coordination_ring | **6.00** | 4.00 | 5.67 | 2.83 | 2.00 |

Latencia media del agente: 1-3 ms (límite 100 ms); **0 timeouts** e inválidas en toda la matriz. Referencias: el compañero oficial `greedy_full_task` jugando con otro greedy logra ~4.2 sopas en cramped_room (nuestro agente con ese mismo compañero: 7.0); el BC humano puro lograba 0-4 sopas solo con compañero greedy y 0 en solitario. La contribución del fine-tuning con RL sobre la destilación pura: +2.8 sopas en cramped+sticky, +1.7 en coordination+sticky y +3.8 en coordination+ε (los escenarios 2-3 de la competencia), manteniendo el resto.

**Lección de ingeniería documentada**: una capa oculta cuadrada (256×256) hace indetectable por formas la orientación de la matriz en el export numpy; un bundle con W1 transpuesta rendía como política casi aleatoria (20% de accuracy efectiva) pese a 99% en validación. El fix: declarar `weight_layout` en el metadato del export y validar accuracy del artefacto exportado contra el dataset (test de paridad export↔entrenamiento), no solo del modelo en memoria.

## 6. Reproducibilidad

```
tools/build_dataset.py      # dataset unificado
train/train_bc.py           # Fase 1 (BC)
train/ppo/train_ppo.py      # Fase 2 (PPO generalista)
tools/train_specialist.py   # especialistas por layout
tools/build_bundle.py       # empaquetado del entregable
tools/evaluate_agent.py     # evaluación con score oficial
```

## 7. Referencias

- Carroll et al. (2019). *On the Utility of Learning about Humans for Human-AI Coordination.* NeurIPS.
- Strouse et al. (2021). *Collaborating with Humans without Human Data* (Fictitious Co-Play). NeurIPS.
- Repo oficial: HumanCompatibleAI/overcooked_ai.
