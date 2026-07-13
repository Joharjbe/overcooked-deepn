# Overcooked-AI — Agente del grupo DeepN

**Integrantes:** Barzola Estrella, Johar Jared · Acervo Correa, Renzo Alfonso

Repositorio de competencia basado en el código oficial del curso (`final.zip`), con
nuestro agente ya configurado en `configs/competition.yaml`.

## El agente (`policies/deepn/`)

- `student_agent.py` — clase `StudentAgent` (plantilla oficial: `__init__(config)`,
  `reset()`, `act(obs) -> int`). Autocontenido: **solo numpy**. Latencia <1 ms
  por acción (límite: 100 ms); cero timeouts en toda nuestra batería.
- `student_weights.npz` — pesos: un modelo generalista + especialistas por layout
  con selección automática (el agente reconoce el layout por su observación
  inicial). **Debe permanecer junto al .py.**

Método (detalles en [INFORME.md](INFORME.md)): imitation learning sobre el dataset
colectivo del curso → destilación DAgger de un experto planificador propio →
fine-tuning PPO contra una población de compañeros (greedy oficial, variantes
sticky/random, random_motion, clones congelados).

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución (comandos oficiales del curso)

```bash
# Un escenario específico (1..6):
python -m src.evaluate_competition --config configs/competition.yaml --scenario 1

# Con renderización:
python -m src.evaluate_competition --config configs/competition.yaml --render
```

Resultados de referencia obtenidos con este mismo evaluador y estos configs
(promedio de las 4 seeds oficiales):

| Escenario | Sopas | Score |
|---|---|---|
| 1 (asymmetric + greedy) | 22.0 | 220,426 |
| 2 (coordination_ring + sticky 0.10) | 8.5 | 86,526 |
| 3 (counter_circuit + sticky 0.15 + random 0.05) | 13.5 | 135,672 |
| 4 (scenario_4 + random_motion, swap de roles) | 10.0 | 100,435 |
