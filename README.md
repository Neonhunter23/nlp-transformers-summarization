# Transformers for News Summarization: Encoder-Decoder vs Decoder-Only

Repositorio de mi trabajo de Transformers y NLP del máster de Data Science en La Salle (Universitat Ramon Llull), en el que comparo modelos encoder-decoder y decoder-only para resumen automático de noticias.

## Pregunta central

> ¿Puede un LLM decoder-only moderno y pequeño (Qwen3-1.7B) adaptado mediante LoRA competir con un encoder-decoder diseñado para seq2seq (Flan-T5-base) con full fine-tuning, en una tarea de resumen abstractivo de noticias?

## Resultados principales

### Evolución V1 → V2 → V3

| Modelo | Versión | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|---|
| BART-large-cnn *(ref)* | Zero-shot | 35.12 | 14.54 | 25.54 | 29.37 |
| Pegasus-cnn *(ref)* | Zero-shot | 34.97 | 14.35 | 25.81 | 31.83 |
| **Flan-T5-base** | **V3 best** | **33.14** | **13.00** | **23.45** | **27.35** |
| Flan-T5-base | V1 zero-shot | 24.96 | 9.03 | 19.17 | 21.64 |
| **Qwen3-1.7B** | **V3 best** | **30.41** | **8.18** | **20.78** | **27.75** |
| Qwen3-1.7B | V1 zero-shot | 22.56 | 5.29 | 14.91 | 18.17 |

### LLM-as-judge (1-5)

| Modelo | Coherencia | Fidelidad | Fluidez |
|---|---|---|---|
| Flan-T5-base | **4.00** | **4.18** | **4.18** |
| Qwen3-1.7B | 3.46 | 3.80 | 3.76 |

### Hallazgos clave

- **El fine-tuning aporta ~8 puntos de ROUGE-1** en ambas familias con solo 10-20k muestras.
- **T5-base (250M) supera a Qwen3 (1.7B) en ROUGE-1 y calidad percibida** — la arquitectura encoder-decoder sigue siendo más eficiente por parámetro para tareas seq2seq.
- **Qwen3 gana en ROUGE-Lsum y estructura narrativa** — resúmenes mejor segmentados por frases.
- **Qwen3 con LoRA es 12x más eficiente en training** (56 min vs 755 min, 0.37% de parámetros entrenados).
- **ROUGE y calidad percibida no correlacionan perfectamente** — evaluación multi-dimensional es imprescindible.

## Modelos

| Modelo | Parámetros | Tipo | Fine-tuning | max_input_length |
|---|---|---|---|---|
| [Flan-T5-base](https://huggingface.co/google/flan-t5-base) | 250M | Encoder-decoder | Full (fp32 + AdamW) | 512 |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | Decoder-only | LoRA r=16 (bf16) | 1536 (1024 training) |
| [BART-large-cnn](https://huggingface.co/facebook/bart-large-cnn) | 400M | Encoder-decoder | Pre-tuned (referencia) | 1024 |
| [Pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail) | 568M | Encoder-decoder | Pre-tuned (referencia) | 1024 |

## Dataset

[CNN/DailyMail 3.0.0](https://huggingface.co/datasets/cnn_dailymail) — 300k+ pares artículo-resumen de noticias en inglés.

- Artículos: ~600 palabras de media (~850 tokens T5, ~770 tokens Qwen3)
- Resúmenes: ~35-45 palabras (~66 tokens T5, ~57 tokens Qwen3)
- Ratio de compresión: 5-7%

## Estructura del proyecto

```
transformers-summarization/
├── config/
│   └── config.yaml                     # Hiperparámetros, rutas, seeds
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploración del dataset
│   ├── 02_baseline.ipynb               # V1: inferencia zero-shot
│   ├── 03_finetune_t5.ipynb            # V2: fine-tuning T5-base
│   ├── 04_finetune_qwen.ipynb          # V2: LoRA fine-tuning Qwen3
│   ├── 05_v3_t5_search.ipynb           # V3: HP search T5
│   ├── 06_v3_qwen_search.ipynb         # V3: HP search Qwen3
│   ├── 07_llm_judge.ipynb              # Evaluación con LLM-as-judge
│   ├── 08_interpretability.ipynb       # Attention maps y análisis
│   ├── 09_error_analysis.ipynb         # Clasificación de errores
│   └── 10_final_analysis.ipynb         # Tablas y gráficos finales
├── src/
│   ├── data/
│   │   ├── loader.py                   # Carga de dataset y config
│   │   └── preprocessing.py            # Tokenización por arquitectura
│   ├── models/
│   │   └── loader.py                   # Wrappers unificados de modelos
│   ├── training/
│   │   ├── trainer.py                  # Builders de Trainer seq2seq/causal
│   │   └── experiments.py              # Runner de HP search con skip/resume
│   ├── evaluation/
│   │   ├── metrics.py                  # ROUGE, BERTScore
│   │   ├── inference.py                # Generación unificada
│   │   └── llm_judge.py                # LLM-as-judge local
│   └── interpretability/
│       └── attention_viz.py            # Attention maps y token importance
├── results/
│   ├── checkpoints/                    # Modelos guardados (V2, V3)
│   ├── figures/                        # Gráficos exportados
│   └── tables/                         # CSVs con métricas
├── report/
│   └── informe.pdf                     # Memoria final
├── requirements.txt
├── .gitignore
└── README.md
```

## Metodología

### V1 — Baseline zero-shot
Inferencia directa de los 4 modelos sin fine-tuning. Establece la línea base y los techos de referencia (BART, Pegasus).

### V2 — Fine-tuning estándar
- **T5-base:** full fine-tuning en fp32, 10k muestras, 2 epochs.
- **Qwen3-1.7B:** LoRA (r=16, α=32), bf16, 10k muestras, 1 epoch.

### V3 — Optimización de hiperparámetros
3 configuraciones por modelo variando learning rate, epochs, datos, y (para Qwen3) LoRA targets. El runner soporta **reanudación desde checkpoint** y **skip de experimentos completados** para tolerancia a interrupciones.

### Evaluación avanzada
- **LLM-as-judge:** evaluación de coherencia, fidelidad y fluidez (1-5) con modelo local.
- **Interpretabilidad:** cross-attention maps (T5) y self-attention maps (Qwen3).
- **Análisis de errores:** clasificación automática de alucinaciones, omisiones, repeticiones.

## Requisitos

### Hardware
- GPU con ≥12 GB VRAM (desarrollado en RTX 5070)
- 32 GB RAM
- ~15 GB de disco para modelos + checkpoints

### Software

```bash
# Entorno
uv venv --python 3.11
.venv\Scripts\activate    # Windows
source .venv/bin/activate  # Linux

# PyTorch con CUDA (ajustar versión según GPU)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Dependencias
uv pip install -r requirements.txt
```

## Ejecución

Los notebooks se ejecutan en orden numérico. Cada uno depende de los anteriores:

```
01_eda → 02_baseline → 03/04_finetune → 05/06_search → 07/08/09_analysis → 10_final
```

Los notebooks de fine-tuning y HP search son largos (1-12h según la config). El runner de V3 (`experiments.py`) soporta interrupciones: si paras a mitad, al re-ejecutar se saltan los experimentos completados y se reanudan los parciales desde el último checkpoint.

## Decisiones técnicas relevantes

- **T5-base en lugar de T5-large:** T5-large es numéricamente inestable en bf16 en consumer GPUs, produciendo loss divergente y overflow errors. T5-base en fp32 entrena de forma estable y habilita una comparativa más interesante contra Qwen3 (7x diferencia de parámetros).
- **LoRA solo sobre attention layers:** incluir FFN layers (gate/up/down_proj) triplicó los parámetros entrenables y multiplicó el training por 10x sin mejora en ROUGE.
- **`max_input_length=1024` en training de Qwen3:** compromiso entre VRAM (12 GB) y truncación (~15% vs 5.6% con 1536). La inferencia usa los 1536 completos.
- **Reload limpio post-training para Qwen3:** el Trainer deja `gradient_checkpointing=True` y `use_cache=False`, que corrompen la generación con beam search. Se documenta el bug y la solución.

## Autor

Daniel Ruiz Jiménez — Máster en Data Science, La Salle (Universitat Ramon Llull), 2025-2026.