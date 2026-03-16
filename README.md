# Can Small Fine-Tuned Models Match the Quality of Large LLMs for Mental Health Support?

## Project Overview

This project investigates whether a small, fine-tuned language model can produce mental health counseling responses of comparable quality to large LLMs prompted with sophisticated techniques. The study compares four distinct approaches across eight therapeutic evaluation dimensions.

## Research Question

> *Can a small fine-tuned model (LLaMA 3 8B + QLoRA) match the quality of a large LLM (Gemini 2.5 Flash) when provided with zero-shot, chain-of-thought, or role-based prompts for mental health support?*

---

## Approaches Compared

| # | Approach | Description |
|---|----------|-------------|
| 1 | **Fine-Tuned Model** | LLaMA 3 8B fine-tuned on mental health Q&A data using QLoRA |
| 2 | **Zero-Shot Prompting** | Gemini 2.5 Flash with a plain question and ~300-word constraint |
| 3 | **Chain-of-Thought (CoT) Prompting** | Gemini 2.5 Flash guided through 5-step therapeutic reasoning |
| 4 | **Role Prompting** | Gemini 2.5 Flash instructed to act as a licensed counselor |

---

## Datasets

Three Hugging Face datasets were merged and shuffled to form the training and test sets:

| Dataset | Size | Description |
|---------|------|-------------|
| `Amod/mental_health_counseling_conversations` | 3,512 rows | Counseling-style Q&A pairs |
| `mpingale/mental-health-chat-dataset` | 2,612 rows | Therapist-answered questions with metadata |
| `heliosbrahma/mental_health_chatbot_dataset` | 172 rows | General mental health Q&A |

**Final split:**
- Training set: ~6,195 examples
- Test set: 101 examples

---

## Model & Fine-Tuning

### Base Model
- **Meta LLaMA 3 8B Instruct** (`meta-llama/Meta-Llama-3-8B-Instruct`)

### Quantization
- 4-bit quantization using `BitsAndBytesConfig` (NF4, double quantization, float16 compute)

### LoRA Configuration
- `lora_alpha`: 16
- `lora_dropout`: 0.1
- `r` (rank): 8
- `task_type`: CAUSAL_LM

### Training Parameters
- Epochs: 3
- Batch size: 2 (per device)
- Learning rate: 6e-5
- Optimizer: `paged_adamw_32bit`
- Max sequence length: 128 tokens
- Scheduler: Constant LR

---

## Evaluation: LLM-as-a-Judge

All four approaches were evaluated by **Gemini 2.5 Flash** acting as a judge, scoring responses across eight criteria:

| Criterion | Weight |
|-----------|--------|
| Empathy and Validation | 20% |
| Clinical Appropriateness and Safety | 20% |
| Therapeutic Depth and Insight | 15% |
| Cultural Competence and Sensitivity | 15% |
| Comprehensiveness | 10% |
| Actionability and Guidance | 10% |
| Engagement and Therapeutic Alliance | 5% |
| Language Quality and Communication | 5% |

Each response is scored on a **1–10 scale**, and a weighted total score out of 10 is computed.

---

## Results Summary

### Comparison of All Four Approaches (101 test questions)

| Criterion | Fine-Tuned Model | Zero-Shot + LLM | CoT + LLM | Role Prompt + LLM |
|-----------|:-:|:-:|:-:|:-:|
| Empathy & Validation | 229 | 246 | **281** | 276 |
| Clinical Appropriateness | 206 | 250 | **267** | 261 |
| Therapeutic Depth | 188 | 230 | **259** | 263 |
| Cultural Competence | 222 | 221 | **244** | 243 |
| Comprehensiveness | 162 | 245 | **258** | 252 |
| Actionability | 160 | **251** | 236 | 241 |
| Engagement | 187 | 236 | **283** | 279 |
| Language Quality | 182 | 248 | **277** | 276 |
| **TOTAL** | **1536** | **1927** | **2105** | **2091** |

### Fine-Tuned vs. Base Model (25 test questions)

| Criterion | Base LLaMA 3 8B | Fine-Tuned LLaMA 3 8B |
|-----------|:-:|:-:|
| Empathy & Validation | 18 | **19** |
| Clinical Appropriateness | 15 | **17** |
| Therapeutic Depth | 15 | 15 |
| Cultural Competence | 15 | 15 |
| Comprehensiveness | 12 | **14** |
| Actionability | 8 | **16** |
| Engagement | 16 | **17** |
| Language Quality | **15** | 14 |
| **TOTAL** | **114** | **127** |

**Key finding:** Fine-tuning significantly improved the base model (especially Actionability +100%), but prompted large LLMs still outperform the fine-tuned small model overall. CoT prompting achieved the highest total score.

---

## Project Structure

```
NLP_Final_project.ipynb     # Main notebook
sample_data/
  ├── test_df.csv                      # Held-out test set (101 examples)
  ├── fine_tuned_model_answers.csv     # Responses from fine-tuned LLaMA
  ├── zero_shot_prompt_answers.csv     # Zero-shot Gemini responses
  ├── cleaned_cot_prompt_answers.csv   # CoT Gemini responses (cleaned)
  ├── role_prompt_answers.csv          # Role-prompted Gemini responses
  ├── llm_as_a_judge.csv               # All 4 responses + judge scores
  └── llm_as_a_judge_base_llama.csv    # Base vs. fine-tuned comparison
```

---

## Setup & Requirements

### Prerequisites
- Python 3.10+
- GPU with at least 16GB VRAM (T4 or better recommended)
- HuggingFace account with access to `meta-llama/Meta-Llama-3-8B-Instruct`
- Google Gemini API key

### Installation

```bash
pip install datasets evaluate rouge_score loralib accelerate bitsandbytes trl peft transformers
pip install google-genai
```

### Authentication

```python
from huggingface_hub import login
login(token="YOUR_HUGGINGFACE_TOKEN")

API_KEY = "YOUR_GEMINI_API_KEY"
```

---

## How to Run

1. **Install dependencies** (first notebook cell)
2. **Login** to HuggingFace
3. **Run data processing** — loads and merges the three datasets, creates a train/test split
4. **Fine-tune the model** — runs QLoRA training (~24 minutes on T4 GPU)
5. **Generate answers** from all four approaches
6. **Evaluate** using the LLM-as-a-judge pipeline
7. **Aggregate scores** and view the summary table

> **Note:** The CoT responses include reasoning steps before the counseling answer. A post-processing step extracts only the "Step 5: Formulate Response" section before evaluation.

---

## Key Libraries

| Library | Purpose |
|---------|---------|
| `transformers` | Model loading, tokenization, generation |
| `peft` | LoRA adapters |
| `trl` | SFT training with `SFTTrainer` |
| `bitsandbytes` | 4-bit quantization |
| `datasets` | Dataset loading and processing |
| `google-genai` | Gemini API client |
| `pandas` | Data manipulation |
| `scikit-learn` | Train/test split |