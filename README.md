# 🏥 MediScribe — Clinical NER with Fine-Tuned BioBERT

**MediScribe** turns unstructured clinical text — notes, discharge summaries, prescriptions — into structured, machine-readable data. It fine-tunes **BioBERT** on the **BC5CDR** biomedical corpus to detect **Disease** and **Chemical/Drug** entities, then serves the model through a live **Gradio** web app on Hugging Face Spaces.

[![Live Demo](https://img.shields.io/badge/🤗%20Demo-Live%20App-blue)](https://huggingface.co/spaces/RATHANSUMBET14/mediscribe-demo)
[![Model](https://img.shields.io/badge/🤗%20Model-mediscribe--biobert--ner-yellow)](https://huggingface.co/RATHANSUMBET14/mediscribe-biobert-ner)
[![Dataset](https://img.shields.io/badge/Dataset-BC5CDR-green)](https://huggingface.co/datasets/tner/bc5cdr)

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🚀 Live Demo (Gradio) | https://huggingface.co/spaces/RATHANSUMBET14/mediscribe-demo |
| 🤖 Fine-Tuned Model | https://huggingface.co/RATHANSUMBET14/mediscribe-biobert-ner |
| 📚 Dataset | [tner/bc5cdr](https://huggingface.co/datasets/tner/bc5cdr) |
| 🧠 Base Checkpoint | [dmis-lab/biobert-v1.1](https://huggingface.co/dmis-lab/biobert-v1.1) |

---

## 📋 Overview

Clinical notes and prescriptions are written as free text, so the facts that matter most — diagnoses, drugs, dosages — are buried in narrative sentences instead of stored as structured data. Manually reading and tagging every chart is slow, expensive, and inconsistent between reviewers.

MediScribe addresses this with a **Named Entity Recognition (NER)** pipeline: given a clinical sentence, it labels every span of text that refers to a **Disease** or a **Chemical/Drug**, and returns the result as clean, structured JSON — ready for an EHR, billing system, or research pipeline to ingest.

```text
Input:  "Prescribed 325 mg Aspirin for chest pain."
Output: Aspirin     → Chemical
        chest pain  → Disease
```

---

## 🧠 Model

| | |
|---|---|
| **Base checkpoint** | `dmis-lab/biobert-v1.1` (pretrained on PubMed abstracts + PMC full-text articles) |
| **Task head** | `AutoModelForTokenClassification` |
| **Framework** | Hugging Face `Trainer` API |
| **Epochs** | 3 |
| **Batch size** | 16 (train & eval) |
| **Learning rate** | 2e-5 |
| **Weight decay** | 0.01 |
| **Model selection** | Best checkpoint by eval F1 |

BioBERT was chosen over generic BERT and ClinicalBERT because its PubMed/PMC pretraining gives it a biomedical vocabulary and a strong sense of clinical entity boundaries, while remaining a fully open checkpoint that doesn't require a data-use agreement (unlike MIMIC-III-pretrained ClinicalBERT for some use cases).

## 📚 Dataset

[`tner/bc5cdr`](https://huggingface.co/datasets/tner/bc5cdr) — the BioCreative V CDR corpus: PubMed abstracts annotated for chemical–disease relations, in BIO tagging format.

| Split | Sentences |
|---|---|
| Train | 5,228 |
| Validation | 5,330 |
| Test | 5,865 |

**Entity taxonomy (BIO scheme):**

| Tag | Meaning |
|---|---|
| `O` | Not an entity |
| `B-Disease` / `I-Disease` | Start / continuation of a disease span |
| `B-Chemical` / `I-Chemical` | Start / continuation of a chemical/drug span |

> **Note:** BC5CDR only covers `Chemical` and `Disease`. Extending to the full `Drug / Dosage / Route / Frequency / Symptom / Disease` taxonomy would require an i2b2/n2c2 medication-extraction dataset, which needs a data-use agreement since it contains real de-identified clinical notes.

---

## ⚙️ How it works

1. **Tokenization & subword alignment** — WordPiece splits rare clinical terms (e.g. `Hydrochlorothiazide`) into subword fragments. Only the first fragment of each word keeps the true label; continuation fragments are masked with `-100` so they're ignored by the loss.
2. **Fine-tuning** — BioBERT is fine-tuned as a token classifier on BC5CDR using the Hugging Face `Trainer`.
3. **Evaluation** — scored with [`seqeval`](https://github.com/chakki-works/seqeval) for strict span-level precision/recall/F1 (an entity only counts as correct if both its boundaries *and* its label match exactly).
4. **Span merging / post-processing** — a custom `clean_and_merge_spans()` function merges adjacent same-label subword fragments (e.g. `"A"` + `"##cute Myocardial Infarction"` → `"Acute Myocardial Infarction"`) and re-slices the clean text from the original string, removing every `##` artifact.
5. **Deployment** — the fine-tuned model and tokenizer are pushed to the Hugging Face Hub via `push_to_hub()`, then loaded into a Gradio app deployed as a Hugging Face Space.

---

## 📊 Results

Evaluated on the held-out BC5CDR test split:

| Metric | Score |
|---|---|
| Precision | 84.6% |
| Recall | 90.4% |
| **F1 (seqeval, strict span-level)** | **87.4%** |
| Token-level accuracy | 97.6% |
| Final training loss | 0.192 |

Seqeval's strict span-F1 is a much higher bar than token accuracy — it only counts an entity as correct if the *entire* span boundary and label match exactly, not just individual tokens.

---

## 🖥️ Live Demo

Try it directly: **[huggingface.co/spaces/RATHANSUMBET14/mediscribe-demo](https://huggingface.co/spaces/RATHANSUMBET14/mediscribe-demo)**

The Gradio app:
1. Takes a pasted clinical note, discharge summary, or prescription.
2. Runs the fine-tuned BioBERT pipeline (`aggregation_strategy="simple"` merges subwords back into whole words).
3. Displays entities highlighted directly in the text (`HighlightedText`).
4. Returns the same result as structured JSON (`JSON` panel), ready for downstream systems.

**Example output:**

```json
{
  "diseases": [
    { "entity": "chest tightness", "confidence": 0.9796, "span": [30, 45] },
    { "entity": "Acute Myocardial Infarction", "confidence": 0.9730, "span": [86, 113] },
    { "entity": "Hypertension", "confidence": 0.9203, "span": [128, 140] }
  ],
  "chemicals_and_drugs": [
    { "entity": "Aspirin", "confidence": 0.8481, "span": [162, 169] },
    { "entity": "Metformin", "confidence": 0.9563, "span": [182, 191] },
    { "entity": "Lisinopril", "confidence": 0.9521, "span": [204, 214] }
  ]
}
```

---

## 🚀 Quickstart

### Use the model directly (no training required)

```bash
pip install transformers
```

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

repo_id = "RATHANSUMBET14/mediscribe-biobert-ner"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForTokenClassification.from_pretrained(repo_id)

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # merges subwords back into whole words
)

text = "Patient presented with severe chest tightness. Diagnosed with Acute Myocardial Infarction. Prescribed 325 mg Aspirin and Metformin."
for ent in ner_pipeline(text):
    print(f"{ent['word']:<30} | {ent['entity_group']:<10} | {ent['score']:.3f}")
```

### Reproduce training from scratch

```bash
pip install transformers datasets evaluate seqeval accelerate torch
```

Open `mediscribe-project-using-nlp.ipynb` and run all cells top to bottom. It will:
1. Load `tner/bc5cdr` from Hugging Face.
2. Tokenize and align labels with subword masking.
3. Fine-tune `dmis-lab/biobert-v1.1` for 3 epochs.
4. Evaluate with `seqeval`.
5. Save the model locally and optionally push it to the Hugging Face Hub.

> ⚠️ If you push to your own Hub repo, authenticate with `huggingface-cli login` or an environment variable — **never hardcode your access token in the notebook.**

---

## 🗂️ Project structure

```
.
├── mediscribe-project-using-nlp.ipynb   # end-to-end training + deployment notebook
└── README.md
```

---

## 🧭 Challenges & Future Scope

**Challenges**
- Only 2 entity types today (Chemical, Disease) vs. the fuller clinical taxonomy (Drug, Dosage, Route, Frequency, Symptom, Disease).
- Real-world clinical notes are noisier than clean PubMed abstracts.
- Full medication-extraction datasets (i2b2/n2c2) require a data-use agreement for real de-identified patient data.

**Future scope**
- Extend to i2b2/n2c2 medication data for Dosage / Route / Frequency extraction.
- Quantize to ONNX for faster, cheaper inference at scale.
- Integrate structured output with FHIR-based EHR systems.

---

## 🙏 Acknowledgments

- [BioBERT](https://github.com/dmis-lab/biobert) — Lee et al., DMIS Lab
- [BC5CDR corpus](https://huggingface.co/datasets/tner/bc5cdr) — BioCreative V Chemical-Disease Relation task
- [Hugging Face](https://huggingface.co/) — `transformers`, `datasets`, `evaluate`, Hub, and Spaces
- [Gradio](https://www.gradio.app/) — demo interface

## 📄 License

Specify your license here (e.g. MIT) — add a `LICENSE` file to the repo root.
