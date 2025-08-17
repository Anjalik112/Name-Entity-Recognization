# 🧬 Medical Named Entity Recognition & Linking (CADEC + LLMs)

This project performs **Named Entity Recognition (NER)** on medical forum posts to identify mentions of:
- **Drugs**
- **Diseases**
- **Symptoms**
- **Adverse Drug Reactions (ADRs)**  

It further links detected ADRs to **SNOMED-CT** and **MedDRA** codes using:
- **String matching (RapidFuzz / difflib)**
- **Semantic similarity (SentenceTransformers)**
- **LLM-powered evaluation (Groq LLM API)**

---

## 🔧 Project Setup & Installation

### 1️⃣ Python Version  
Use **Python 3.11.** (✅ tested).  

---
### 🔑 Environment Variables  

Create a `.env` file in the project root to store API keys and defaults:  

```env
DEFAULT_PROVIDER="groq"  
DEFAULT_MODEL="llama-3.1-8b-instant"  
GROQ_API_KEY="your_groq_api_key_here"
```

---

## 📂 Dataset

We use the **CADEC (Corpus of Adverse Drug Events)** dataset.  
The dataset is expected to be in the `cadec/` directory with the following structure:

cadec/
├── text/ # Raw forum post text
├── original/ # Original annotations: ADR, Drug, Disease, Symptom
├── meddra/ # ADR annotations mapped to MedDRA terminology
├── sct/ # ADR annotations linked to SNOMED-CT codes

---

## 🚀 Project Workflow

The pipeline is split into **6 tasks**, each implemented as both **`.py` scripts** and **Jupyter notebooks (`.ipynb`)**.

---

### 📌 Step 1: Entity Enumeration

**Script:** `task_1.py` / `task_1.ipynb`  

**Purpose:**
- Parse `.ann` files in `cadec/original/`  
- Count unique entities for each label type (**ADR, Drug, Disease, Symptom**)  
- Provide a first overview of dataset distribution  

**Output:** Entity statistics printed and stored  

The dataset contains **3399 unique ADR mentions**, which is substantially higher than Drugs (323), Diseases (164), and Symptoms (148).  
This highlights the dataset’s strong emphasis on **Adverse Drug Reactions**, aligning with its primary research goal.  

---

### 📌 Step 2: NER with Pre-trained & LLM Models

**Script:** `task_2_.py` / `task_2.ipynb`  

**Purpose:**
- Run **Named Entity Recognition (NER)** on raw forum posts (`cadec/text/`).  

**Models Used:**
- 🤗 Hugging Face: `d4data/biomedical-ner-all`  
- ⚡ Groq-hosted LLMs → for **fast, scalable inference** during experimentation  

**Process:**
1. Load text file  
2. Run token classification via NER pipeline  
3. Merge sub-word tokens into complete entities  
4. Map model output labels → `{ADR, Drug, Disease, Symptom}`  

**Output:**  
- `.ann`-style predictions stored in `predicted/`  
**Output:**  
- Extracted 6 entity spans (ADR, Drug, Symptom) from `ARTHROTEC.20.txt`.  
- Saved predictions in `.ann` format with entity IDs, labels, spans, and text.  


---
### 📌 Step 3: Standard Evaluation  

**Script:** `task_3.py` / `task_3.ipynb`  

**Purpose:**  
- Compare predictions (`predicted/`) against gold standard (`original/`)  
- Strict evaluation: entity is **correct only if both text and label exactly match**  

**Metrics:**  
- Precision  
- Recall  
- F1-score 

**Output**
**Output:**  
- Shows a sample of gold vs predicted entities with their labels and spans.  
- Reports evaluation metrics (**Precision, Recall, F1**) based on strict matching.  

---


### 📌 Step 4: ADR Evaluation with MedDRA  

**Script:** `task_4.py` / `task_4.ipynb`  

**Purpose:**  
- Specialized evaluation focused **only on ADRs**  
- Uses `cadec/meddra/` annotations as gold standard  
- Evaluates how well the model detects **Adverse Drug Reactions** 

**output**
**Output:**  
- Displays ground truth vs predicted ADR entities side by side.  
- Reports strict evaluation metrics (**TP, FP, FN, Precision, Recall, F1**).  


### 📌 Step 5: Relaxed Evaluation  

**Script:** `task_5.py` / `task_5.ipynb`  

**Purpose:**  
- Provides a **more forgiving evaluation**  
- A prediction counts as **True Positive** if its span **overlaps** with a ground truth span of the same label  
- Useful when the model’s predicted boundaries differ slightly from the annotated spans  
**Output:**  
- Prints per-label and overall **Precision, Recall, F1-scores** under relaxed matching.  
- Highlights cases where predictions partially overlap but are still counted as correct.  


### 📌 Step 6: ADR Entity Linking (SNOMED-CT)  

**Script:** `task_6.py` / `task_6.ipynb`  

**Purpose:**  
- Normalize detected ADRs by linking them to **SNOMED-CT medical concepts**  

**Linking Approaches:**  
1. 🔎 **Fuzzy String Matching** (RapidFuzz / Difflib)  
2. 🧠 **Sentence Embedding Similarity** (SentenceTransformers + Cosine similarity)  

**Outputs:**  
- Comparison tables → *Predicted ADRs vs SNOMED concepts*  
- Agreement statistics → *(Fuzzy vs Embedding match)*  
- Joined catalog → *Original ↔ SNOMED mappings*  


### 🤖 LLM Integration  

We also integrate **Groq LLM API** for:  
- 📄 JSON parsing & strict output validation  
- 📊 Evaluations on large batches  
- ⚡ Automating ADR-code mapping beyond embeddings  

**Models Used:**  
- 🦙 `llama3-70b-8192` (via **Groq API**)  
- 🔎 `sentence-transformers/all-MiniLM-L6-v2` (local embeddings for semantic similarity)  
