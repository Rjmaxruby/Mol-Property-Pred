# Molecular Toxicity Predictor

A deployable AI system that predicts the toxicity of chemical compounds across 12 biological targets simultaneously — from a chemical name or SMILES string, with no lab test required. Includes a white-box explainability module that generates per-target structural explanations and a cheminformatics narrative report powered by an LLM.

Built as a portfolio project demonstrating applied ML for real-world scientific problems, directly relevant to pharma and biotech AI/ML roles.

**Live API:** runs locally via FastAPI · interactive docs at `/docs`

---

## What it does

Enter a chemical name like `caffeine` or a SMILES string like `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` and the system returns:

- A 2D visualisation of the molecule
- Toxicity predictions across all 12 Tox21 biological targets
- Probability scores and confidence levels per target
- Model reliability (AUC-ROC) shown per target so you know how much to trust each prediction

Hit **Explain** and you additionally get:

- Atom-level saliency map highlighting which parts of the molecule drove each prediction
- Per-target structural analysis — different targets highlight different atoms
- A cheminformatics report written by Llama 4 Scout identifying the key functional groups, their known toxicological mechanisms, and concrete medicinal chemistry recommendations (bioisosteres, scaffold changes)

**Example — Dasatinib (cancer drug):** flags toxic on NR-AhR (80%), SR-p53 (81%), SR-ATAD5 (78%). The explain module identifies the pyrimidine ring and sulfonamide group as primary structural drivers, consistent with Dasatinib's known multi-target kinase inhibitor profile.

---

## Why this matters

Before a drug candidate can be tested on humans, it needs to be screened for toxicity. Running lab assays on thousands of molecules is expensive and slow. This system predicts toxicity from molecular structure alone — potentially flagging dangerous compounds earlier and at a fraction of the cost.

The explainability module closes a critical gap: most ML toxicity models are black boxes. This system tells a medicinal chemist *which part of the molecule* is driving the toxicity signal and *why* — output a chemist can actually act on.

---

## Dataset

**Tox21** — a real dataset created by the NIH, FDA, and EPA containing ~7,800 chemical compounds each tested against 12 biological pathways.

| Target | Description | Toxic % |
|---|---|---|
| NR-AR | Androgen Receptor | 4.3% |
| NR-AR-LBD | Androgen Receptor Ligand Binding Domain | 3.5% |
| NR-AhR | Aryl Hydrocarbon Receptor | 11.7% |
| NR-Aromatase | Aromatase Enzyme | 5.2% |
| NR-ER | Estrogen Receptor | 12.8% |
| NR-ER-LBD | Estrogen Receptor Ligand Binding Domain | 5.0% |
| NR-PPAR-gamma | PPAR-gamma receptor | 2.9% |
| SR-ARE | Antioxidant Response Element | 16.2% |
| SR-ATAD5 | DNA Damage Response | 3.7% |
| SR-HSE | Heat Shock Response Element | 5.8% |
| SR-MMP | Mitochondrial Membrane Potential | 15.8% |
| SR-p53 | p53 Tumour Suppressor | 6.2% |

All targets are heavily class-imbalanced (2.9%–16.2% toxic), which is a key challenge this project addresses directly.

---

## Model

### Architecture: Multi-Task Graph Attention Network (GAT)

Molecules are represented as graphs — atoms as nodes, bonds as edges — rather than flat fingerprint vectors. This preserves the actual chemical structure and allows the model to learn from molecular topology directly.

**Node features per atom (6):** element number, degree, formal charge, aromaticity, ring membership, hydrogen count

**Architecture:**
- 3 × GATConv layers (128 hidden dims, 4 attention heads) with dropout
- Global mean pooling — converts per-atom representations into a single molecule vector
- Shared dense layer (128 → 64) with dropout
- 12 independent output heads — one per toxicity target

**Training details:**
- Per-target `pos_weight` to handle class imbalance (range: 5.1× to 33.1× depending on target)
- Masked loss — only trains on labels that exist for each molecule
- Early stopping (patience=10, evaluated every 5 epochs)
- Adam optimizer, lr=0.001

### Baseline comparison

| Model | Mean AUC-ROC | Notes |
|---|---|---|
| Random Forest (ECFP fingerprints) | 0.7535 | Fast, interpretable via SHAP |
| XGBoost (ECFP fingerprints) | 0.6965 | Underperformed on sparse binary data |
| **Multi-Task GAT (final)** | **0.8284** | Best overall |

The upgrade from GCNConv to GATConv (4 attention heads) improved mean AUC from 0.7967 → 0.8284. Attention heads allow the model to weight neighbour atoms differently rather than averaging uniformly, which better captures the relevance of specific chemical environments.

**Why XGBoost underperformed:** sequential boosting struggles with high-dimensional sparse binary inputs (2048 mostly-zero fingerprint bits). Random Forest's independent averaging handles this format better — consistent with published benchmarks.

### Per-target results (Multi-Task GAT)

| Target | AUC-ROC |
|---|---|
| NR-AR | 0.7739 |
| NR-AR-LBD | 0.8914 |
| NR-AhR | 0.8587 |
| NR-Aromatase | 0.8542 |
| NR-ER | 0.6714 ⚠️ |
| NR-ER-LBD | 0.7355 |
| NR-PPAR-gamma | 0.8824 |
| SR-ARE | 0.8242 |
| SR-ATAD5 | 0.8695 |
| SR-HSE | 0.8212 |
| SR-MMP | 0.9108 |
| SR-p53 | 0.8475 |
| **Mean** | **0.8284** |

⚠️ NR-ER (Estrogen Receptor) is the weakest target at 0.67 AUC. This is consistent with published literature — estrogen receptor biology is notably complex and NR-ER is a known difficult benchmark in cheminformatics. The API flags this explicitly in every response.

Published multi-task Tox21 benchmarks typically report 0.80–0.84 mean AUC with more complex architectures. Our result of 0.8284 is competitive and sits at the upper end of that range.

---

## White-box Explainability

The `/explain` endpoint provides per-target structural explanations using **Gradient × Input saliency** — a more causally grounded method than attention weights.

### How it works

For each flagged target:
1. A backward pass through the GAT computes gradients of that target's predicted probability with respect to node features
2. `saliency = |gradient × input|` summed across feature dimensions gives one importance score per atom
3. Top-N atoms are matched against a SMARTS functional group library (~35 named groups: sulfonamides, nitro groups, aryl halides, ring systems, toxicophores, etc.)
4. All findings are batched into a single Groq API call → Llama 4 Scout writes a structured cheminformatics report

### Why Gradient × Input rather than attention?

GAT attention weights indicate *where the model looked* during message passing, but attention ≠ importance in a causal sense. Gradient × Input is a published saliency method that directly measures how much each atom feature influences the output score — a more defensible signal for scientific reporting.

### Example output for Dasatinib

```
NR-AR-LBD (52.6%): aromatic nitrogen [saliency 1.40], sulfonamide (-SO₂NH₂) [0.76]
NR-AhR    (80.4%): aromatic nitrogen [0.54], benzene ring [0.46]
SR-p53    (81.0%): aromatic nitrogen [0.41], benzene ring [0.35]
```

Note how SR-p53 and NR-AR-LBD highlight *different* atoms despite being the same molecule — SR-p53 focuses on the pyrimidine linker while NR-AR-LBD additionally implicates the sulfonamide group.

The LLM narrative then explains the mechanisms:
> *"The pyrimidine ring receives highest attention across SR-p53, NR-AhR, and SR-ATAD5. Pyrimidine-containing scaffolds are well-documented AhR ligands... The sulfonamide group (-SO₂NH₂) is additionally flagged for NR-AR-LBD — sulfonamides are known to interact with androgen receptor binding domains... Recommendation: consider replacing the sulfonamide with a methylsulfone bioisostere (-SO₂CH₃) to reduce AR-LBD binding while preserving the pharmacophore."*

---

## API

Built with FastAPI. Run locally and explore at `http://localhost:8000/docs`.

### `POST /predict`

```json
{
  "input": "caffeine",
  "input_type": "auto"
}
```

`input_type`: `"auto"` | `"name"` | `"smiles"`

**Response includes:** `resolved_smiles`, `molecule_image` (base64 PNG), `num_atoms`, `num_bonds`, `predictions` (per-target: probability, toxic, confidence, description, model AUC), `note`

### `POST /explain`

```json
{
  "input": "dasatinib",
  "input_type": "name",
  "top_n_atoms": 6
}
```

**Response includes:**
- `highlighted_molecule_image` — molecule with top saliency atoms highlighted in amber
- `atom_saliency_scores` — per-atom importance scores (merged max across all targets)
- `explanations` — per-target: probability, top saliency atoms, matched functional groups with scores
- `all_functional_groups` — full SMARTS inventory of detected groups
- `narrative` — LLM-generated cheminformatics report
- `method_note` — honest description of the explainability method and its limitations

### `GET /health`

```json
{"status": "ok", "model": "Tox21GAT", "targets": 12}
```

---

## Input handling

- **Chemical name** (`"caffeine"`) → PubChem lookup → SMILES → RDKit validation → prediction
- **SMILES string** → RDKit validation → prediction
- **Auto-detect** → tries SMILES parse first; falls back to PubChem name lookup

Invalid SMILES or unknown names return clear HTTP 400/404 errors rather than silent failures.

---

## Project structure

```
mol-property-pred/
├── app.py                        # FastAPI app — /predict, /explain, /health
├── schemas.py                    # Pydantic request models
├── config.py                     # Environment config (model path, device, API keys)
│
├── models/
│   ├── gat.py                    # Tox21GAT architecture + gradient saliency
│   └── predictor.py              # Model loading + inference wrapper
│
├── chemistry/
│   ├── functional_groups.py      # SMARTS library (~35 named groups)
│   ├── rdkit_utils.py            # Graph construction, molecule imaging
│   ├── input_processor.py        # Name/SMILES resolution
│   └── resolver.py               # PubChem lookup
│
├── explainability/
│   ├── attention.py              # Attention aggregation utilities
│   └── report_builder.py        # Saliency → explanations → /explain response
│
├── LLM/
│   ├── client.py                 # Groq API client
│   ├── prompts.py                # Cheminformatics report prompt
│   └── report.py                 # Prompt → narrative pipeline
│
├── static/
│   └── index.html                # Dark-themed frontend
│
├── saved_models/
│   └── best_multitask_gnn.pt     # Trained GAT weights (0.8284 mean AUC)
│
├── 04_multitask_gnn.ipynb        # Model training notebook
├── results.json                  # Per-target AUC results
├── tox21.csv                     # Tox21 dataset
└── requirements.txt
```

---

## Running locally

```bash
git clone https://github.com/Rjmaxruby/mol-property-pred.git
cd mol-property-pred
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

Add a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key_here
```

Then start the server:
```bash
# Terminal 1 — API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — frontend
python -m http.server 8080 --directory static
```

Open `http://localhost:8080` for the UI or `http://localhost:8000/docs` for the interactive API explorer.

To retrain the model, open `04_multitask_gnn.ipynb` and run all cells. Training takes ~30–40 minutes on CPU.

---

## Stack

Python · PyTorch · PyTorch Geometric · RDKit · scikit-learn · XGBoost · SHAP · FastAPI · Groq (Llama 4 Scout) · PubChemPy · pandas · matplotlib
