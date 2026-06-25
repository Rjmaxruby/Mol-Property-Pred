# Molecular Toxicity Predictor

A deployable AI system that predicts the toxicity of chemical compounds across 12 biological targets simultaneously — from a chemical name or SMILES string, with no lab test required.

Built as a portfolio project demonstrating applied ML for real-world scientific problems, directly relevant to pharma and biotech AI/ML roles.

**Live API:** runs locally via FastAPI with an auto-generated interactive docs UI at `/docs`

---

## What it does

Enter a chemical name like `caffeine` or a SMILES string like `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` and the system returns:

- A 2D visualisation of the molecule
- Toxicity predictions across all 12 Tox21 biological targets
- Probability scores and confidence levels per target
- Model reliability (AUC-ROC) shown per target so you know how much to trust each prediction

**Example output for caffeine:** all 12 targets non-toxic, high confidence across most.

**Example output for Dasatinib (cancer drug):** flags as toxic on NR-AhR (0.66), SR-ARE (0.56), SR-p53 (0.60), and SR-HSE (0.50) — consistent with its known activity as a potent multi-target kinase inhibitor.

---

## Why this matters

Before a drug candidate can be tested on humans, it needs to be screened for toxicity. Running lab assays on thousands of molecules is expensive and slow. This system predicts toxicity from molecular structure alone — potentially flagging dangerous compounds earlier and at a fraction of the cost.

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

### Architecture: Multi-Task Graph Neural Network

Molecules are represented as graphs — atoms as nodes, bonds as edges — rather than flat fingerprint vectors. This preserves the actual chemical structure and allows the model to learn from molecular topology directly.

**Node features per atom (6):** element number, degree, formal charge, aromaticity, ring membership, hydrogen count

**Architecture:**
- 3 × GCNConv layers (128 hidden dims) with dropout — message passing lets each atom aggregate information from its chemical neighbourhood
- Global mean pooling — converts per-atom representations into a single molecule vector
- Shared dense layer (128 → 64) with dropout
- 12 independent output heads — one per toxicity target

**Training details:**
- Per-target `pos_weight` to handle class imbalance (range: 5.1× to 33.1× depending on target)
- Masked loss function — only trains on labels that exist for each molecule (not all molecules were tested on all 12 targets)
- Early stopping (patience=10, evaluated every 5 epochs) — best checkpoint saved at epoch 295
- Adam optimizer, lr=0.001

### Comparison: Baseline vs GNN

| Model | Mean AUC-ROC | Notes |
|---|---|---|
| Random Forest (ECFP fingerprints) | 0.7535 | Fast, interpretable via SHAP |
| XGBoost (ECFP fingerprints) | 0.6965 | Underperformed on sparse binary data |
| **Multi-Task GNN** | **0.7967** | Best overall |

**Why XGBoost underperformed:** its sequential boosting strategy struggles with high-dimensional sparse binary inputs (2048 mostly-zero fingerprint bits). Random Forest's independent averaging handles this format better — consistent with published benchmarks on fingerprint-based molecular ML.

### Per-target results (Multi-Task GNN)

| Target | AUC-ROC |
|---|---|
| NR-AR | 0.7887 |
| NR-AR-LBD | 0.8379 |
| NR-AhR | 0.8344 |
| NR-Aromatase | 0.7862 |
| NR-ER | 0.6608 ⚠️ |
| NR-ER-LBD | 0.7596 |
| NR-PPAR-gamma | 0.7906 |
| SR-ARE | 0.7845 |
| SR-ATAD5 | 0.8483 |
| SR-HSE | 0.7592 |
| SR-MMP | 0.8776 |
| SR-p53 | 0.8328 |
| **Mean** | **0.7967** |

⚠️ NR-ER (Estrogen Receptor) is the weakest target at 0.66 AUC. This is consistent with published literature — estrogen receptor biology is notably complex, and NR-ER is a known difficult benchmark in cheminformatics. The API flags this explicitly in its response.

Published multi-task Tox21 benchmarks typically report 0.80–0.84 mean AUC with more complex architectures. Our result of 0.7967 is competitive given the lightweight model design.

---

## API

Built with FastAPI. Run locally and explore at `http://localhost:8000/docs`.

### `POST /predict`

**Request:**
```json
{
  "input": "caffeine",
  "input_type": "name"
}
```

Or with a SMILES string:
```json
{
  "input": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
  "input_type": "smiles"
}
```

Or let the system auto-detect:
```json
{
  "input": "caffeine",
  "input_type": "auto"
}
```

**Response includes:**
- `resolved_smiles` — the SMILES string used (useful when input was a name)
- `molecule_image` — base64-encoded PNG of the 2D molecular structure
- `num_atoms`, `num_bonds`
- `predictions` — per-target: probability, toxic (bool), confidence (low/medium/high), description, model AUC
- `note` — honest caveat about NR-ER reliability

### `GET /health`
```json
{"status": "ok", "model": "MultiTaskGNN", "targets": 12}
```

---

## Input handling

The system accepts both chemical names and SMILES strings:

- **Chemical name** (`"caffeine"`, `"aspirin"`) → PubChem lookup → SMILES → RDKit validation → prediction
- **SMILES string** → RDKit validation → prediction
- **Auto-detect** → if input contains chemistry characters (`=`, `#`, `@`, etc.) or starts with uppercase, treated as SMILES; otherwise looked up by name

Invalid SMILES or unknown names return a clear HTTP 400/404 error rather than a silent failure.

---

## Project structure

```
mol-property-pred/
├── 01_eda.ipynb              # EDA, fingerprint generation, Random Forest + XGBoost baseline, SHAP
├── 04_multitask_gnn.ipynb    # Multi-task GNN: graph construction, architecture, training
├── app.py                    # FastAPI prediction API
├── best_multitask_gnn.pt     # Saved GNN weights (best checkpoint)
├── results.json              # Per-target AUC results
├── tox21.csv                 # Local copy of Tox21 dataset
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
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` to explore the API interactively.

To retrain the model, open `04_multitask_gnn.ipynb` and run all cells. Training takes approximately 30–40 minutes on CPU (345 epochs with early stopping).

# Terminal 1 — start the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — serve the frontend
python -m http.server 8080
---

## Stack

Python · PyTorch · PyTorch Geometric · RDKit · scikit-learn · XGBoost · SHAP · FastAPI · PubChemPy · pandas · matplotlib
