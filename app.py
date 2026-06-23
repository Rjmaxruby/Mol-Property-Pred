"""
app.py  —  Molecular Toxicity Prediction API
Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import pubchempy as pcp
import base64
from io import BytesIO

# ── Target metadata ──────────────────────────────────────────────
TARGETS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

TARGET_DESCRIPTIONS = {
    "NR-AR":         "Androgen Receptor — involved in testosterone signalling",
    "NR-AR-LBD":     "Androgen Receptor Ligand Binding Domain",
    "NR-AhR":        "Aryl Hydrocarbon Receptor — regulates response to environmental toxins",
    "NR-Aromatase":  "Aromatase Enzyme — involved in estrogen biosynthesis",
    "NR-ER":         "Estrogen Receptor — involved in estrogen signalling",
    "NR-ER-LBD":     "Estrogen Receptor Ligand Binding Domain",
    "NR-PPAR-gamma": "PPAR-gamma — regulates fatty acid storage and glucose metabolism",
    "SR-ARE":        "Antioxidant Response Element — oxidative stress pathway",
    "SR-ATAD5":      "DNA Damage Response pathway",
    "SR-HSE":        "Heat Shock Response Element — cellular stress response",
    "SR-MMP":        "Mitochondrial Membrane Potential — cell death pathway",
    "SR-p53":        "p53 Tumour Suppressor pathway",
}

TARGET_AUCS = {
    "NR-AR": 0.7887, "NR-AR-LBD": 0.8379, "NR-AhR": 0.8344,
    "NR-Aromatase": 0.7862, "NR-ER": 0.6608, "NR-ER-LBD": 0.7596,
    "NR-PPAR-gamma": 0.7906, "SR-ARE": 0.7845, "SR-ATAD5": 0.8483,
    "SR-HSE": 0.7592, "SR-MMP": 0.8776, "SR-p53": 0.8328,
}

# ── Model definition (must match training) ───────────────────────
class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_tasks=12):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_tasks)
        ])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=0.2, train=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.dropout(x, p=0.2, train=self.training)
        x = torch.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.shared(x)
        return torch.cat([head(x) for head in self.task_heads], dim=1)

# ── Load model ───────────────────────────────────────────────────
model = MultiTaskGNN()
model.load_state_dict(torch.load("best_multitask_gnn.pt", map_location="cpu"))
model.eval()
print("Model loaded successfully")

# ── Molecule utilities ───────────────────────────────────────────
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalNumHs(),
    ]

def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if not edges:
        return None, None
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, batch=torch.zeros(x.shape[0], dtype=torch.long)), mol

def mol_to_image_base64(mol) -> str:
    img = Draw.MolToImage(mol, size=(300, 300))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def resolve_name_to_smiles(name: str) -> Optional[str]:
    """Chemical name → SMILES via PubChem."""
    try:
        results = pcp.get_compounds(name, "name")
        if results:
            return results[0].isomeric_smiles
    except Exception:
        pass
    return None

from rdkit.Chem.inchi import MolToInchiKey

def resolve_smiles_to_name(smiles: str) -> Optional[str]:
    """SMILES → common name via PubChem InChIKey lookup."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchikey = MolToInchiKey(mol)
        if not inchikey:
            return None
        results = pcp.get_compounds(inchikey, "inchikey")
        if results:
            compound = results[0]
            if compound.synonyms:
                return compound.synonyms[0]
            if compound.iupac_name:
                return compound.iupac_name
    except Exception:
        pass
    return None

def confidence(prob: float) -> str:
    if prob < 0.3 or prob > 0.7:
        return "high"
    elif prob < 0.4 or prob > 0.6:
        return "medium"
    return "low"

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(title="Molecular Toxicity Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    input: str
    input_type: str = "auto"  # "smiles", "name", or "auto"

@app.post("/predict")
async def predict(req: PredictRequest):
    smiles = None
    input_was_smiles = False

    # ── Resolve input to SMILES ──────────────────────────────────
    if req.input_type == "smiles":
        smiles = req.input
        input_was_smiles = True
    elif req.input_type == "name":
        smiles = resolve_name_to_smiles(req.input)
        if not smiles:
            raise HTTPException(404, f"Could not find '{req.input}' in PubChem.")
    else:
        # Auto-detect: SMILES tend to have chemistry characters
        chemistry_chars = set("=#@+\\/%[]")
        if any(c in req.input for c in chemistry_chars) or req.input[0].isupper():
            smiles = req.input
            input_was_smiles = True
        else:
            smiles = resolve_name_to_smiles(req.input)
            if not smiles:
                smiles = req.input  # fall back to treating as SMILES
                input_was_smiles = True

    # ── Validate SMILES ──────────────────────────────────────────
    graph, mol = smiles_to_graph(smiles)
    if graph is None:
        raise HTTPException(400, f"Invalid or unparseable SMILES: '{smiles}'")

    # ── Reverse lookup: get common name if input was SMILES ──────
    common_name = None
    if input_was_smiles:
        common_name = resolve_smiles_to_name(smiles)

    resolved_label = f"{smiles} ({common_name})" if common_name else smiles

    # ── Predict ──────────────────────────────────────────────────
    with torch.no_grad():
        out = torch.sigmoid(model(graph))
        probs = out[0].tolist()

    # ── Build response ───────────────────────────────────────────
    predictions = {}
    for t, prob in zip(TARGETS, probs):
        predictions[t] = {
            "probability": round(prob, 4),
            "toxic": prob >= 0.5,
            "confidence": confidence(prob),
            "description": TARGET_DESCRIPTIONS[t],
            "model_auc": TARGET_AUCS[t],
        }

    return {
        "input": req.input,
        "resolved_smiles": resolved_label,
        "valid": True,
        "molecule_image": mol_to_image_base64(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "predictions": predictions,
        "note": "NR-ER predictions are less reliable (model AUC: 0.66). All others above 0.75."
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "MultiTaskGNN", "targets": len(TARGETS)}