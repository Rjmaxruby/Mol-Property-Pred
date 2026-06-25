"""
app.py  —  Molecular Toxicity Prediction API
Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn as nn
import pubchempy as pcp
import base64
from io import BytesIO
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
    "NR-AR": 0.7739, "NR-AR-LBD": 0.8914, "NR-AhR": 0.8587,
    "NR-Aromatase": 0.8542, "NR-ER": 0.6714, "NR-ER-LBD": 0.7355,
    "NR-PPAR-gamma": 0.8824, "SR-ARE": 0.8242, "SR-ATAD5": 0.8695,
    "SR-HSE": 0.8212, "SR-MMP": 0.9108, "SR-p53": 0.8475,
}

class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_tasks=12, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.2)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.2)
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

model = MultiTaskGNN(input_dim=6, hidden_dim=128, num_tasks=12, heads=4)
model.load_state_dict(torch.load("best_multitask_gnn.pt", map_location="cpu"))
model.eval()
print("Model loaded successfully")

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

def resolve_name_to_smiles(name: str):
    try:
        results = pcp.get_compounds(name, "name")
        if results:
            return results[0].isomeric_smiles
    except Exception:
        pass
    return None

def resolve_smiles_to_name(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            smiles = Chem.MolToSmiles(mol)
        results = pcp.get_compounds(smiles, "smiles")
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

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Molecular Toxicity Predictor")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://140.245.201.251:8080", "http://localhost:8080"],    
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    input: str
    input_type: str = "auto"

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, req: PredictRequest):
    smiles = None
    input_was_smiles = False

    if req.input_type == "smiles":
        smiles = req.input
        input_was_smiles = True
    elif req.input_type == "name":
        smiles = resolve_name_to_smiles(req.input)
        if not smiles:
            raise HTTPException(404, f"Could not find '{req.input}' in PubChem.")
    else:
        # Auto-detect: try SMILES first, then compound name
        chemistry_chars = set("=#@+\\/%[]")
        if any(c in req.input for c in chemistry_chars):
            # Likely SMILES due to chemistry characters
            smiles = req.input
            input_was_smiles = True
        else:
            # Try as SMILES first
            mol_test = Chem.MolFromSmiles(req.input)
            if mol_test is not None:
                smiles = req.input
                input_was_smiles = True
            else:
                # Try as compound name
                smiles = resolve_name_to_smiles(req.input)
                if not smiles:
                    # If both fail, assume it was meant to be a name
                    raise HTTPException(400, f"'{req.input}' is not valid SMILES and not found in PubChem. Try specifying 'name' or 'SMILES' explicitly.")

    graph, mol = smiles_to_graph(smiles)
    if graph is None:
        raise HTTPException(400, f"Invalid or unparseable SMILES: '{smiles}'")

    common_name = None
    if input_was_smiles:
        common_name = resolve_smiles_to_name(smiles)

    resolved_label = f"{smiles} ({common_name})" if common_name else smiles

    with torch.no_grad():
        out = torch.sigmoid(model(graph))
        probs = out[0].tolist()

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
        "note": "NR-ER predictions are less reliable (model AUC: 0.67). All others above 0.73."
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "MultiTaskGAT", "targets": len(TARGETS)}