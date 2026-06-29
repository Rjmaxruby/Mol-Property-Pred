"""
app.py  —  Molecular Toxicity Prediction API
Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from schemas import PredictRequest, ExplainRequest

from chemistry.input_processor import process_input
from chemistry.rdkit_utils import (
    smiles_to_graph,
    mol_to_image_base64,
)

from explainability.report_builder import (
    build_prediction_dict,
    build_explanation,
)

from models.predictor import predictor
from models.gat import TARGETS


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

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, req: PredictRequest):

    # Shared input processing
    smiles, mol, common_name, input_was_smiles = process_input(req)

    # Build graph once
    graph, _ = smiles_to_graph(smiles)

    # Run prediction
    probs = predictor.predict_graph(graph)

    # Format predictions
    predictions = build_prediction_dict(probs)

    resolved_label = (
        f"{smiles} ({common_name})"
        if common_name else smiles
    )

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
    return {
        "status": "ok",
        "model": "Tox21GAT",
        "targets": len(TARGETS),
    }
 

# In app.py, replace the /explain endpoint body with this:

@app.post("/explain")
@limiter.limit("5/minute")
async def explain(request: Request, req: ExplainRequest):

    smiles, mol, common_name, input_was_smiles = process_input(req)

    graph, _ = smiles_to_graph(smiles)

    # Changed: predict_with_attention_graph → predict_with_saliency_graph
    probs, saliency_maps = predictor.predict_with_saliency_graph(
        graph,
        target_names=req.targets,  # None = auto-explain all flagged targets
    )

    explanation = build_explanation(
        mol=mol,
        smiles=smiles,
        common_name=common_name,
        probabilities=probs,
        saliency_maps=saliency_maps,   # Changed: atom_scores → saliency_maps
        top_n_atoms=req.top_n_atoms,
        targets=req.targets,
    )

    return explanation