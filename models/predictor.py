import torch

from config import DEVICE, MODEL_PATH
from chemistry.rdkit_utils import smiles_to_graph
from models.gat import Tox21GAT, TARGETS


class ToxicityPredictor:

    def __init__(self):

        self.model = Tox21GAT(
            input_dim=6,
            hidden_dim=128,
            num_tasks=12,
            heads=4,
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

        self.model.load_state_dict(checkpoint)
        self.model.to(DEVICE)
        self.model.eval()

        print("Model loaded successfully")

    @torch.no_grad()
    def predict(self, smiles: str):
        """Predict directly from a SMILES string."""
        graph, _ = smiles_to_graph(smiles)
        output = self.model(graph)
        return torch.sigmoid(output)[0].tolist()

    @torch.no_grad()
    def predict_graph(self, graph):
        """Predict using an already-built PyG graph."""
        output = self.model(graph)
        return torch.sigmoid(output)[0].tolist()

    def predict_with_saliency_graph(self, graph, target_names=None):
        """
        Predict + compute per-target gradient saliency maps.

        Args:
            graph        — PyG Data object
            target_names — list of target name strings to explain,
                           e.g. ["SR-p53", "NR-AhR"].
                           If None, explains all targets flagged ≥ 0.35.

        Returns:
            probs         — list[float], all 12 probabilities
            saliency_maps — dict {target_name: list[float]} atom scores
        """

        # Step 1: fast predict to get probs (no grad needed)
        with torch.no_grad():
            raw = self.model(graph)
            probs = torch.sigmoid(raw)[0].tolist()

        # Step 2: decide which targets to explain
        if target_names:
            target_indices = [
                TARGETS.index(t) for t in target_names
                if t in TARGETS
            ]
        else:
            # Auto: explain all targets with prob >= 0.35
            threshold = 0.35
            target_indices = [
                i for i, p in enumerate(probs)
                if p >= threshold
            ]

        if not target_indices:
            return probs, {}

        # Step 3: per-target saliency (runs backward once per target)
        _, saliency_by_idx = self.model.compute_saliency(graph, target_indices)

        # Step 4: remap int keys → target name strings
        saliency_maps = {
            TARGETS[idx]: scores
            for idx, scores in saliency_by_idx.items()
        }

        return probs, saliency_maps

    # ── keep old attention methods so nothing else breaks ────────────────
    @torch.no_grad()
    def predict_with_attention(self, smiles: str):
        """Legacy — kept for compatibility."""
        graph, _ = smiles_to_graph(smiles)
        output, atom_scores = self.model.forward_with_attention(graph)
        return torch.sigmoid(output)[0].tolist(), atom_scores

    @torch.no_grad()
    def predict_with_attention_graph(self, graph):
        """Legacy — kept for compatibility."""
        output, atom_scores = self.model.forward_with_attention(graph)
        return torch.sigmoid(output)[0].tolist(), atom_scores


predictor = ToxicityPredictor()