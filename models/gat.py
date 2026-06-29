from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


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


class Tox21GAT(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=128,
        num_tasks=12,
        heads=4,
    ):
        super().__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.2)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.2)

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.task_heads = nn.ModuleList(
            [nn.Linear(64, 1) for _ in range(num_tasks)]
        )

    def forward(self, data) -> torch.Tensor:
        """Standard forward pass — used by /predict."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=0.2, train=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.dropout(x, p=0.2, train=self.training)
        x = torch.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)
        x = self.shared(x)

        return torch.cat([head(x) for head in self.task_heads], dim=1)

    def forward_for_saliency(self, data):
        x_input = data.x.clone().detach().requires_grad_(True)
        edge_index = data.edge_index
        batch = data.batch

        x = torch.relu(self.conv1(x_input, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.shared(x)
        logits = torch.cat([head(x) for head in self.task_heads], dim=1)

        return logits, x_input

    def compute_saliency(
        self,
        data,
        target_indices: List[int],
    ) -> Tuple[List[float], dict]:
        """
        Gradient × Input saliency maps, one per requested target.

        For each target:
          1. Run forward_for_saliency() to get logits + input node tensor
          2. Call sigmoid(logits[0, target_idx]).backward()
          3. Atom importance = (grad * input).abs().sum(dim=1)
             — this is the standard gradient × input saliency formula

        Args:
            data          — PyG Data object (single molecule, batch=0)
            target_indices — list of int, which of the 12 targets to explain

        Returns:
            probs          — list[float], sigmoid probabilities for all 12 targets
            saliency_maps  — dict {target_idx: list[float]} atom importance per target
        """
        self.eval()

        num_atoms = data.x.shape[0]

        # First: get probabilities with no_grad (fast, clean)
        with torch.no_grad():
            raw = self.forward(data)
            probs = torch.sigmoid(raw)[0].tolist()

        # Second: per-target saliency (needs grad)
        saliency_maps = {}

        for target_idx in target_indices:

            # Fresh forward pass for each target so gradients don't accumulate
            logits, x_input = self.forward_for_saliency(data)

            # Sigmoid of this target's logit → scalar → backward
            score = torch.sigmoid(logits[0, target_idx])
            score.backward()

            # gradient × input, summed across feature dim → [num_atoms]
            grad = x_input.grad  # [num_atoms, 6]
            saliency = (grad * x_input.detach()).abs().sum(dim=1)  # [num_atoms]

            saliency_maps[target_idx] = saliency.tolist()

        return probs, saliency_maps