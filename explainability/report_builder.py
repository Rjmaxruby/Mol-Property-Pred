from models.gat import (
    TARGETS,
    TARGET_AUCS,
    TARGET_DESCRIPTIONS,
)

from chemistry.functional_groups import identify_functional_groups
from chemistry.rdkit_utils import (
    mol_to_image_base64,
    mol_to_highlighted_base64,
)

from LLM.report import call_groq_narrative


def confidence(prob: float) -> str:
    if prob < 0.3 or prob > 0.7:
        return "high"
    elif prob < 0.4 or prob > 0.6:
        return "medium"
    return "low"


def build_prediction_dict(probabilities):
    """Formats prediction probabilities into the /predict response structure."""
    return {
        target: {
            "probability": round(prob, 4),
            "toxic": prob >= 0.5,
            "confidence": confidence(prob),
            "description": TARGET_DESCRIPTIONS[target],
            "model_auc": TARGET_AUCS[target],
        }
        for target, prob in zip(TARGETS, probabilities)
    }


def _top_atoms_from_saliency(saliency_scores, top_n):
    """
    Returns (top_atoms list, top_atoms_set) sorted by saliency descending.
    """
    sorted_atoms = sorted(
        range(len(saliency_scores)),
        key=lambda i: saliency_scores[i],
        reverse=True,
    )
    top_atoms = sorted_atoms[:top_n]
    return top_atoms, set(top_atoms)


def _match_groups_to_atoms(atom_scores, top_atoms_set, functional_groups):
    """
    Matches high-saliency atoms to SMARTS functional groups.
    Returns list of matched group dicts sorted by saliency score desc.
    """
    matched = []

    for group in functional_groups:
        overlap = set(group["atoms"]) & top_atoms_set
        if not overlap:
            continue

        group_score = sum(atom_scores[a] for a in overlap) / len(overlap)

        matched.append({
            "name": group["name"],
            "smarts": group["smarts"],
            "atoms": group["atoms"],
            "saliency_score": round(group_score, 4),
            "overlap_atoms": sorted(list(overlap)),
        })

    matched.sort(key=lambda g: g["saliency_score"], reverse=True)
    return matched


def build_target_explanations(
    probabilities,
    saliency_maps,
    functional_groups,
    top_n_atoms=6,
    targets=None,
):
    """
    Builds per-target explanation objects.

    Key difference from attention version:
      - Each target gets its OWN saliency scores → its own top atoms
      - Highlighted atoms differ per target (SR-p53 ≠ NR-AhR)
      - saliency_maps: dict {target_name: list[float]}
    """
    explanations = []
    findings_for_groq = []

    selected = targets if targets else TARGETS

    for i, target in enumerate(TARGETS):
        if target not in selected:
            continue

        prob = probabilities[i]
        toxic = prob >= 0.5

        # Use this target's own saliency map if available,
        # otherwise fall back to uniform zeros (target was below threshold)
        if target in saliency_maps:
            scores = saliency_maps[target]
            top_atoms, top_atoms_set = _top_atoms_from_saliency(scores, top_n_atoms)
            matched_groups = _match_groups_to_atoms(scores, top_atoms_set, functional_groups)
        else:
            top_atoms = []
            top_atoms_set = set()
            matched_groups = []

        explanations.append({
            "target": target,
            "description": TARGET_DESCRIPTIONS[target],
            "probability": round(prob, 4),
            "toxic": toxic,
            "confidence": confidence(prob),
            "model_auc": TARGET_AUCS[target],
            "top_saliency_atoms": top_atoms,        # renamed from top_attended_atoms
            "matched_functional_groups": matched_groups,
        })

        if prob >= 0.35:
            findings_for_groq.append({
                "target": target,
                "description": TARGET_DESCRIPTIONS[target],
                "probability": prob,
                "toxic": toxic,
                "confidence": confidence(prob),
                "top_atoms": top_atoms,
                "matched_groups": matched_groups,
            })

    return explanations, findings_for_groq


def build_explanation(
    mol,
    smiles,
    common_name,
    probabilities,
    saliency_maps,           # dict {target_name: list[float]}  ← replaces atom_scores
    top_n_atoms=6,
    targets=None,
):
    """
    Builds the complete /explain response.

    saliency_maps is now a dict keyed by target name, so each target
    contributes its own set of highlighted atoms to the response.
    The molecule highlight image uses the UNION of all flagged targets' top atoms.
    """
    num_atoms = mol.GetNumAtoms()
    functional_groups = identify_functional_groups(mol)

    explanations, findings_for_groq = build_target_explanations(
        probabilities=probabilities,
        saliency_maps=saliency_maps,
        functional_groups=functional_groups,
        top_n_atoms=top_n_atoms,
        targets=targets,
    )

    # Highlighted image = union of all flagged targets' top atoms
    all_important_atoms = set()
    for exp in explanations:
        if exp["probability"] >= 0.35:
            all_important_atoms.update(exp["top_saliency_atoms"])

    highlighted_img, img_format = mol_to_highlighted_base64(
        mol, all_important_atoms
    )

    # Atom score map: use the highest saliency score across all targets per atom
    # so the frontend has a single merged view if needed
    merged_scores = [0.0] * num_atoms
    for scores in saliency_maps.values():
        for i, s in enumerate(scores[:num_atoms]):
            if s > merged_scores[i]:
                merged_scores[i] = s

    atom_score_map = {
        str(i): {
            "saliency_score": round(merged_scores[i], 4),
            "symbol": mol.GetAtomWithIdx(i).GetSymbol(),
            "in_top_n": i in all_important_atoms,
        }
        for i in range(num_atoms)
    }

    # Groq narrative
    if findings_for_groq:
        try:
            narrative = call_groq_narrative(smiles, common_name, findings_for_groq)
            groq_error = None
        except Exception as e:
            narrative = f"Groq Error: {repr(e)}"
            groq_error = repr(e)
    else:
        narrative = (
            f"No targets flagged above the 0.35 threshold for "
            f"{common_name or smiles}. "
            f"The molecule appears low-risk across all 12 Tox21 endpoints."
        )
        groq_error = None

    return {
        "resolved_smiles": smiles,
        "common_name": common_name,
        "num_atoms": num_atoms,
        "num_bonds": mol.GetNumBonds(),
        "plain_molecule_image": mol_to_image_base64(mol),
        "highlighted_molecule_image": highlighted_img,
        "highlighted_molecule_format": img_format,
        "all_functional_groups": functional_groups,
        "atom_saliency_scores": atom_score_map,       # renamed from atom_attention_scores
        "explanations": explanations,
        "narrative": narrative,
        "groq_error": groq_error,
        "method_note": (
            "Atom importance derived from Gradient × Input saliency, "
            "computed per target via a backward pass through the GAT model. "
            "Each target's highlighted atoms reflect which atom features "
            "most influenced that specific prediction. "
            "This is more causally grounded than attention weights."
        ),
    }