from chemistry.functional_groups import identify_functional_groups


def get_top_attention_atoms(atom_scores, top_n):
    """
    Returns the top-N most attended atoms and a set for fast lookup.
    """

    sorted_atoms = sorted(
        range(len(atom_scores)),
        key=lambda i: atom_scores[i],
        reverse=True,
    )

    top_atoms = sorted_atoms[:top_n]

    return top_atoms, set(top_atoms)


def match_attention_to_groups(
    atom_scores,
    top_atoms_set,
    functional_groups,
):
    """
    Matches highly attended atoms to SMARTS functional groups.

    Returns:
        [
            {
                "name": "...",
                "smarts": "...",
                "atoms": [...],
                "attention_score": 0.812,
                "overlap_atoms": [...]
            },
            ...
        ]
    """

    matched_groups = []

    for group in functional_groups:

        overlap = set(group["atoms"]) & top_atoms_set

        if not overlap:
            continue

        attention = (
            sum(atom_scores[a] for a in overlap)
            / len(overlap)
        )

        matched_groups.append(
            {
                "name": group["name"],
                "smarts": group["smarts"],
                "atoms": group["atoms"],
                "attention_score": round(attention, 4),
                "overlap_atoms": sorted(list(overlap)),
            }
        )

    matched_groups.sort(
        key=lambda g: g["attention_score"],
        reverse=True,
    )

    return matched_groups


def build_atom_score_map(
    mol,
    atom_scores,
    top_atoms_set,
):
    """
    Creates a frontend-friendly atom attention dictionary.
    """

    atom_map = {}

    for i in range(mol.GetNumAtoms()):

        atom_map[str(i)] = {
            "attention_score": round(atom_scores[i], 4),
            "symbol": mol.GetAtomWithIdx(i).GetSymbol(),
            "in_top_n": i in top_atoms_set,
        }

    return atom_map


def build_attention_summary(
    mol,
    atom_scores,
    top_n_atoms,
):
    """
    Complete attention analysis for a molecule.

    Returns:
        top_atoms
        top_atoms_set
        atom_score_map
        matched_groups
    """

    top_atoms, top_atoms_set = get_top_attention_atoms(
        atom_scores,
        top_n_atoms,
    )

    functional_groups = identify_functional_groups(mol)

    matched_groups = match_attention_to_groups(
        atom_scores,
        top_atoms_set,
        functional_groups,
    )

    atom_score_map = build_atom_score_map(
        mol,
        atom_scores,
        top_atoms_set,
    )

    return (
        top_atoms,
        top_atoms_set,
        atom_score_map,
        functional_groups,
        matched_groups,
    )