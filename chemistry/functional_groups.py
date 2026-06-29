from rdkit import Chem

# ── SMARTS lookup table ───────────────────────────────────────────────────────
# More specific patterns first — RDKit tries them in order
SMARTS_GROUPS = {
    # Sulphur
    "[S](=O)(=O)[NH2]":      "sulfonamide (-SO2NH2)",
    "[S](=O)(=O)[NH]":       "sulfonamide (-SO2NH-)",
    "[S](=O)(=O)":           "sulfonyl group (-SO2-)",
    "[S](=O)":               "sulfoxide (-S=O)",
    "[SH]":                  "thiol (-SH)",
    "[s]":                   "aromatic sulfur",
    # Nitrogen
    "[N+](=O)[O-]":          "nitro group (-NO2)",
    "c[NH2]":                "aromatic amine (Ar-NH2)",
    "[NH2]":                 "primary amine (-NH2)",
    "[NH][C](=O)":           "amide (-NHCO-)",
    "[C](=O)[NH2]":          "primary amide (-CONH2)",
    "[N]=[N]":               "azo group (-N=N-)",
    "[N]=[C]=[O]":           "isocyanate (-NCO)",
    "[N]=[C]=[S]":           "isothiocyanate (-NCS)",
    "[n]":                   "aromatic nitrogen (pyridine-like)",
    "[nH]":                  "aromatic N-H (pyrrole-like)",
    # Halogens
    "c[F,Cl,Br,I]":         "aryl halide",
    "[F]":                   "fluorine substituent",
    "[Cl]":                  "chlorine substituent",
    "[Br]":                  "bromine substituent",
    "[I]":                   "iodine substituent",
    # Oxygen
    "c[OH]":                 "phenol (Ar-OH)",
    "[C](=O)[OH]":           "carboxylic acid (-COOH)",
    "[C](=O)[O][C]":         "ester (-COO-)",
    "[OH]":                  "hydroxyl group (-OH)",
    "[C](=O)":               "carbonyl (C=O)",
    "[O-]":                  "alkoxide / phenoxide anion",
    # Phosphorus
    "[P](=O)([O])([O])":     "phosphate group",
    "[P](=O)":               "phosphonyl",
    # Toxicophore alerts
    "[C]=[C][C](=O)":        "Michael acceptor (alpha,beta-unsaturated carbonyl)",
    "[CH2][Cl,Br,I]":       "alkyl halide (potential alkylating agent)",
    # Ring systems
    "c1ccccc1":              "benzene ring",
    "c1ccncc1":              "pyridine ring",
    "c1ccoc1":               "furan ring",
    "c1ccsc1":               "thiophene ring",
    "C1CCNCC1":              "piperidine ring",
    "C1COCCN1":              "morpholine ring",
    "C1CNCCN1":              "piperazine ring",
}
 
# Pre-compile for performance
_COMPILED_SMARTS = {}
for _smarts, _name in SMARTS_GROUPS.items():
    _pat = Chem.MolFromSmarts(_smarts)
    if _pat is not None:
        _COMPILED_SMARTS[_smarts] = (_pat, _name)
 
 
def identify_functional_groups(mol):
    """Returns list of {smarts, name, atoms} for every matched group in mol."""
    found = []
    seen = set()
    for smarts, (pattern, name) in _COMPILED_SMARTS.items():
        for match in mol.GetSubstructMatches(pattern):
            key = (name, frozenset(match))
            if key not in seen:
                seen.add(key)
                found.append({"smarts": smarts, "name": name, "atoms": list(match)})
    return found
 