from rdkit import Chem
import pubchempy as pcp

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
        if mol is not None:
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