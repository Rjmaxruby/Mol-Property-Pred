from fastapi import HTTPException
from rdkit import Chem

from chemistry.resolver import (
    resolve_name_to_smiles,
    resolve_smiles_to_name,
)


def process_input(req):

    smiles = None
    input_was_smiles = False

    if req.input_type == "smiles":
        smiles = req.input
        input_was_smiles = True

    elif req.input_type == "name":
        smiles = resolve_name_to_smiles(req.input)

        if not smiles:
            raise HTTPException(
                404,
                f"Could not find '{req.input}' in PubChem."
            )

    else:
        chemistry_chars = set("=#@+\\/%[]")

        if any(c in req.input for c in chemistry_chars):
            smiles = req.input
            input_was_smiles = True

        else:
            mol_test = Chem.MolFromSmiles(req.input)

            if mol_test is not None:
                smiles = req.input
                input_was_smiles = True
            else:
                smiles = resolve_name_to_smiles(req.input)

                if not smiles:
                    raise HTTPException(
                        400,
                        f"'{req.input}' is not valid SMILES "
                        "and not found in PubChem."
                    )

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise HTTPException(
            400,
            f"Invalid SMILES: '{smiles}'"
        )

    common_name = (
        resolve_smiles_to_name(smiles)
        if input_was_smiles
        else req.input
    )

    return smiles, mol, common_name, input_was_smiles