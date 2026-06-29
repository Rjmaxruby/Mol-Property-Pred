from rdkit import Chem
import torch
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
import base64
from torch_geometric.data import Data


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

def mol_to_highlighted_base64(mol, highlight_atoms, size=(400, 400)):
    """
    Draws mol with amber-highlighted atoms, returns (base64_str, format).
    Uses RDKit's native PNG/SVG drawing to avoid cairosvg dependencies.
    """
    if not highlight_atoms:
        return mol_to_image_base64(mol), "png"

    color = (1.0, 0.75, 0.2)   # amber
    color_map = {idx: color for idx in highlight_atoms}

    try:
        # Try RDKit's native PNG drawer first
        drawer = rdMolDraw2D.MolDraw2DPNG(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = False
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(highlight_atoms),
            highlightAtomColors=color_map,
            highlightBonds=[],
        )
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        return base64.b64encode(png_bytes).decode(), "png"
        
    except (AttributeError, Exception):
        # Fallback to SVG if PNG drawing fails on this environment
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = False
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(highlight_atoms),
            highlightAtomColors=color_map,
            highlightBonds=[],
        )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return base64.b64encode(svg.encode()).decode(), "svg"
 