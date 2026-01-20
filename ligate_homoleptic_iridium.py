import pandas as pd

from dplutils.pipeline import PipelineTask
from rdkit import Chem
from rdkit.Chem import AddHs, AllChem

from octahedral_embed import ligate


def ligate_homoleptic_iridium(df: pd.DataFrame) -> pd.DataFrame:
    """Build homoleptic Ir(L)3 complexes from ligand SMILES.

    Expected input columns:
      - ligand_identifier
      - ligand_SMILES   (ligand SMILES with dummy atoms '*' at coordination sites)

    Adds output columns:
      - complex_identifier   (e.g. "<ligand_identifier>_homoleptic")
      - complex_structure    (RDKit MolBlock / SDF block for the complex, or None)
      - ligate_error         (None on success, else an error string)

    Notes:
      - This step does NOT create fac/mer isomers. That should happen later when
        you generate 3D geometries (embedding), not during ligation.
    """
    for col in ["complex_identifier", "complex_structure", "ligate_error"]:
        if col not in df.columns:
            df[col] = None

    for i, row in df.iterrows():
        complex_id = f"{row['ligand_identifier']}_homoleptic"
        df.at[i, "complex_identifier"] = complex_id

        try:
            lig = Chem.MolFromSmiles(row["ligand_SMILES"])
            if lig is None:
                raise ValueError("MolFromSmiles failed")
            # Ligate three copies of the ligand to iridium
            # Making copies in memory instead of doing [lig] * 3 which assumes
            # ligate does not modify the object
            complex_mol = ligate([Chem.Mol(lig) for _ in range(3)])
            complex_mol = AddHs(complex_mol)
            AllChem.Compute2DCoords(complex_mol)
            complex_mol.SetProp("_Name", complex_id)

            df.at[i, "complex_structure"] = Chem.MolToMolBlock(complex_mol)
            df.at[i, "ligate_error"] = None

        except Exception as exc:
            df.at[i, "complex_structure"] = None
            df.at[i, "ligate_error"] = f"{type(exc).__name__}: {exc}"

    return df


LigateHomolepticIrTask = PipelineTask(
    "ligate_homoleptic_iridium",
    ligate_homoleptic_iridium,
    batch_size=200,
)
