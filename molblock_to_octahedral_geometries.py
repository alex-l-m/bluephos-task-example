import pandas as pd
from dplutils.pipeline import PipelineTask
from rdkit import Chem
from rdkit.Chem import AddHs, MolToXYZBlock

from octahedral_embed import octahedral_embed


ISOMERS = ("fac", "mer")


def molblock_to_octahedral_geometries(df: pd.DataFrame) -> pd.DataFrame:
    """Generate fac and mer octahedral geometries from complex MolBlocks.

    Expected input columns:
      - complex_identifier
      - complex_structure   (RDKit MolBlock / SDF block as a string)

    Output: two rows per input row (fac + mer), with added columns:
      - isomer
      - isomer_identifier          (e.g. "<complex_identifier>_fac")
      - octahedral_embed_xyz       (XYZ block string or "failed")
      - octahedral_embed_error     (None on success, else an error string)

    Assumptions:
      - complex_structure is readable by RDKit (MolFromMolBlock)
      - octahedral_embed can embed most reasonable complexes
    """
    out = []

    for _, row in df.iterrows():
        base_id = str(row["complex_identifier"])
        block = row["complex_structure"]

        mol0 = None
        base_error = None

        if block is None or pd.isna(block) or str(block) == "":
            base_error = "missing_complex_structure"
        else:
            mol0 = Chem.MolFromMolBlock(str(block), removeHs=False)
            if mol0 is None:
                base_error = "MolFromMolBlock failed"

        for iso in ISOMERS:
            new = dict(row)
            new["isomer"] = iso
            new["isomer_identifier"] = f"{base_id}_{iso}"

            if mol0 is None:
                new["octahedral_embed_xyz"] = "failed"
                new["octahedral_embed_error"] = base_error
                out.append(new)
                continue

            try:
                mol = Chem.Mol(mol0)       # copy
                mol.RemoveAllConformers()  # avoid carrying 2D coords into embedding
                mol = AddHs(mol)
                mol.SetProp("_Name", new["isomer_identifier"])

                octahedral_embed(mol, isomer=iso)

                new["octahedral_embed_xyz"] = MolToXYZBlock(mol)
                new["octahedral_embed_error"] = None

            except Exception as exc:
                new["octahedral_embed_xyz"] = "failed"
                new["octahedral_embed_error"] = f"{type(exc).__name__}: {exc}"

            out.append(new)

    return pd.DataFrame(out)


MolblockToOctahedralGeometriesTask = PipelineTask(
    "molblock_to_octahedral_geometries",
    molblock_to_octahedral_geometries,
    batch_size=50,
    num_cpus=1,
)
