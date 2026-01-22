'''Concatenate all parquet files in the working directory and save as a csv file'''

import os
from glob import glob
import pandas as pd
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolFromMolBlock
from rdkit.Chem.Draw import MolToImage

combined = pd.concat(pd.read_parquet(f) for f in glob("*.parquet"))
combined.to_csv("combined.csv", index=False)

ligand_image_outdir = 'ligand_images'
if not os.path.exists(ligand_image_outdir):
    os.mkdir(ligand_image_outdir)
structure_image_outdir = 'structure_images'
if not os.path.exists(structure_image_outdir):
    os.mkdir(structure_image_outdir)
octahedral_file_outdir = 'octahedral_embed_xyz'
if not os.path.exists(octahedral_file_outdir):
    os.mkdir(octahedral_file_outdir)
tblite_file_outdir = 'tblite_triplet_xyz'
if not os.path.exists(tblite_file_outdir):
    os.mkdir(tblite_file_outdir)
for row in combined.itertuples():
    ligand_smiles = row.ligand_SMILES
    ligand_rdkit_mol = MolFromSmiles(ligand_smiles)
    ligand_image = MolToImage(ligand_rdkit_mol, size=(300, 300))
    ligand_id = row.ligand_identifier
    ligand_image_outpath = os.path.join(ligand_image_outdir, f"{ligand_id}.png")
    ligand_image.save(ligand_image_outpath)
    
    complex_id = row.complex_identifier
    mol_block = row.complex_structure
    mol = MolFromMolBlock(mol_block)
    structure_image = MolToImage(mol, size=(300, 300))
    structure_image_outpath = os.path.join(structure_image_outdir, f"{complex_id}_structure.png")
    structure_image.save(structure_image_outpath)

    isomer_id = row.isomer_identifier
    xyz_text = row.octahedral_embed_xyz
    xyz_outpath = os.path.join(octahedral_file_outdir, f"{isomer_id}.xyz")
    with open(xyz_outpath, 'w') as f:
        f.write(xyz_text)

    xyz_text = row.tblite_triplet_optimized_xyz
    xyz_outpath = os.path.join(tblite_file_outdir, f"{isomer_id}.xyz")
    with open(xyz_outpath, 'w') as f:
        f.write(xyz_text)
