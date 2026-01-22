from read_ligand_smiles_file import ReadLigandSmilesTask
from ligate_homoleptic_iridium import LigateHomolepticIrTask
from molblock_to_octahedral_geometries import MolblockToOctahedralGeometriesTask
from make_ase_tasks import make_energy_task, make_optimization_task
from setup_functions import tblite_singlet_setup, tblite_triplet_setup
from dplutils.pipeline import PipelineGraph
from dplutils.cli import get_argparser, cli_run
from dplutils.pipeline.ray import RayStreamGraphExecutor

import ray

ap = get_argparser()
ap.set_defaults(file='run.yaml')
args = ap.parse_args()

ray.init()

graph = PipelineGraph([
    ReadLigandSmilesTask,
    LigateHomolepticIrTask,
    MolblockToOctahedralGeometriesTask,
    make_optimization_task(tblite_triplet_setup, 'octahedral_embed_xyz', 'tblite_triplet_optimized_xyz'),
    make_energy_task(tblite_singlet_setup, 'tblite_triplet_optimized_xyz', 'tblite_singlet_energy'),
    make_energy_task(tblite_triplet_setup, 'tblite_triplet_optimized_xyz', 'tblite_triplet_energy')
    ])

executor = RayStreamGraphExecutor(graph, max_batches=1)

cli_run(executor, args)
