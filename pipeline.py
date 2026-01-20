from read_ligand_smiles_file import ReadLigandSmilesTask
from ligate_homoleptic_iridium import LigateHomolepticIrTask
from molblock_to_octahedral_geometries import MolblockToOctahedralGeometriesTask
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
    ])

executor = RayStreamGraphExecutor(graph, max_batches=1)

cli_run(executor, args)
