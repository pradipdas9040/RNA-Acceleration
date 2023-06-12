from scanpy.plotting import paga_compare, rank_genes_groups

from .gridspec import gridspec
from .heatmap import heatmap
from .paga import paga
from .proportions import proportions
from .scatter import diffmap, draw_graph, pca, phate, scatter, tsne, umap
from .simulation import simulation
from .summary import summary
from .utils import hist, plot
from .acc import acc
from .acc_embedding import acc_embedding
from .acc_embedding_grid import acc_embedding_grid
from .acc_embedding_stream import acc_embedding_stream
from .acc_graph import acc_graph

__all__ = [
    "diffmap",
    "draw_graph",
    "gridspec",
    "heatmap",
    "hist",
    "paga",
    "paga_compare",
    "pca",
    "phate",
    "plot",
    "proportions",
    "rank_genes_groups",
    "scatter",
    "simulation",
    "summary",
    "tsne",
    "umap",
    "acc",
    "acc_embedding",
    "acc_embedding_grid",
    "acc_embedding_stream",
    "acc_graph",
]