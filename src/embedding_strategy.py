import numpy as np
import umap
from sklearn.decomposition import PCA

class DimensionalityReductionStrategy:
    """
    Strategy Pattern for different dimensionality reduction techniques.
    """
    def fit_transform(self, data):
        raise NotImplementedError

class PCA_DimensionalityReduction(DimensionalityReductionStrategy):
    def fit_transform(self, data):
        return PCA(n_components=3).fit_transform(data)

class UMAP_DimensionalityReduction(DimensionalityReductionStrategy):
    def fit_transform(self, data):
        return umap.umap_.UMAP(n_neighbors=20, n_components=3, metric='cosine', metric_kwds=None,
                              output_metric='euclidean',
                              output_metric_kwds=None, n_epochs=100, learning_rate=1.0, init='spectral',
                              min_dist=0.1, spread=1.0, low_memory=True,
                              n_jobs=-1, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0,
                              negative_sample_rate=5, transform_queue_size=4.0,
                              a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1,
                              target_metric='categorical', target_metric_kwds=None, target_weight=0.5,
                              transform_seed=42, transform_mode='embedding', force_approximation_algorithm=False,
                              verbose=False, tqdm_kwds=None, unique=False, densmap=False, dens_lambda=2.0,
                              dens_frac=0.3, dens_var_shift=0.1, output_dens=False, disconnection_distance=None,
                              precomputed_knn=(None, None, None)).fit_transform(data)

class DimensionalityReducer:
    def __init__(self, strategy: DimensionalityReductionStrategy):
        self.strategy = strategy

    def fit_transform(self, data):
        return self.strategy.fit_transform(data)