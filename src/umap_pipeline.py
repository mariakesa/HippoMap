'''
https://pmc.ncbi.nlm.nih.gov/articles/instance/10659301/bin/media-1.pdf
Low dimensional manifold visualization with UMAP (unsupervised). Neural data were first
preprocessed before dimensionality reduction. Neural spiking data (spike count) during maze
learning was binned into 100ms bin. The data was then smoothed using a 500 ms wide Gaussian
kernel. The nonlinear dimensionality reduction algorithm UMAP was then applied to this matrix.
Each point in the low dimensional manifold corresponds to the population activity at a single time
bin in the session, and collectively the cloud of points maps out the animal’s navigational
trajectories during the task. The code is available at
https://github.com/lmcinnes/umap/blob/master/doc/how_umap_works.rst.
UMAP hyperparameters: n_neighbors = 20, metric = 'cosine', output_metric = 'euclidean',
learning_rate = 1.0, init = 'spectral', min_dist = 0.1, spread = 1.0,repulsion_strength = 1.0,
negative_sample_rate = 5, target_metric = 'categorical', dens_lambda = 2.0, dens_frac = 0.3,
dens_var_shift=0.1.
'''

import numpy as np
import scipy.ndimage as ndimage
import umap
import matplotlib.pyplot as plt

class UMAPPipeline:
    """
    Preprocessing pipeline for unsupervised UMAP analysis.
    """
    def __init__(self, spike_data, bin_size=0.1, smooth_window=3):
        """
        Args:
            spike_data (numpy array): Raw spike count data (neurons x time bins)
            bin_size (float): Binning window size in seconds
            smooth_window (float): Smoothing window size in seconds
        """
        self.start_time = 7845.984  # Adjusting for non-zero starting time
        self.end_time = 10023.71196667
        self.spike_data = [
            neuron_spikes[(neuron_spikes > self.start_time) & (neuron_spikes < self.end_time)]
            for neuron_spikes in spike_data['spike_time'][0]
        ]
        self.bin_size = bin_size
        self.smooth_window = smooth_window
        self.smoothed_spikes = None
        self.embedding = None

    def bin_spike_data(self):
        """Bin spike train data into time bins considering non-zero start time."""
        num_bins = int((self.end_time - self.start_time) / self.bin_size)
        binned_spikes = np.zeros((len(self.spike_data), num_bins + 1))
        bin_edges = np.arange(self.start_time, self.end_time + self.bin_size, self.bin_size)
        
        for i, neuron_spikes in enumerate(self.spike_data):
            if len(neuron_spikes) > 0:  # Ensure there are spikes before binning
                hist_count, _ = np.histogram(neuron_spikes, bins=bin_edges)
                binned_spikes[i, :len(hist_count)] = hist_count
        
        self.binned_spikes = binned_spikes
        return self
    
    def smooth_spike_data(self):
        """Apply Gaussian smoothing to the binned spike count data."""
        #smooth_bins = int(self.smooth_window / self.bin_size)
        self.smoothed_spikes = ndimage.gaussian_filter1d(self.binned_spikes, sigma=self.smooth_window, axis=1, mode='nearest')
        #self.smoothed_spikes= ndimage.gaussian_filter(self.binned_spikes, self.smooth_window,axes=0,mode='nearest',radius=self.smooth_window)
        print(self.smoothed_spikes.shape)
        return self
    
    def apply_umap(self, n_components=3):
        """Apply UMAP dimensionality reduction."""
        reducer=umap.umap_.UMAP(n_neighbors=20, n_components=6, metric='cosine', metric_kwds=None,
                              output_metric='euclidean',
                              output_metric_kwds=None, n_epochs=2000, learning_rate=1.0, init='spectral',
                              min_dist=0.1, spread=1.0, low_memory=True,
                              n_jobs=-1, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0,
                              negative_sample_rate=5, transform_queue_size=4.0,
                              a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1,
                              target_metric='categorical', target_metric_kwds=None, target_weight=0.5,
                              transform_seed=42, transform_mode='embedding', force_approximation_algorithm=False,
                              verbose=False, tqdm_kwds=None, unique=False, densmap=False, dens_lambda=2.0,
                              dens_frac=0.3, dens_var_shift=0.1, output_dens=False, disconnection_distance=None,
                              precomputed_knn=(None, None, None))
        self.embedding = reducer.fit_transform(self.smoothed_spikes.T)
        return self
    
    def visualize_umap_embedding(self):
        """Visualize the UMAP embedding in 3D with trajectory colored by time."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a time-based colormap
        time_colors = np.linspace(0, 1, self.embedding.shape[0])
        scatter = ax.scatter(
            self.embedding[:, 0], self.embedding[:, 1], self.embedding[:, 2], 
            c=time_colors, cmap='viridis', s=5, alpha=0.7
        )
        
        fig.colorbar(scatter, ax=ax, label="Time")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title("3D UMAP Embedding Colored by Time")
        plt.show()
    
    def preprocess_and_fit(self, n_components=3):
        """Run all preprocessing steps."""
        return (self
                .bin_spike_data()
                .smooth_spike_data()
                .apply_umap(n_components)
                .visualize_umap_embedding()
                .embedding)