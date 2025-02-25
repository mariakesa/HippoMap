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
    def __init__(self, spike_data, duration=1000, bin_size=0.1, smooth_window=0.5):
        """
        Args:
            spike_data (numpy array): Raw spike count data (neurons x time bins)
            bin_size (float): Binning window size in seconds
            smooth_window (float): Smoothing window size in seconds
        """
        self.spike_data = spike_data['spike_time'][0]
        self.bin_size = bin_size
        self.smooth_window = smooth_window
        self.smoothed_spikes = None
        self.embedding = None
        self.duration=duration
        ##print(self.spike_data.keys())

    def bin_spike_data(self):
        """Bin spike train data into time bins."""
        num_bins = int(self.duration / self.bin_size)
        binned_spikes = np.zeros((len(self.spike_data), num_bins))
        bin_edges = np.linspace(0, self.duration, num_bins + 1)
        
        for i, neuron_spikes in enumerate(self.spike_data):
            if len(neuron_spikes) > 0:  # Ensure there are spikes before binning
                binned_spikes[i], _ = np.histogram(neuron_spikes, bins=bin_edges)
        
        self.binned_spikes = binned_spikes
        
        return self
    
    def smooth_spike_data(self):
        """Apply Gaussian smoothing to the binned spike count data."""
        smooth_bins = int(self.smooth_window / self.bin_size)
        self.smoothed_spikes = ndimage.gaussian_filter1d(self.binned_spikes, sigma=smooth_bins, axis=1, mode='nearest')
        return self
    
    def apply_umap(self, n_components=3):
        """Apply UMAP dimensionality reduction."""
        reducer = umap.UMAP(
            n_neighbors=20, metric='cosine', output_metric='euclidean',
            learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0,
            repulsion_strength=1.0, negative_sample_rate=5,
            target_metric='categorical', dens_lambda=2.0, dens_frac=0.3,
            dens_var_shift=0.1, n_components=n_components
        )
        self.embedding = reducer.fit_transform(self.smoothed_spikes.T)
        return self
    
    def visualize_umap_embedding(self):
        """Visualize the UMAP embedding in 3D."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.embedding[:, 0], self.embedding[:, 1], self.embedding[:, 2], s=5, alpha=0.7)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title("3D UMAP Embedding")
        plt.show()

        return self
    
    def preprocess_and_fit(self, n_components=3):
        """Run all preprocessing steps."""
        return (self
                .bin_spike_data()
                .smooth_spike_data()
                .apply_umap(n_components)
                .visualize_umap_embedding()
                .embedding)