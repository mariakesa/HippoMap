from data_loader import DataLoader
from preprocessing_pipeline import PreprocessingPipeline
from embedding_strategy import UMAP_DimensionalityReduction
import numpy as np

base_name='RO1_240106'
base_name='RO1_240110'
base_name='e13_26m1_210913'
#This one doesn't work: base_name='e15_13f1_220118'

data_loader = DataLoader(base_name, 1)
data = data_loader.load_data()

params = {
    'spk_bin': 0.02,  # Time bin size for spike binning (seconds)
    'sm_window': 5,  # Gaussian smoothing window size
    'speed_lim': 5,  # Minimum speed threshold for filtering
    'trial_bin': 5,  # Binning size for trials
    'pos_bin': 5,  # Position bin size
    'num_shuffle': 0,  # Number of shuffles for noise augmentation
    'sub_cell': 'all',  # Filter for specific cell types ('all', 'pyramidal', 'interneuron')
    'show_plot': False  # Whether to display plots during preprocessing
}

reducer=UMAP_DimensionalityReduction()
pipeline = PreprocessingPipeline(data, params, 1, reducer)
processed_data = pipeline.preprocess()

print(processed_data.keys())

np.save('embedding.npy', processed_data['embedding'])
