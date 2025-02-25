from data_loader import DataLoader
from preprocessing_pipeline_lean import UMAPPipeline
import numpy as np

base_name='RO1_240106'
base_name='RO1_240110'
base_name='e13_26m1_210913'
#This one doesn't work: base_name='e15_13f1_220118'

data_loader = DataLoader(base_name, 1)
data = data_loader.load_data()

pipeline = UMAPPipeline(data['spike_time'])
processed_data = pipeline.preprocess_and_fit()

np.save('embedding.npy', processed_data)
