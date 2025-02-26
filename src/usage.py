from data_loader import DataLoader
from umap_pipeline import UMAPPipeline
import numpy as np

base_name='RO1_240106'
base_name='RO1_240110'
base_name='e13_26m1_210913'
#This one doesn't work: base_name='e15_13f1_220118'

data_loader = DataLoader(base_name, 1)
data = data_loader.load_data()


beh_time = data['behavior']['behavior']['timestamps'][0][0]
print(beh_time)
session_epochs = data['session']['session']['epochs'][0][0]
#print(session_epochs[0][0].shape)
startTime = session_epochs[0,1]['startTime']
stopTime = session_epochs[0,1]['stopTime']
beh_ts = (beh_time>startTime) &  (beh_time<stopTime)
print(beh_ts)
print(startTime)
print(stopTime)
pipeline = UMAPPipeline(data['spike_time'])
processed_data = pipeline.preprocess_and_fit()

