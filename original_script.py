try:
    import google.colab # type: ignore
    IN_COLAB = True
except:
    IN_COLAB = False
import os, sys

if IN_COLAB:
    # Install packages
    %pip install umap-learn==0.5.3
    %pip install mat73
    %pip install plotly

import os
import argparse
from time import time

import umap

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from google.colab import files # for check file and download

import mat73
import pickle

import random
import numpy as np
import math
import pandas as pd
import csv
import statistics

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


from scipy import stats as st
from scipy.ndimage import gaussian_filter
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import NearestNDInterpolator
import scipy.io as sio
from scipy import interpolate

if IN_COLAB:

  ! git clone https://github.com/winnieyangwannan/Selection-of-experience-for-memory-by-hippocampal-sharp-wave-ripples.git

if IN_COLAB:

  %ls
  %cd /content/Selection-of-experience-for-memory-by-hippocampal-sharp-wave-ripples
  %ls
  %cd Data
  %ls
  # (1) if the github link has been updated, copy the new link (even though it looks the same)
  # (2) if folder has exist error occur, run this:
  # (3) %rm -rf /content/Selection-of-experience-for-memory-by-hippocampal-sharp-wave-ripples
  # (4) restart the session after this step

def load_behavior_data(basename,sessionID):

    # LOAD SPIK TIME DATA
    spike_fname = basename + '.spike_time.mat'
    try:
        spike_mat = sio.loadmat(spike_fname)
    except:
        spike_mat = mat73.loadmat(spike_fname)

    spike_time = spike_mat['spike_time'][0]
    # LOAD BEHAVIOR DATA
    beh_fname = basename + '.Behavior.mat'
    try:
        beh_mat = sio.loadmat(beh_fname)
    except:
        beh_mat = mat73.loadmat(beh_fname)

    beh_time = beh_mat['behavior']['timestamps'][0][0]
    pos_x =beh_mat['behavior']['position'][0][0]['x'][0][0]
    pos_y = beh_mat['behavior']['position'][0][0]['y'][0][0]
    pos_lin = beh_mat['behavior']['position'][0][0]['lin'][0][0]
    trial_data = beh_mat['behavior']['masks'][0][0]['TRIALS'][0][0]

    # LOAD SESSION INFO
    session_fname = basename + '.session.mat'
    try:
        session_mat = sio.loadmat( session_fname)
    except:
        session_mat = mat73.loadmat( session_fname)

    try:
        session_epochs = session_mat['session']['epochs'][0][0]
        startTime = session_epochs[0,sessionID-1]['startTime'][0][0][0][0]
        stopTime = session_epochs[0,sessionID-1]['stopTime'][0][0][0][0]
    except:
        session_epochs = session_mat['session']['epochs']
        startTime = session_epochs[sessionID-1]['startTime']
        stopTime = session_epochs[sessionID-1]['stopTime']

    # LOAD CELL TYPE
    cell_fname = basename + '.cell_type.mat'
    try:
        cell_mat = sio.loadmat(cell_fname)
    except:
        cell_mat = mat73.loadmat(cell_fname)

    cell_type = cell_mat['cell_type'][0]
    # pyramidal_cell = []
    # for id,ct in enumerate(cell_type):
    #     if ct == 'Pyramidal Cell':
    #         pyramidal_cell.append(id)

    data = {}
    data= {'spike_time' : spike_time,
         'beh_time':beh_time,
         'pos_x':pos_x,
         'pos_y':pos_y,
         'pos_lin':pos_lin,
         'trial_data':trial_data,
         'session_epochs':session_epochs,
         'startTime':startTime,
         'stopTime':stopTime,
         'cell_type':cell_type,
    }

    del spike_fname,session_fname, beh_fname, cell_fname
    return data

def preprocess_behavior_data(data,params):

  spike_time = data['spike_time']
  beh_time = data['beh_time']
  pos_x = data['pos_x']
  pos_y = data['pos_y']
  pos_lin = data['pos_lin']
  trial_data = data['trial_data']
  session_epochs = data['session_epochs']
  startTime = data['startTime']
  stopTime = data['stopTime']
  cell_type = data['cell_type']

  speed_lim = params['speed_lim']
  spk_bin = params['spk_bin']
  sm_window = params['sm_window']
  seed = params['seed']
  trial_bin = params['trial_bin']
  pos_bin = params['pos_bin']
  num_shuffle = params['num_shuffle']
  sub_cell  = params['sub_cell']
  show_plot = params['show_plot']
  # 3.1 Get data to the behavior epoch

  beh_ts = (beh_time>startTime) &  (beh_time<stopTime)
  ts_beh =  beh_time[beh_ts]
  x_pos = pos_x[beh_ts]
  y_pos = pos_y[beh_ts]
  lin_pos = pos_lin[beh_ts]
  trial = trial_data[beh_ts]
  epoch_ind = sessionID*np.ones_like(x_pos)

  # 3.2.SPIKE COUNT
  num_cell = spike_time.shape[0]
  spk_ts = np.arange(ts_beh[0], ts_beh[-1], spk_bin)
  spk_count = np.zeros((num_cell,spk_ts.shape[0]-1))


  for spk in range(num_cell):
      # SPIKE COUNT
      hist_count = np.histogram(spike_time[spk],bins=spk_ts)
      spk_count[spk,:] = hist_count[0]

  spk_data = np.transpose(spk_count)
  spk_ts = spk_ts[1:]
  num_cell = spk_data.shape[1]

  # 3.3 interpolate the behavior data according to spike time
  interp_func = interpolate.interp1d(ts_beh, x_pos,kind ="cubic")
  x_pos_interp = interp_func(spk_ts)

  interp_func = interpolate.interp1d(ts_beh, y_pos,kind ="cubic")
  y_pos_interp = interp_func(spk_ts)

  interp_func = interpolate.interp1d(ts_beh, lin_pos,kind ="nearest")
  lin_pos_interp = interp_func(spk_ts)

  interp_func = interpolate.interp1d(ts_beh, trial, kind = "nearest")
  trial_interp = interp_func(spk_ts)

  interp_func = interpolate.interp1d(ts_beh, epoch_ind, kind = "nearest")
  epoch_interp = interp_func(spk_ts)

  # 3.4. smooth the behavior data
  x_pos_sm =gaussian_filter(x_pos_interp, sm_window)
  y_pos_sm =gaussian_filter(y_pos_interp, sm_window)
  lin_pos_sm =gaussian_filter(lin_pos_interp, sm_window)

  # # 3.5. plot the smoothed behavior data

  fig = go.Figure()
  fig.add_trace(go.Scatter(x= ts_beh[:] , y =lin_pos[:],mode='lines',name='Position'))
  fig.update_layout(
      title="Linearized Position",
      xaxis_title="Time (s)",
      yaxis_title="Trial no.",
      # legend_title="Legend Title",
      font=dict(
          family="sans-serif",
          size=20,
          color="black"
      )
  )
  if show_plot:
    fig.show()

  # 3.6. Get the trial data into trial blocks
  bins = np.arange(min(trial_interp),max(trial_interp)+5,5)
  trial_bin = np.histogram(trial_interp,bins=bins)
  trial_bins = trial_bin[1]
  num_trial_bin = trial_bins.shape[0]
  trial_count =  trial_bin[0]
  trial_num_ds_bin = trial_interp
  for tt,tb in enumerate(trial_bins):
    if tt<num_trial_bin-1:

        trial_num_ds_bin[(trial_interp>=trial_bins[tt]) & (trial_interp<(trial_bins[tt+1]))]=tt
    else:

        trial_num_ds_bin[trial_interp>=trial_bins[tt]]=tt

  trial_num_ds_bin = np.array(trial_num_ds_bin, dtype=int)


  fig = go.Figure()
  fig.add_trace(go.Scatter(x= spk_ts[:] , y =lin_pos_sm[:],mode='lines',name='Position'))
  fig.add_trace(go.Scatter(x= spk_ts[:] , y =trial_num_ds_bin[:],mode='lines',name='Trial'))
  fig.update_layout(
      title="Trial Block",
      xaxis_title="Time (s)",
      yaxis_title="Trial",
      font=dict(
          family="sans-serif",
          size=20,
          color="black"
      )
  )
  if show_plot:
    fig.show()
  # 3.7. Get the position data into pisition bins
  bins = np.arange(min(lin_pos_sm),max(lin_pos_sm)+5,5)
  pos_bin = np.histogram(lin_pos_sm,bins=bins)
  pos_bins = pos_bin[1]
  num_pos_bin = pos_bins.shape[0]
  pos_count =  pos_bin[0]
  pos_lin_ds_bin = lin_pos_interp
  for tt,tb in enumerate(pos_bins):
    if tt<num_pos_bin-1:

        pos_lin_ds_bin[(lin_pos_sm>=pos_bins[tt]) & (lin_pos_sm<(pos_bins[tt+1]))]=tt
    else:

        pos_lin_ds_bin[lin_pos_sm>=pos_bins[tt]]=tt
  pos_lin_ds_bin = np.array(pos_lin_ds_bin, dtype=int)
  fig = go.Figure()
  fig.add_trace(go.Scatter(x= spk_ts[:] , y =lin_pos_sm[:],mode='lines',name='Position'))
  fig.add_trace(go.Scatter(x= spk_ts[:] , y =pos_lin_ds_bin[:],mode='lines',name='Position bin'))
  fig.update_layout(
      title="Positin Bin",
      xaxis_title="Time (s)",
      yaxis_title="Position bins",
      font=dict(
          family="sans-serif",
          size=20,
          color="black"
      )
  )
  if show_plot:
    fig.show()


  # 3.8 Apply speed threshold
  speed_c = np.zeros_like(lin_pos_sm)

  for i in range(len(lin_pos_sm)-1):
      speed_c[i] = abs(lin_pos_sm[i+1]-lin_pos_sm[i])
  speed_c = speed_c/np.mean(np.diff(spk_ts))

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=spk_ts,y=speed_c,name='Speed'))
  fig.add_trace(go.Scatter(x=spk_ts,y=lin_pos_sm,mode='lines',name='Position'))
  fig.add_trace(go.Scatter(x=spk_ts[speed_c>=speed_lim],y=lin_pos_sm[speed_c>=speed_lim],mode='markers',name='Position > speed_lim', marker=dict(
          size=2,
          )))
  if show_plot:
    fig.show()
  #
  speed_ts =  speed_c>=speed_lim
  pos_lin_bin = pos_lin_ds_bin[speed_ts]
  pos_x = x_pos_sm[speed_ts]
  pos_y = y_pos_sm[speed_ts]
  trial_num_bin = trial_num_ds_bin[speed_ts]
  epoch = epoch_interp[speed_ts]
  spk_data_speed = spk_data[speed_ts,:]



  # 3.9 Smooth and z score spike data
  spk_count_sm = np.zeros_like(spk_data_speed)
  spk_count_sm_z = np.zeros_like(spk_data_speed)

  spk_count_sm =gaussian_filter(spk_data_speed, sm_window,axes=0,mode='nearest',radius=sm_window)
  print(spk_count_sm.shape)
  spk_count_sm_z = stats.zscore(spk_count_sm,axis=0)
  print(spk_count_sm_z.shape)


  # plot spike data
  fig = px.imshow( np.transpose(spk_count_sm_z[0:500,:]))
  fig.update_layout(
      title="Spike Data",
      xaxis_title="Time (s)",
      yaxis_title="Cell",
      # legend_title="Legend Title",
      font=dict(
          family="sans-serif",
          size=20,
          color="black"
      )
  )
  if show_plot:
    fig.show()
  # 3.10 Add shuffle data (noise cloud)
  if num_shuffle>0 :
    shuffle_len= spk_count_sm_z.shape[0]*num_shuffle
    data_len = spk_count_sm_z.shape[0]
    data_embedding_all = spk_count_sm_z
    for i in range(num_shuffle):
      np.random.seed(i)
      embedding_shuffle = np.zeros_like(spk_count_sm_z)
      # shuffle row, for each collumn independently
      for c in range(spk_count_sm_z.shape[1]):
          shuffle_row = np.random.permutation(spk_count_sm_z.shape[0])
          embedding_shuffle[:,c] = spk_count_sm_z[shuffle_row,c]
      # shuffle column, for each row independently
      for r in range(spk_count_sm_z.shape[0]):
          shuffle_col = np.random.permutation(spk_count_sm_z.shape[1])
          embedding_shuffle[r,:] = embedding_shuffle[r,shuffle_col]
      # embedding_shuffle = np.random.permutation(spk_count_sm_z)
      data_embedding_all = np.concatenate((data_embedding_all,embedding_shuffle),axis =0)

  else:
    shuffle_len = 0
    data_len = spk_count_sm_z.shape[0]
    data_embedding_all = spk_count_sm_z

  # 3.11 Get specific cell type
  pyramidal = []
  interneuron = []

  for cc,cell in enumerate(cell_type):
    if cell == 'Pyramidal Cell':
      pyramidal.append(True)
      interneuron.append(False)

    else:
      pyramidal.append(False)
      interneuron.append(True)

  if sub_cell =='all':
    data_embedding = data_embedding_all
  elif sub_cell == 'pyramidal':
    data_embedding = data_embedding_all[:,pyramidal]
  elif sub_cell == 'interneuron':
    data_embedding = data_embedding_all[:,interneuron]

  # output data
  data_preprocess = {}
  data_preprocess = {'data_embedding':data_embedding,
                    'speed_ts':speed_ts,
                    'pos_lin_bin':pos_lin_bin,
                    'pos_x':pos_x,
                    'pos_y':pos_y,
                    'lin_pos_sm':lin_pos_sm,
                    'trial_num_bin':trial_num_bin,
                    'epoch':epoch,
                    'shuffle_len':shuffle_len,
                    'data_len':data_len,
                    'sub_cell':sub_cell,

                    }
  return data_preprocess

def run_dimensionality_reduction(data_preprocess,params):
  data_embedding = data_preprocess['data_embedding']
  speed_ts = data_preprocess['speed_ts']
  pos_lin_bin = data_preprocess['pos_lin_bin']
  pos_x = data_preprocess['pos_x']
  pos_y = data_preprocess['pos_y']
  lin_pos_sm = data_preprocess['lin_pos_sm']

  trial_num_bin = data_preprocess['trial_num_bin']
  epoch = data_preprocess['epoch']
  data_len = data_preprocess['data_len']
  shuffle_len = data_preprocess['shuffle_len']

  method = params['method']
  n_component = params['n_component']
  iterations = params['iterations']
  supervised = params['supervised']
  sub_cell  = params['sub_cell']

  decoding_target='trial'
  ########################### DECODING LABEL/TARGET #############################
  if decoding_target =='pos':

      target1 = np.ones((data_len,1))
      target2 = 10000*np.ones((shuffle_len,1))
      target = np.concatenate([target1, target2], axis=0)

      target_knn_1 =  np.expand_dims(pos_lin_bin,axis=1)
      target_knn_2 = 10000*np.ones((shuffle_len,1))
      target_knn = np.concatenate([target_knn_1,target_knn_2],axis=0)
      target_knn = np.ravel(target_knn)

  elif decoding_target =='trial':

      target1 = np.expand_dims(trial_num_bin,axis=1)
      target2 = 10000*np.ones((shuffle_len,1))
      target = np.concatenate([target1,target2],axis=0)
      print(target.shape)
      target_knn = target
      target_knn = np.ravel(target_knn)



  ##################  Perform dimensionality reduction ########################
  if n_component > np.shape(data_embedding)[1]:
      n_component = np.shape(data_embedding)[1]
  print('n_component')
  print(n_component)

  if method == 'pca':
      utrans = PCA(n_component=n_component)
  elif method =='umap':
      utrans = umap.umap_.UMAP(n_neighbors=20, n_components=n_component, metric='cosine', metric_kwds=None,
                              output_metric='euclidean',
                              output_metric_kwds=None, n_epochs=iterations, learning_rate=1.0, init='spectral',
                              min_dist=0.1, spread=1.0, low_memory=True,
                              n_jobs=-1, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0,
                              negative_sample_rate=5, transform_queue_size=4.0,
                              a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1,
                              target_metric='categorical', target_metric_kwds=None, target_weight=0.5,
                              transform_seed=42, transform_mode='embedding', force_approximation_algorithm=False,
                              verbose=False, tqdm_kwds=None, unique=False, densmap=False, dens_lambda=2.0,
                              dens_frac=0.3, dens_var_shift=0.1, output_dens=False, disconnection_distance=None,
                              precomputed_knn=(None, None, None))

  print('embedding ongoing...')

  print(f"original data shape: {data_embedding.shape}")
  print(f"target shape: {target.shape}")

  # DIMENSIONALITY REDUCTION WITH UMAP
  if supervised:
    embedding = utrans.fit_transform(data_embedding, y=target)
  else:
    embedding = utrans.fit_transform(data_embedding)


  print(f"low-dimensional data shape: {target.shape}")

  data_result = {}
  data_result = {'data_embedding':data_embedding,
                 'speed_ts':speed_ts,
                 'pos_lin_bin':pos_lin_bin,
                 'trial_num_bin':trial_num_bin,
                 'epoch':epoch,
                 'embedding':embedding,
                 'data_len':data_len,
                 'shuffle_len':shuffle_len,
                 'target':target,
                 'sub_cell':sub_cell,
                 'pos_x':pos_x,
                 'pos_y':pos_y,
                 'lin_pos_sm':lin_pos_sm,
                }
  return data_result

def plot_umap_result(data_result,dim1=0,dim2=1,dim3=2,with_shuffle=False):

    data_embedding = data_result['data_embedding']
    print("data_embedding")
    print(data_embedding.shape)
    embedding = data_result['embedding']
    pos_lin_bin = data_result['pos_lin_bin']
    lin_pos_sm = data_result['lin_pos_sm']
    pos_x = data_result['pos_x']
    pos_y = data_result['pos_y']
    trial_num_bin = data_result['trial_num_bin']
    data_len = data_result['data_len']
    shuffle_len = data_result['shuffle_len']
    target = data_result['target']

    # make interactive plot with plotly

    fig = make_subplots(rows=2, cols=2,
                        # specs=[[{'is_3d': True}, {'is_3d': False}]],
                        specs=[[{'type': 'scene'}, {'type': 'xy'}],
                                [{'type': 'scene'}, {'type': 'scene'}]], # scence make it 3d
                        subplot_titles=['Position manifold', 'Position trajectory',
                                        'Trial manifold', 'Trial trajectory'],
                        print_grid =False)
    # row 1 col 1 : Position manifold
    fig.add_trace(go.Scatter3d(x=embedding[:data_len,dim1], y=embedding[:data_len,dim2], z=embedding[:data_len,dim3],mode='markers',marker=dict(
            size=1,
            color=pos_lin_bin,                # set color to an array/list of desired values
            colorscale='jet',   # choose a colorscale
            opacity=0.8)), row=1, col=1)
    if with_shuffle:
        fig.add_trace(go.Scatter3d(x=embedding[data_len:,dim1], y=embedding[data_len:,dim2],z=embedding[data_len:,dim3],mode='markers',marker=dict(
                size=1,
                color='grey',                # set color to an array/list of desired values
                opacity=0.8)), row=1, col=1)
    # row 1 col 2: Position trajectory
    fig.add_trace(go.Scatter(x=pos_x, y=pos_y,mode='markers',marker=dict(
            size=1,
            color=pos_lin_bin,                # set color to an array/list of desired values
            colorscale='jet',   # choose a colorscale
            opacity=0.8)), row=1, col=2,
            )

    # row 2 col 1: Trial manifold
    fig.add_trace(go.Scatter3d(x=embedding[:data_len,dim1], y=embedding[:data_len,dim2], z=embedding[:data_len,dim3],mode='markers',marker=dict(
            size=1,
            color=trial_num_bin,                # set color to an array/list of desired values
            colorscale='jet',   # choose a colorscale
            opacity=0.8)), row=2, col=1)
    if with_shuffle:
        fig.add_trace(go.Scatter3d(x=embedding[data_len:,dim1], y=embedding[data_len:,dim2],z=embedding[data_len:,dim3],mode='markers',marker=dict(
                size=1,
                color='grey',                # set color to an array/list of desired values
                opacity=0.8)), row=2, col=1)
    # row 2 col 2: Trial trajectory
    fig.add_trace(go.Scatter3d(x=pos_x, y=pos_y,z = trial_num_bin,mode='markers',marker=dict(
            size=1,
            color=trial_num_bin,                # set color to an array/list of desired values
            colorscale='jet',   # choose a colorscale
            opacity=0.8,
            )), row=2, col=2)

    fig.update_xaxes(showgrid=False, zeroline=False,row=1, col=2)
    fig.update_xaxes(showgrid=False, zeroline=False,row=2, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False,row=1, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False,row=2, col=2)

    # no x axis y axis and no grid
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)


    # update figure size
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ))
    # update background color of all subplots as well as paper color
    fig.update_layout(
        plot_bgcolor='black',paper_bgcolor="black"

    )
    # update the font color
    fig.update_layout(font = dict(color = 'white'))

    fig.show()
    fig.write_html('umap_pos.html')
    return fig