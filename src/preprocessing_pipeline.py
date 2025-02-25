import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate

class PreprocessingPipeline:
    """
    Pipeline for preprocessing hippocampal neural and behavioral data.
    """
    def __init__(self, data, params, session_id):
        self.data = data
        self.params = params
        self.session_id = session_id
    
    def filter_behavior(self):
        """Filter behavior data within session time range using the correct session time extraction."""
        session_data = self.data['session']['session']
        try:
            session_epochs = session_data['epochs'][0][0]
            start_time = session_epochs[0, self.session_id]['startTime'][0][0][0][0]
            stop_time = session_epochs[0, self.session_id]['stopTime'][0][0][0][0]
        except:
            session_epochs = session_data['epochs']
            start_time = session_epochs[self.session_id]['startTime']
            stop_time = session_epochs[self.session_id]['stopTime']
        
        timestamps = self.data['behavior']['behavior']['timestamps'][0][0]
        timestamps = self.data['behavior']['behavior']['timestamps'][0][0]
        mask = (timestamps > start_time) & (timestamps < stop_time)
        filtered_behavior = {
            'timestamps': timestamps[mask],
            'x': self.data['behavior']['behavior']['position'][0][0]['x'][0][0][mask],
            'y': self.data['behavior']['behavior']['position'][0][0]['y'][0][0][mask],
            'lin': self.data['behavior']['behavior']['position'][0][0]['lin'][0][0][mask],
            'trial': self.data['behavior']['behavior']['masks'][0][0]['TRIALS'][0][0][mask]
        }
        self.data['behavior_filtered'] = filtered_behavior
        return self
    
    def bin_spike_data(self):
        """Bin spike train data based on the selected time bin size, following the original script."""
        spk_bin = self.params['spk_bin']
        spike_time = self.data['spike_time']['spike_time'][0][0]
        ts_beh = self.data['behavior_filtered']['timestamps']
        print(spike_time, ts_beh)
        #print(spike_time)
        #print(ts_beh)
        num_cell = spike_time.shape[0]
        spk_ts = np.arange(ts_beh[0], ts_beh[-1], spk_bin)
        spk_count = np.zeros((num_cell, spk_ts.shape[0] - 1))
        

        for spk in range(num_cell):
            hist_count = np.histogram(spike_time[spk], bins=spk_ts)
            spk_count[spk, :] = hist_count[0]
        
        self.data['spike_binned'] = spk_count.T
        self.data['time_bins'] = spk_ts
        return self
    
    def interpolate_behavior(self):
        """Interpolate position and behavioral variables to match spike timestamps."""
        spk_ts = self.data['time_bins']
        for key in ['x', 'y', 'lin', 'trial']:
            interp_func = interpolate.interp1d(self.data['behavior_filtered']['timestamps'], 
                                               self.data['behavior_filtered'][key], 
                                               kind='cubic', fill_value='extrapolate')
            self.data[f'behavior_interp_{key}'] = interp_func(spk_ts)
        return self
    
    def smooth_behavior(self):
        """Apply Gaussian smoothing to behavioral variables."""
        sm_window = self.params['sm_window']
        for key in ['x', 'y', 'lin']:
            self.data[f'behavior_smooth_{key}'] = ndimage.gaussian_filter(self.data[f'behavior_interp_{key}'], sm_window)
        return self
    
    def apply_speed_threshold(self):
        """Filter out time points where speed is below the threshold."""
        speed_lim = self.params['speed_lim']
        lin_pos = self.data['behavior_smooth_lin']
        speed = np.abs(np.diff(lin_pos)) / np.mean(np.diff(self.data['time_bins']))
        speed_mask = speed >= speed_lim
        self.data['filtered_spike_data'] = self.data['spike_binned'][speed_mask]
        return self
    
    def normalize_spike_data(self):
        """Normalize spike data (z-score across time)."""
        self.data['spike_zscore'] = (self.data['filtered_spike_data'] - np.mean(self.data['filtered_spike_data'], axis=0)) / \
                                     np.std(self.data['filtered_spike_data'], axis=0)
        return self
    
    def preprocess(self):
        """Run all preprocessing steps in sequence."""
        return (self
                .filter_behavior()
                .bin_spike_data()
                .interpolate_behavior()
                .smooth_behavior()
                .apply_speed_threshold()
                .normalize_spike_data()
                .data)
