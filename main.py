# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:49:11 2019

@author: Κόκκινος
"""

from offline import Offline
from real_time import rt
from base.data_setup import *

import scipy.io as sio

def offline_analysis(Fs = 250, IMdur = 4, TrnF = 0.5, mode = 'CSP_OVR', bp = None, 
                     filtering = None, FB_CSP = None, bank = None):
    
    import time
    start = time.time()

    EEG = sio.loadmat('lab_rec\PANTAZ_EEG_s2.mat')['rec']
    LABELS = sio.loadmat('lab_rec\PANTAZ_LABELS_s2.mat')['LABELS'].flatten()
    trig = sio.loadmat('lab_rec\PANTAZ_TRIG_s2.mat')['trig'].flatten()
        
    off = Offline(mode = mode, Fs = Fs, TrnF = TrnF, filtering = filtering, bp = bp,
                  FB_CSP = FB_CSP, bank = bank, IMdur = IMdur)   
    
    off.offline_fit(EEG, LABELS, trig = trig)
    tst_data, tst_labels = off.offline_test_specs(EEG, LABELS, trig = trig)
    pred = off.offline_predict(tst_data)
    
    ac, cf = Results(pred, tst_labels, res_type = 'full') 
    print(ac)
    print(cf)

    print(time.time()-start)

def train_system(Fs = 250, IMdur = 4, mode = 'CSP_OVR', bp = None, 
                 filtering = None, FB_CSP = None, bank = None):
    
    EEG = sio.loadmat('lab_rec\PANTAZ_EEG_s2.mat')['rec']
    LABELS = sio.loadmat('lab_rec\PANTAZ_LABELS_s2.mat')['LABELS'].flatten()
    trig = sio.loadmat('lab_rec\PANTAZ_TRIG_s2.mat')['trig'].flatten()
        
    off = Offline(mode = mode, Fs = Fs, TrnF = 1, filtering = filtering, bp = bp,
                  FB_CSP = FB_CSP, bank = bank, IMdur = IMdur)   
    
    off.offline_fit(EEG, LABELS, trig = trig)
    off.save_bcis('train')
                        

def real_time_system(mode = 'CSP_OVR', time_window = 4, vote_window = 4, overlap = 0):
    b_c_i = rt(mode = mode, tim_window = 4, vote_window = 8, overlap = 0.5)
    b_c_i.load_bcis('train')
    b_c_i.begin_stream()
    buffer = b_c_i.main_loop()
    b_c_i.save_recording(buffer)


#Define Specs

Fs = 250 # sampling frequency
IMdur = 4 # duration of Imaginary Movements during training
TrnF = 0.5 # Percentage of training examples for offline analysis

mode = 'IMvsall' # mode of system. Options: sync, CSP_OVR, IMvsall, IMvsRest, Rvsall

FB_CSP = 4 # chooses method. If int then method:FBCSP with k = FBCSP, if None then method:CSP
bp = [5,30] # cutoff frequencies for bandpass filtering if method: CSP
filtering = None # type of filtering, options: None, bandpass, full(+notch), set to None for FBCSP 
bank = [range(4,32,3),4] # Filterbank specs

time_window = 4 # chunk to predict class (secs)
vote_window = 4 # window for final decision (secs), must be time_window multiple
overlap = 0 # overlap between chunks (percentage)

#Run system
if __name__ == '__main__':

    offline_analysis(Fs = Fs, IMdur = IMdur, TrnF = TrnF, mode = mode, bp = bp, 
                     filtering = filtering, FB_CSP = FB_CSP, bank = bank)

    # train_system(Fs = Fs, IMdur = IMdur, mode = mode, bp = bp, 
    #             filtering = filtering, FB_CSP = FB_CSP, bank = bank)

    # real_time_system(mode = 'CSP_OVR', time_window = time_window, 
    #                 vote_window = vote_window, overlap = overlap)





        




