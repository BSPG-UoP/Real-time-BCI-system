# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:16:14 2019

@author: Kokkinos
"""

import numpy as np

from base.bci_transducer import Transducer 
from base.data_setup import *

from offline import Offline

from Lib.warnings import warn

class Simul(Offline):

    def __init__(self, mode = 'IMvsall', tim_window = 4, vote_window = 4, overlap = 0,
                 TrnF = 0.5, IMdur = 4, Fs=None, filtering=None, bp = None, 
                 components = 4, red_type = None, FB_CSP = None, bank = None, neighbors = 5):
        
#        Transducer.__init__(self, Fs, filtering, bp, components, 
#                            red_type, FB_CSP, bank, neighbors)
#        
#        if mode == 'IMvsall' or 'IMvsRest' or 'Rvsall' or 'CSP_OVR' or 'sync':
#            self.mode = mode
#        else:
#            raise ValueError('Inappropriate mode value')
#        
#        self.TrnF = TrnF
#        self.IMdur = IMdur * Fs
        
        Offline.__init__(self, mode, TrnF, IMdur, Fs, filtering, bp, components, 
                         red_type, FB_CSP, bank, neighbors)
        
        if self.mode == 'sync':
            self.tim_window = self.IMdur
            self.vote_window = self.tim_window
            self.step = 2 * self.IMdur
       
        else:
            self.tim_window = tim_window * self.Fs       
            self.vote_window = vote_window * self.Fs
#            if vote_window == tim_window:
#                self.overlap = 0
#                if overlap != 0:
#                    warn('Overlap value set to 0')
#            else:
#                self.overlap = overlap
            self.overlap = overlap
            self.step = int(self.tim_window * (1 - self.overlap))
    

    
    def simul_test_specs(self, data, LABELS, trig):
        
        trigs, labels = data_specs(trig, self.IMdur, labels = LABELS, mode = self.mode)

        #setup testing data and labels
        Tr_data, Tr_labels, Tst_data ,Tst_labels = data_Split(data, LABELS, 
                                                              self.TrnF, trig = trigs[0], 
                                                              shuffle = False)
        tst_labels = rt_labels(self.IMdur, LABELS, self.TrnF, self.mode)
                
        return Tst_data, tst_labels
        
    
    def simul_predict(self, tst_data):
        
        tim_window = int(self.tim_window)     
        step = self.step
        vote_window = int(self.vote_window)
            
        vote = [0] * len(self.bcis)
        prediction = []
        
        for index in range(0, len(tst_data), step):

            chunk = (tst_data[index : index+tim_window].T)
            if len(chunk.T)<27:
                break

            
            chunk = chunk.reshape((1, chunk.shape[0], chunk.shape[1]))
            
            pred = []
            for i, bci in enumerate(self.bcis[:-1]):
                
                pred.append(bci.predict(chunk))
            
                if pred[i] != 0:
                    vote[i] += 1
                else:
                    vote[i] -= 1
                    
            pred.append(self.bcis[-1].predict(chunk))
            
            if pred[-1] == 1:
                vote[-1] += 1
            else:
                vote[-1] -= 1
            if (index + step) % vote_window == 0:
                
                if self.mode == 'IMvsall' or self.mode == 'IMvsRest':
                    
                    if vote[0] <= 0 and vote[1] <= 0:
                        prediction.extend([0] * vote_window)
                    elif vote[2] == 1:
                        prediction.extend([1] * vote_window)
                    else:
                        prediction.extend([2] * vote_window)
                        
                elif self.mode == 'Rvsall':
                    
                    if vote[0] <= 0:
                        prediction.extend([0] * vote_window)
                    elif vote[1] > 0: #tsek gia labels 0,1,2
                        prediction.extend([1] * vote_window)
                    else:
                        prediction.extend([2] * vote_window)
                        
                else:
                    prediction.extend([pred[-1]] * vote_window)
                    
                    
                vote = [0] * len(vote)
                
        return prediction


if __name__ == '__main__': 
    
    import scipy.io as sio
    
    """    
    EEG = sio.loadmat('lab_rec\PANTAZ_EEG_s2.mat')['EEGDATA']
    LABELS = sio.loadmat('lab_rec\PANTAZ_LABELS_s2.mat')['LABELS'].flatten()
    trig = sio.loadmat('lab_rec\PANTAZ_TRIG_s2.mat')['trig'].flatten()
    
    Fs= 250
    
    IMdur = 4*Fs
    
    TrnF = 0.5
    
    bp = [5,30]
    
    EEG = EEG/1000000000
    
    bank = [range(4,32,3),4]
    
    import time
    start = time.time()   
    
    sim = Simul(mode = 'CSP_OVR', tim_window = 4, vote_window = 4, overlap = 0, Fs = Fs, 
                filtering = 'full', bp = bp,  FB_CSP = None, bank = bank)   
    
    sim.offline_fit(EEG, LABELS, trig)
    tst_data, tst_labels = sim.simul_test_specs(EEG, LABELS, trig)
    pred = sim.simul_predict(tst_data)
    
    
    ac, cf = Results(pred, tst_labels, res_type = 'full') 
    print(ac)
    print(cf)
    
    print(time.time()-start)
    """
    electrodes = [10,11,13,14,26,27,29,30,42,43,45,46]
    EEG = sio.loadmat('datasets\BCICIV_ds1\BCICIV_calib_ds1f.mat')['cnt'][:,electrodes]#[:,20:30]
    LABELS = sio.loadmat('datasets\BCICIV_ds1\LABELS_ds1f.mat')['LABELS'].flatten()
    trig = sio.loadmat('datasets\BCICIV_ds1\\trig_ds1f.mat')['trig'].flatten()
    tst_data = sio.loadmat('datasets\BCICIV_ds1\BCICIV_eval_ds1f.mat')['cnt'][:,electrodes]
    tst_labels = sio.loadmat('datasets\BCICIV_ds1\LABELS_eval_ds1f.mat')['y'].flatten()
    tst_data = tst_data
    tst_labels = tst_labels
    

#    i = np.where(np.isnan(tst_labels))
#    tst_data = np.delete(tst_data,i,0)
#    print(tst_data.shape)
#    tst_labels = np.delete(tst_labels,i)

    
    i = np.where(LABELS == -1)
    LABELS[i] = 2
    i = np.where(tst_labels == -1)
    tst_labels[i] = 2

    
    
    Fs= 100
    
    IMdur = 4
    
    TrnF = 1
    
    bp = [1,19]
    
    EEG = EEG
    
    bank = [range(9,32,4),4]
    
    import time
    start = time.time() 

    sim = Simul(mode = 'IMvsRest', tim_window = 4, vote_window = 4, overlap = 0, Fs = Fs, 
                filtering = None, bp = bp,  FB_CSP = 4, bank = bank, TrnF = TrnF)   
    
    sim.offline_fit(EEG, LABELS, trig)
#    tst_data, tst_labels = sim.simul_test_specs(EEG, LABELS, trig)
    pred = sim.simul_predict(tst_data)
    
    pred = pred[:tst_data.shape[0]]
    tst_labels = tst_labels[:len(pred)]
    try:
        tst_labels = np.array(tst_labels)
    except:
        pass
    try:
        pred = np.array(pred)
    except:
        pass   
    i = np.where(np.isnan(tst_labels))
    pred = np.delete(pred,i)
    tst_labels = np.delete(tst_labels,i)
        
    ac, cf = Results(pred, tst_labels, res_type = 'full') 
    print(ac)
    print(cf)
    mse = cf[0,1]+cf[0,2]+cf[1,0]+cf[2,0] + (cf[1,2]+cf[2,1])*2
    print(mse/len(tst_labels))
    print(sum(cf))
    print(time.time()-start)
                    
