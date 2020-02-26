# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:16:14 2019

@author: Kokkinos
"""

import numpy as np

from base.bci_transducer import Transducer 
from base.data_setup import *

from Lib.warnings import warn

from multiprocessing import Process
import multiprocessing

import pickle
import time

import scipy.io as sio
class Offline(Transducer):

    def __init__(self, mode = 'IMvsall',
                 TrnF = 0.5, IMdur = 4, Fs=None, filtering=None, bp = None, 
                 components = 4, red_type = None, FB_CSP = None, bank = None, neighbors = 5):
        
        Transducer.__init__(self, Fs, filtering, bp, components, 
                            red_type, FB_CSP, bank, neighbors)
        
        if mode == 'IMvsall' or 'IMvsRest' or 'Rvsall' or 'CSP_OVR' or 'sync':
            self.mode = mode
            if mode == 'CSP_OVR' and self.FB_CSP != None:
                self.FB_CSP = None
                self.filtering = 'full'
                warn('In CSP_OVR, FBCSP should be False!', Warning)
        else:
            raise ValueError('Inappropriate mode value')
        
        self.TrnF = TrnF
        self.IMdur = IMdur * Fs

    
    def offline_fit(self, data, LABELS, trig = None, pos = 0):
        
        if data.ndim == 3:
            
            if self.TrnF != 1:
                Tr_data, Tr_labels, Tst_data ,Tst_labels = data_Split(data, LABELS, self.TrnF, 
                                                                  shuffle = False, pos = pos)
            bci = Transducer(Fs = self.Fs, filtering = self.filtering, 
                             bp = self.bp,  FB_CSP = self.FB_CSP, bank = self.bank)
            bci.fit(Tr_data, Tr_labels)
            self.bcis = [bci]
        
        else:
            
            trigs, labels = data_specs(trig, self.IMdur, labels = LABELS, mode = self.mode)       

            self.bcis = []
            args = []
            pool = multiprocessing.Pool()
            for trigger, label in zip(trigs, labels):
                args.append((data,trigger,label,pos))
#                args = [data,trigger,label,pos]
#            self.bcis = [self.fit_thread(args)]
            self.bcis = pool.map(self.fit_thread,args)

        return self
    
    def fit_thread(self, args):
        data, trigger, label, pos = args
        if self.TrnF != 1:
                    
            Tr_data, Tr_labels, Tst_data1 ,Tst_labels1 = data_Split(data, label, 
                                                              self.TrnF, trig = trigger, 
                                                              shuffle = False, pos = pos)                                   
            Tr_ep_data = Epochs(Tr_data, trigger[:int(self.TrnF*len(trigger))], self.IMdur)
        
        else:

            Tr_ep_data = Epochs(data, trigger, self.IMdur)
            Tr_labels = label

        bci = Transducer(Fs = self.Fs, filtering = self.filtering, 
                         bp = self.bp,  FB_CSP = self.FB_CSP, bank = self.bank)
        bci.fit(Tr_ep_data,Tr_labels)
        sio.savemat('CSP_filters.mat', {'filters':bci.filters})
        return bci
        
    
    def offline_test_specs(self, data, LABELS, trig = None, pos = 0):
        
        if trig is None and data.ndim == 3:
            
            Tr_data, Tr_labels, Tst_data ,Tst_labels = data_Split(data, LABELS, self.TrnF, 
                                                                  shuffle = False, pos = pos)
            tst_labels = Tst_labels
        else:
        
            trigs, labels = data_specs(trig, self.IMdur, labels = LABELS, mode = self.mode)
            
            if self.mode == 'IMvsRest':
                
                n_trig = []
                for trigger in trig:
                    n_trig.extend([trigger - self.IMdur, trigger])
                data = Epochs(data, n_trig, self.IMdur)
            
            else:
                data = Epochs(data, trigs[0], self.IMdur)
                
            Tr_data, Tr_labels, Tst_data ,Tst_labels = data_Split(data, LABELS, 
                                                              self.TrnF, trig = trigs[0], 
                                                              shuffle = False, pos = pos)
            tst_labels = offline_labels(self.IMdur, LABELS, self.TrnF, self.mode)

        
        
        return Tst_data, tst_labels
        
    def predict_thread(self, args):
        
        bci,tst_data = args
        pred = bci.predict(tst_data)
        return pred
    
    def offline_predict(self, tst_data):
        
        pool = multiprocessing.Pool()
        
        pred = []
        prediction = []
        args = []
        for bci in self.bcis:
            args.append([bci,tst_data])
#            args = [bci, tst_data]
#        pred = [self.predict_thread(args)]
        pred = pool.map(self.predict_thread,args)
        if self.mode == 'IMvsall' or self.mode == 'IMvsRest':        
            for trial in np.array(pred).T:
                                
                if trial[0] == 0 and trial[1] == 0:
                    prediction.extend([0])
                elif trial[2] == 1:
                    prediction.extend([1])
                else:
                    prediction.extend([2])
                    
        elif self.mode == 'Rvsall':
            for trial in np.array(pred).T:    
                if trial[0] == 0:
                    prediction.extend([0])
                elif trial[1] == 1: #tsek gia labels 0,1,2
                    prediction.extend([1])
                else:
                    prediction.extend([2])
                    
        else:
            prediction = pred[0]
        return prediction
    
    def save_bcis(self, filename):
        with open(filename, 'wb') as train:
            pickle.dump(self.bcis, train)


if __name__ == '__main__':
    
    import scipy.io as sio
        
    EEG = sio.loadmat('lab_rec\PANTAZ_EEG_s2.mat')['EEGDATA']
    LABELS = sio.loadmat('lab_rec\PANTAZ_LABELS_s2.mat')['LABELS'].flatten()
    trig = sio.loadmat('lab_rec\PANTAZ_TRIG_s2.mat')['trig'].flatten()

#    EEG = EEG[:int(0.68*len(EEG))]
#    LABELS = LABELS[:int(0.68*len(LABELS))]
#    trig = trig[:int(0.68*len(trig))]

#    EEG = sio.loadmat('datasets\BCI3ds4v_EEG.mat')['cnt']
#    LABELS = sio.loadmat('datasets\BCI3ds4v_LABELS.mat')['LABELS'].flatten()
#    trig = sio.loadmat('datasets\\BCI3ds4v_trig.mat')['trig'].flatten()
#    
#    EEG = sio.loadmat('datasets\BCICIV_ds1\BCICIV_calib_ds1a.mat')['cnt']#[:,20:30]
#    LABELS = sio.loadmat('datasets\BCICIV_ds1\LABELS_ds1a.mat')['LABELS'].flatten()
#    trig = sio.loadmat('datasets\BCICIV_ds1\\trig_ds1a.mat')['trig'].flatten()
    
#    EEG = sio.loadmat('datasets\BCIcomp_II_EEGDATA.mat')['EEGDATA']#[:,20:30]
#    LABELS = sio.loadmat('datasets\BCIcomp_II_LABELS.mat')['LABELS'].flatten()
#    trig = None

    
    
    
#    for i in range(len(LABELS)):
#        if LABELS[i] == -1:
#            LABELS[i] = 2
    
    Fs= 250
    
    IMdur = 4
    
    TrnF = 0.5
    
    bp = [7,14]
    
    #EEG = EEG[int(0.5*EEG.shape[0])]/1000000000
    #LABELS = LABELS[int(0.5*len(LABELS))]
    #trig = trig[int(0.5*len(trig))]
    #print(len(LABELS))
    
    bank = [range(4,32,4),4]
    
    EEG = EEG#/1000000
    
    import time
    start = time.time()
        
    off = Offline(mode = 'IMvsall', Fs = Fs, TrnF = TrnF, filtering = None, bp = bp,
                  FB_CSP = 4, bank = bank, IMdur = IMdur)   

    
    off.offline_fit(EEG, LABELS, trig = trig)
    tst_data, tst_labels = off.offline_test_specs(EEG, LABELS, trig = trig)
    pred = off.offline_predict(tst_data)



                

    ac, cf = Results(pred, tst_labels, res_type = 'full') 
    print(ac)
    print(cf)
    mse = cf[0,1]+cf[0,2]+cf[1,0]+cf[2,0] + (cf[1,2]+cf[2,1])*2
    print(mse/len(tst_labels))
    fpr = (cf[0,1]+cf[0,2])/(cf[0,1]+cf[0,2]+cf[0,0])
    print(fpr)

    
    # off.save_bcis('bcis')
    print(time.time()-start)
                        
