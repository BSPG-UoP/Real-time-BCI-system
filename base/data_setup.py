# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 01:59:18 2019

@author: Κόκκινος
"""

import numpy as np

def Epochs(data, trig, ep_dur):
    
    data = np.array(data)
    trig = np.array(trig)
    ep_data = []
    
    if data.ndim != 2:    
        raise ValueError("Invalid input data dimensions")    
    
    for trigger in trig:
        ep_data += [data[trigger:(trigger+int(ep_dur))],]
        
    ep_data = np.array(ep_data)
    ep_data = np.transpose(ep_data,(0,2,1))
    
    return ep_data
    

def data_specs(trig, IM_dur, ep_dur = None, labels = None, mode = None):
    
    """
    Functionality
    ------------
    Divides given data into epochs/trials
    
    Parameters
    ------------
    data : raw_recording to be epoched, array, shape(samples,channels)    
    specs: specifications on how the data will be epoched
        specs[0]:timepoints for the start of each epoch, array
        specs[1]:Duration of each trial, int or array if epochs durations differ
    
    Returns
    ------------
    ep_data:data epoched, array, shape(trials,channels,sample)
    """
    
            
    if mode == 'Rvsall':

        n_trig = [[]]
        n_labels = [[]]
        
        for trigger in trig:
                
            n_trig[0].extend([trigger - IM_dur, trigger])
            n_labels[0].extend([0,1])
        
        n_trig.append(trig)
        n_labels.append(labels)
        trig = n_trig
        labels = n_labels
    
    elif mode == 'IMvsall':
        
        l = np.unique(labels)
        n_trig = [[],[]]
        n_labels = [[],[]]
        
        for index,cls in enumerate(l):            
            for trigger, label in zip(trig,labels):
                
                n_trig[index].extend([trigger - IM_dur])
                n_labels[index].extend([0])
                
                if label == cls:
                    
                    n_trig[index].extend([ trigger])
                    n_labels[index].extend([label])
                
                else:
                    n_trig[index].extend([ trigger])
                    n_labels[index].extend([0])
        
        n_trig.append(trig)
        n_labels.append(labels)
        trig = n_trig
        labels = n_labels

    elif mode == 'IMvsRest':
        
        l = np.unique(labels)
        n_trig = [[],[]]
        n_labels = [[],[]]
        
        for index,cls in enumerate(l):            
            for trigger, label in zip(trig,labels):
                
                n_trig[index].extend([trigger - IM_dur])
                n_labels[index].extend([0])
                
                if label == cls:
                    
                    n_trig[index].extend([ trigger])
                    n_labels[index].extend([label])
                    
        n_trig.append(trig)
        n_labels.append(labels)
        trig = n_trig
        labels = n_labels
        
    elif mode == 'CSP_OVR':
        
        n_labels = [[]]
        n_trig = [[]]
        for label,trigger in zip(labels,trig):
            n_labels[0].extend([0,label])
            n_trig[0].extend([trigger - IM_dur, trigger])
            
        trig = np.array(n_trig)
        labels = np.array(n_labels)
            
    else:
        trig = [trig]
        labels = [labels]
    
    if ep_dur != None and IM_dur != ep_dur:
        
        L = int(IM_dur / ep_dur)
        n_trig = [[],[]]
        n_labels = [[],[]]
        for index in range(len(trig)):
            for trigger, label in zip(trig[index],labels[index]):
                                
                n_trig[index].extend(range(trigger, trigger + IM_dur, ep_dur))
                n_labels[index].extend([label] * L)
            
        trig = np.array(n_trig)
        labels = np.array(n_labels)
    
    return trig, labels

def data_Split(data, labels, TrnF, pos = 0,shuffle = False, trig = None):
    
    """
    Functionality
    ------------
    Splits data into training and testing data.
    
    Parameters
    ------------
    data : array, shape(trials, channels, samples)
    labels : list
    shuffle: if True the data are shuffled before the splitting, boolean

    Returns
    ------------
    Tr_data : training data, array, shape(trials, channels, channels)
    """
    
    if data.ndim != 3 and data.ndim != 2:
        
        raise ValueError("Invalid input data dimensions")
            
    if data.ndim == 3:
        
        if shuffle:
            
            perm = np.random.permutation(len(labels))
            shuffle_data = []
            shuffle_labels = []
            
            for trial in perm:
                
                shuffle_data += (data[trial],) 
                shuffle_labels += (labels[trial],)
            shuffle_data = np.array(shuffle_data)
            shuffle_labels = np.array(shuffle_labels)
    
            TrN = data.shape[0]
            TrnN = round(TrnF * TrN)
            
            Tr_data = shuffle_data[pos:(pos+TrnN)]
            Tr_labels = shuffle_labels[pos:(pos+TrnN)]
            Tst_data = np.delete(shuffle_data,range(pos,pos+TrnN),axis=0)
            Tst_labels = np.delete(shuffle_labels,range(pos,pos+TrnN),axis=0)
            
            return Tr_data,Tr_labels,Tst_data,Tst_labels
        
        else:
            
            TrN = data.shape[0]
            TrnN = round(TrnF * TrN)
            pos_data = int(pos * data.shape[0])
            pos_labels = int(pos * len(labels))
            
            Tr_data = data[pos_data:(pos_data+TrnN)]
            Tr_labels = labels[pos_labels:(pos_labels+TrnN)]
            Tst_data = np.delete(data,range(pos_data,pos_data+TrnN),axis=0)
            Tst_labels = np.delete(labels,range(pos_labels,pos_labels+TrnN),axis=0)
            
            return Tr_data,Tr_labels,Tst_data,Tst_labels
    
    if data.ndim == 2:

        split = int(len(trig)*TrnF)
        pos_data = int(pos * data.shape[0])
        pos_labels = int(pos * len(labels))
        Tr_data = data[pos_data:pos_data+trig[split]]
        Tr_labels = labels[pos_labels:(pos_labels+split)]
        Tst_data = np.delete(data,range(pos_data,pos_data+trig[split]),axis=0)
        Tst_labels = np.delete(labels,range(pos_labels,pos_labels+split),axis=0)
            
        return Tr_data,Tr_labels,Tst_data,Tst_labels
        

def rt_labels(IMdur, labels, TrnF, mode):
    
    tst_labels = []     
    
    if mode == 'sync':
        for label in labels[int(TrnF*len(labels)):]:
            tst_labels.extend([label]*IMdur)
        tst_labels = np.array(tst_labels).flatten()
        
    else:
        for label in labels[int(TrnF*len(labels)):]:
            tst_labels.extend([[0]*IMdur,[label]*IMdur])
        tst_labels = np.array(tst_labels).flatten()
        
    return tst_labels

def offline_labels(IMdur, labels, TrnF, mode):
    
    tst_labels = []     
    
    if mode == 'sync':
        for label in labels[int(TrnF*len(labels)):]:
            tst_labels.extend([label])
        tst_labels = np.array(tst_labels).flatten()
        
    else:
        for label in labels[int(TrnF*len(labels)):]:
            tst_labels.extend([0,label])
        tst_labels = np.array(tst_labels).flatten()
        
    return tst_labels

from sklearn.metrics import accuracy_score , confusion_matrix

def Results( pred, labels, res_type = 'accuracy'):
        
        if res_type == 'accuracy':
            return accuracy_score(labels, pred)
        
        else:
            return accuracy_score(labels, pred), confusion_matrix(labels, pred)
    
    






