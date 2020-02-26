# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:48:21 2019

@author: Κόκκινος
"""

import numpy as np
import numpy.linalg as na
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy import signal
from sklearn.feature_selection import mutual_info_classif as mi
from threading import Thread
import time

class Feature_Extraction:
    
    def __init__(self, components = 8, red_type = None, FB_CSP = None, bank = None):
        
        self.components = components
        self.red_type = red_type
        self.FB_CSP = FB_CSP
        self.bank = bank
    
    def CSP_Filters(self, data, labels):
        
        """
        Functionality
        ------------
        Extracts the CSP filters fron given data
        
        Arguments
        ------------
        data : training data, array, shape(trials, channels, samples)    
        labels : training labels, list
        
        Returns
        ------------
        self
        """
        
        Classes = np.unique(labels)
        self.ClsN = len(Classes)
        ClsN = self.ClsN
        
        if ClsN < 2:
            raise ValueError("Must have at least 2 classes")
        
        if data.ndim != 3:
            raise ValueError("Invalid input data dimensions")
        
        if data.shape[0] != len(labels):
            raise ValueError("Trials and trial Labels must have the same size")
                   

        ChN = data.shape[1]
        self.ChN = ChN        

        Rcov=np.zeros((ClsN,ChN,ChN))
        Rsum=np.zeros((ChN,ChN))
        filters = []
        
        for cls_iter,cls_label in enumerate(Classes):
            
            idxs = np.where(labels==cls_label)
            for trial in data[idxs]:
                
                Rcov[cls_iter] += covarianceMatrix(trial)
            Rcov[cls_iter] = Rcov[cls_iter]/len(idxs[0])
            Rsum += Rcov[cls_iter]
            
        for x in range(ClsN):
            

            Rx= Rcov[x]
            SFx = spatialFilter(Rx,Rsum)
            filters += [SFx,]
            #Special case: only two tasks, no need to compute any more mean variances
            if ClsN == 2:
                filters=np.asarray(filters)
                for i in range(ChN):
                    filters[:,i,:]=filters[:,i,:]/na.norm(filters[:,i,:])
                filters = filters[0]
                break
        

        if ClsN>2:
            filters=np.asarray(filters)
            for j in range(ClsN):
                for i in range(ChN):
                    filters[j,i,:]=filters[j,i,:]/na.norm(filters[j,i,:])
            filters = filters.reshape((filters.shape[0]*filters.shape[1],filters.shape[2]))
            
        
        #dimesionality reduction
        if ChN % 2 == 0:
            idxs = (np.arange(self.ChN/2)).astype(int)
            filters[2*idxs],filters[2*(idxs+1)-1] = filters[idxs],filters[-idxs-1]
        else:
            idxs = (np.arange(self.ChN/2)).astype(int)
            filters[2*idxs],filters[2*(idxs[:-1]+1)-1] = filters[idxs],filters[-idxs[:-1]-1]
        
        if self.red_type == 'CSP':
            self.filters = filters[:,:self.components]
        else:
            self.filters = filters   
        return self
            
    
    def CSP_Features(self, data):
        
        """
        Functionality
        ------------
        Extracts the logarithm of variance of the CSP features
        
        Parameters
        ------------
        data : data from which to extract features, array, shape(trials, channels, samples) or (channels,samples)    
        filters : CSP filters
        
        Returns
        ------------
        features : array, shape(trials, features) or list, shape(features)
        """
        
        filters = self.filters
        
        if data.ndim == 2:
            data = data.reshape((-1,data.shape[1],data.shape[0]))
        if filters.ndim == 1:
            filters = filters.reshape((-1,filters.shape[0]))
            
        
        features = []
        tr_features = []

        
        for trial in data:
            CSPfeat = filters @ trial
            CSPfeat = np.log(np.var(CSPfeat, axis = 1))
            features += [CSPfeat]
            # for filt in filters:
            #     CSPfeat = filt @ trial    #Implementation of CSP
            #     tr_features.append(np.log(np.var(CSPfeat)))
            # features += [tr_features]
            # tr_features=[]
        
        return features
    
    
    def FB_thread(self, data, labels, i, Low_Cut_Off, Filters, Features, bps):
        self.bp = np.array([Low_Cut_Off, Low_Cut_Off + 4])
        data_bp = self.BP_Filter(data)

        
        Filters[i] = self.CSP_Filters(data_bp,labels)
        Features[i] = self.CSP_Features(data_bp)
        bps[i] = self.bp
    
    def Filter_Bank(self, data, labels):
        k = self.FB_CSP

        Features = []
        Filters = []
        bps = []
        threads = []
        self.red_type = 'CSP'
        self.components = data.shape[1]
#  4,32,3  4  1,32,4 
        for i in self.bank[0]:
            
            self.bp = np.array([i,i+self.bank[1]])
            data_bp = self.BP_Filter(data)

            self.CSP_Filters(data_bp,labels)
            Features += [self.CSP_Features(data_bp),]
            Filters  += [self.filters,]
            bps += [self.bp,]
     
        Filters = np.array(Filters)
        self.FB_filters = Filters.reshape((Filters.shape[0]*Filters.shape[1],Filters.shape[2]))
        Features = np.array(np.transpose(Features,(1,0,2)))
        features = Features.reshape(Features.shape[0],Features.shape[1]*Features.shape[2])
        I = mi(features,labels)
#        I = MIBIF(features,labels,4)
        sel_feat = np.array(sorted(((value, index) for index, value in enumerate(I)), reverse=True))
        sel_feat = sel_feat.astype(int)[:,1]
        

        for i in range(k):
            if sel_feat[i]%2 == 0 and (sel_feat[i]+1) not in sel_feat[:k]:
                sel_feat = np.insert(sel_feat,k,sel_feat[i]+1)
                k +=1
            elif sel_feat[i]%2 != 0 and (sel_feat[i]-1) not in sel_feat[:k]:
                sel_feat = np.insert(sel_feat,k,sel_feat[i]-1)
                k +=1
        self.FB_filters = self.FB_filters[sel_feat[:k]]
        index=(sel_feat[:k]/self.ChN).astype(int)
        bps=np.array(bps)
        self.bps = bps[index]

        
        return self
        
    def FDA_Filters(self, features, labels):
        
        """
        Functionality
        ------------
        Extracts the CSP filters fron given data
        
        Parameters
        ------------
        data : training data, array, shape(trials, channels, samples)    
        labels : training labels, list
        
        Returns
        ------------
        filters:CSP filters, array, shape(channels, channels)
        """
        
        ClsN = self.ClsN
        
        sh=features.shape
        ftrmean = np.mean(features, axis=1)
        
        SB = np.zeros((sh[0],sh[0]))
        
        for i in range(0,ClsN):
            idxs = np.where(labels==i+1)
            clsFtrs = features[:,idxs[0]]
            clsMean = np.mean(clsFtrs, axis=1)-ftrmean
            clsMean=np.transpose(np.array([clsMean]))
            idxs = np.array(idxs)
            SB = SB + len(idxs[0,:]) * (clsMean @(clsMean.T))
                            
        SW = np.dot(features, np.transpose(features)) - SB
        if na.matrix_rank(SW)<sh[0]:
            SW = SW + 1e-6 * np.eye(sh[0])
        a,b = sla.eigs(np.dot(na.inv(SW),SB),ClsN-1)
        #b=np.real(b)
        self.FDAW = b
        return self
    
    def FDA_Features(self, features):
        
        features = ((self.FDAW.T) @ (features.T)).T  
        
        return features

def select_features(features, labels, k):
    I = mi(features,labels)
#   I = MIBIF(features,labels,4)
    sel_feat = np.array(sorted(((value, index) for index, value in enumerate(I)), reverse=True))
    sel_feat = sel_feat.astype(int)[:,1]
    

    for i in range(k):
        if sel_feat[i]%2 == 0 and (sel_feat[i]+1) not in sel_feat[:k]:
            sel_feat = np.insert(sel_feat,k,sel_feat[i]+1)
            k +=1
        elif sel_feat[i]%2 != 0 and (sel_feat[i]-1) not in sel_feat[:k]:
            sel_feat = np.insert(sel_feat,k,sel_feat[i]-1)
            k +=1
    
def MIBIF( Features, labels, k):
    TrN = Features.shape[0]
    
    I = []
    Hw = 0
    
    for w in [1,2]:
        Iw = np.where(labels == w)
        nw = len(Iw[0])
        pw = nw/TrN
        Hw += -pw*np.log2(pw)
   
    for fj in Features.T:
        stdev = np.std(fj)
        h_w_fj = 0
        for w in [1,2]:
            for fji in fj:
                p_fji = 0
                for label in [1,2]:
                    p_fji_w = 0
                    I_label = np.where(labels == label)
                    n_label = len(I_label[0])
                    p_label = n_label/TrN
                    for fjk in fj[I_label]:
                        p_fji_w += KDE(TrN,stdev,fji-fjk)
                    p_fji_w = p_fji_w/n_label
                    p_fji += p_fji_w * p_label
                p_w_fji = p_fji_w * p_label / p_fji
                h_w_fj += p_w_fji * np.log2(p_w_fji)

        I.append(h_w_fj-Hw)
    return I
                 
            

def KDE(TrN,stdev,feature):
    
    h = 1/np.log2(TrN)#(4/(3*TrN))**(1/5)*stdev
    fi = np.exp(-(feature**2)/(2*(h**2))) /np.sqrt(2*np.pi)  
    return fi     
        

    
def covarianceMatrix(A):
    Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
    return Ca

# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Rx,Rsum):
    
    not_Rx = Rsum-Rx
    E,U = la.eig(Rsum)
    
    # CSP requires the eigenvalues E and eigenvector U be sorted in descending order
    order = np.argsort(E)[::-1]
    E = E[order]
    U = U[:,order]

    
    # Find the whitening transformation matrix
    P = np.sqrt(la.inv(np.diag(E))) @ (U.T) 
    
    # The mean covariance matrices may now be transformed
    Sa = P @ (Rx @ (P.T)) 
    Sb = P @ (not_Rx @ (P.T))
    
    # Find and sort the generalized eigenvalues and eigenvector
    E1,U1 = la.eig(Sa)
    order = np.argsort(E1)[::-1]
    E1 = E1[order]
    U1 = U1[:,order]

    # The projection matrix (the spatial filter) may now be obtained
    SFa = (U1.T) @ P
    
    return SFa.astype(np.float32).real
