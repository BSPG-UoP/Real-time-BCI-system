# -*- coding: utf-8 -*-
"""
A complete, offline BCI transducer with implementation of CSP filters
It includes all three phases of a BCI system, namely:
    1)pre-processing
    2)feature extraction
    3)classification

It can be used for a full offline system, only training mode, or for any partial functionality

Functions:
-Filtering
-Epochs
-TrnN_Trials
-CSP_filters
-Features
-Classification

@author: Kokkinos
"""



import numpy as np
from scipy import signal
import numpy.linalg as na
import scipy.linalg as la

from sklearn.metrics import accuracy_score , confusion_matrix

from .pre_processing import Pre_processing as pre
from .feat_extraction import Feature_Extraction as feat
from .classification import Classification as clsf

from Lib.warnings import warn
#import time


#thelw self:classes,filters,features kai isws ChN klp.
#na allaksw pws briskw ChN klp(shape[])


class Transducer(pre, feat, clsf):
    
    def __init__(self, Fs=None, filtering=None, bp = None, components = 4, 
                 red_type = None, FB_CSP = None, bank = None, neighbors = 5):
               
        
        pre.__init__(self, Fs, filtering, bp)
        feat.__init__(self, components, red_type, FB_CSP, bank)
        clsf.__init__(self, neighbors = 5)
        
        if self.filtering is not None and FB_CSP is not None:
            
            warn('Filtering and FB_CSP cannot be applied simoultaneously!', Warning)
            inputt = input("Choose parameter to set to None(filtering/FB_CSP):")
            if inputt != 'filtering' and inputt != 'FB_CSP':
                self.filtering = None
                print("Wrong input, filtering set to None")
            elif inputt == 'filtering':
                self.filtering = None
            else:
                self.FB_CSP = None
                        
    def fit(self, data, labels):
        
        
        #filtering
        
        if self.filtering == 'notch':
            data = self.Notch_Filter(data)
        elif self.filtering == 'bandpass':
            data = self.BP_Filter(data)
        elif self.filtering == 'full':
            data = self.Notch_Filter(data)
            data = self.BP_Filter(data)
        
        #CSP filters and features
        
        if self.FB_CSP != None:
            self.Filter_Bank(data,labels)
            features = []
            for bp,filt in zip(self.bps,self.FB_filters):
                self.bp = bp
                data_bp = self.BP_Filter(data)
                self.filters = filt
                features += [self.CSP_Features(data_bp),]
            features = np.array(features).T[0]
        else:
            self.CSP_Filters(data,labels)
            features = self.CSP_Features(data)
            
#        if self.FB_CSP :
#            features = []
#            for bp,filt in zip(self.bps,self.FB_filters):
#                self.bp = bp
#                data_bp = self.BP_Filter(data)
#                self.filters = filt
#                features += [self.CSP_Features(data_bp),]
#            features = np.array(features).T[0]
#        else:
#            features = self.CSP_Features(data)        
               
        #Feature extraction and dimensionality reduction
        

        
        if self.red_type == 'FDA':
            self.FDA_Filters(features.T, labels)
            features = self.FDA_Features(features)
                    
        #Classifier training
        
        self.Clsf_Fit(features,labels)
        
        return self
    
    def predict(self, data):
        
#        start_time = time.time()
        
        
        #Notch filtering
        
        if self.filtering == 'notch':
            data = self.Notch_Filter(data)
        elif self.filtering == 'bandpass':
            data = self.BP_Filter(data)
        elif self.filtering == 'full':
            data = self.Notch_Filter(data)
            data = self.BP_Filter(data)
        
        #Feature extraction and dimensionality reduction
        
        if self.FB_CSP != None:
            features = []
            for bp,filt in zip(self.bps,self.FB_filters):
                self.bp = bp
                data_bp = self.BP_Filter(data)
                self.filters = filt
                features += [self.CSP_Features(data_bp),]
            features = np.array(features).T[0]
                
        else:
            features = self.CSP_Features(data)
                
        if self.red_type == 'FDA':
            features = self.FDA_Features(features)
        
        #Classifier predictions
        
        pred = self. Clsf_Predict(features)
        
#        print(time.time() - start_time)
        
        return pred
    
    def visualization():
        pass
    
if __name__ == '__main__':
    pass
    

        
    

