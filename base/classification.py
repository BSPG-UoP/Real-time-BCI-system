# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:11:53 2019

@author: Κόκκινος
"""

import numpy.linalg as na

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score , confusion_matrix

class Classification:
    
    def __init__(self, neighbors = 5):
        
        self.neighbors = neighbors


    def Clsf_Fit(self, data, labels):

        KNN = KNeighborsClassifier(n_neighbors = self.neighbors, n_jobs = -1)
        
        KNN.fit(data, labels)
        
        self.KNN = KNN
        
        return self
    
    def Clsf_Predict(self, data):
        
#        prob = self.KNN.predict_proba(data)
        pred = self.KNN.predict(data)
#        if pr[0] != 0 and prob[:,1] >0.6:
#            pred = pr
#        else:
#            pred = 0
        
        return pred
    
        
        
        
        
        
        
