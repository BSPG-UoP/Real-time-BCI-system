# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:07:36 2019

@author: Κόκκινος
"""

import numpy as np
from scipy import signal

class Pre_processing:
    
    def __init__(self, Fs=None,  filtering=None, bp = None):
        
        self.Fs = Fs
        
        try:
            self.bp = np.array(bp)
        except:
            self.bp = bp
        
        self.filtering = filtering
        if self.filtering != 'notch' and self.filtering != 'bandpass' and self.filtering != 'full' and self.filtering != None:
            raise ValueError("Wrong filtering input")

    
    def Notch_Filter(self, data):
        
        """
        Funcionality
        ------------
        Implements a 50Hz Notch filter
        
        Arguments
        ------------
        data : raw_recording to be filtered, array, shape(samples,channels) or (trials,channels,samples) 
        
        Returns
        ------------
        data : filtered_data, array, shape(same as input)
        """
    
        data = np.array(data)

        #Filter design
        
        bp_notch_stop = np.array([49.0, 51.0])/(self.Fs/2)
        order = 2
        b,a = signal.butter(order,bp_notch_stop, 'bandstop')

        if data.ndim != 2 and data.ndim != 3:             
            raise ValueError("Invalid input data dimensions")
        
        #Filter implementation
        
        if data.ndim == 2:   
            
            ChN = data.shape[1]
            for channel in range(ChN):
                data[:,channel] = signal.filtfilt(b,a,data[:,channel])
        
        else:
            
            ChN = data.shape[1]
            TrN = data.shape[0]
            for trial in range(TrN):
                for channel in range(ChN):
                    data[trial,channel] = signal.filtfilt(b,a,data[trial,channel]) 
        
        return data
    
    def BP_Filter(self, data):
        
        """
        Funcionality
        ------------
        Implements a bandpass butter filter
        
        Parameters
        ------------
        data : raw_recording to be filtered, array, shape(samples,channels) or (trials,channels,samples) 
        
        Returns
        ------------
        data : filtered_data, array, shape(same as input)
        """
        
        data = np.array(data)

        #Filter design
        
        bp = self.bp/(self.Fs/2) 
        order = 4
        b,a = signal.butter(order,bp,btype='bandpass')
        
        if data.ndim != 2 and data.ndim != 3:  
            
            raise ValueError("Invalid input data dimensions")
        
        #Filter implementation
        
        if data.ndim == 2:   
            
            ChN = data.shape[1]
            for channel in range(ChN):
                data[:,channel] = signal.filtfilt(b,a,data[:,channel])
        
        else:
            
            ChN = data.shape[1]
            TrN = data.shape[0]
            for trial in range(TrN):
                for channel in range(ChN):
                    data[trial,channel] = signal.filtfilt(b,a,data[trial,channel])

        return data
        

            
            
            
            
            
            
            