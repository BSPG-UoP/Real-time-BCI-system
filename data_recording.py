# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:52:22 2020

@author: Kokkinos
"""

"""Example program to show how to read a multi-channel time series from LSL."""
import numpy as np
import scipy.io as sio
from pylsl import StreamInlet, resolve_stream
from tkinter import *

from graphics import Graphics

class data_rec(Graphics):
    
    def __init__(self):
        
        Graphics.__init__(self)
                
        with open("visual_cues.txt") as f:
            content = f.readlines()
            content = [line.rstrip('\n') for line in content]
        
        self.Fs = int(content[0]) # Sampling Frequency
        self.Rdur = int(content[1]) # Rest visual cue duration
        self.IMdur = int(content[2]) # Imaginary Movement visual cue duration
        self.Prdur = int(content[3])

        del content[0:4]
        self.content = np.array(content)
        
        self.vcN = len(self.content) # number of visual cues
        idxs = np.where(self.content == 'REST')
        self.RN = len(idxs[0]) # number of REST visual cues
        idxs = np.where(self.content == 'PREPARE')
        self.PRN = len(idxs[0])
        self.IMN = len(self.content) - self.RN - self.PRN # number of Imaginary Movements visual cues
        self.recdur = self.RN * self.Rdur * self.Fs + self.IMN * self.IMdur * self.Fs + self.PRN * self.Prdur * self.Fs # duration of the recording
        
        
    
    def begin_stream(self):
        
        print("looking for an EEG stream...")
        self.streams = resolve_stream('type', 'EEG')
        
        # create a new inlet to read from the stream
        self.inlet = StreamInlet(self.streams[0])
        
    def record_loop(self):
        
        cSTR = 0
        cVC = 0
        cdur = 0
        dur = self.Prdur
        buffer = []
        
        while cSTR < self.recdur:
    
            sample, timestamp = self.inlet.pull_sample()
            buffer += [sample,]
        
            if cdur % (dur * self.Fs) == 0:
                
                if self.content[cVC] == 'REST':
                    print("REST")
                    self.delete_all()
                    cdur = 0
                    dur = self.Rdur
                    cVC = cVC+1
                    
                elif self.content[cVC] == 'LEFT':
                    print("LEFT")
                    self.left_arrow()
                    cdur = 0
                    dur = self.IMdur
                    cVC = cVC+1
                    
                elif self.content[cVC] == 'RIGHT':
                    print("RIGHT")
                    self.right_arrow()
                    cdur = 0
                    dur = self.IMdur
                    cVC = cVC+1
                    
                elif self.content[cVC]=='PREPARE':
                    self.Concentration_Cross()
                    cdur = 0
                    dur = self.Prdur
                    cVC = cVC+1
                    
            cSTR = cSTR + 1
            cdur = cdur + 1
        
        return buffer


    def save_recording(self, buffer):
    
        LABELS = []
        trig = []
        offset = self.IMdur * self.Fs
        IM_interval = 2* self.IMdur * self.Fs
        
        idxs=np.where(self.content=='REST')
        self.content = np.delete(self.content,idxs)
        idxs=np.where(self.content=='PREPARE')
        self.content = np.delete(self.content,idxs)
        
        for i, visual_cue in enumerate(self.content):
            
            if visual_cue == 'LEFT':
                LABELS += [1,]
                trig += [offset + i * IM_interval,]
            else:
                LABELS += [2,]
                trig += [offset + i * IM_interval,]
                
        trig = np.array(trig)
        LABELS = np.array(LABELS)
        buffer = np.array(buffer)
        
        # create matlab files
        sio.savemat('trig.mat', {'trig':trig})
        sio.savemat('rec.mat', {'rec':buffer})
        sio.savemat('LABELS.mat', {'LABELS':LABELS})

rec = data_rec()
rec.begin_stream()
buffer = rec.record_loop()
rec.save_recording(buffer)
