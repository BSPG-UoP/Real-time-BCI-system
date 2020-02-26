# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:06:40 2020

@author: Kokkinos

lines for telnet communication: 31,32,136,139,149,152,201,204,212,215,296
"""
from threading import Thread

import numpy as np
import scipy.io as sio
from pylsl import StreamInlet, resolve_stream
from tkinter import *
import telnetlib
import pickle
import threading

from graphics import Graphics

class rt(Graphics):
    
    def __init__(self, mode = 'IMvsall', tim_window = 4, vote_window = 4, overlap = 0, 
                 IM_window = 2, HOST = "192.168.4.1"):
        
        if mode == 'IMvsall' or 'Rvsall' or 'IMvsRest' or 'CSP_OVR' or'sync':
            self.mode = mode
        else:
            raise ValueError('Inappropriate mode value')
        
#        self.HOST = HOST
#        self.tn = telnetlib.Telnet(self.HOST)
        
        with open("visual_cues.txt") as f:
            content = f.readlines()
            content = [line.rstrip('\n') for line in content]
        
        self.Fs = int(content[0]) # Sampling Frequency
        self.Rdur = int(content[1]) # Rest visual cue duration
        self.Prdur = int(content[3])

        content = np.array(content)
        self.vcN = len(content) # number of visual cues
        idxs = np.where(content == 'REST')
        self.RN = len(idxs[0]) # number of REST visual cues
        idxs = np.where(content == 'PREPARE')
        self.PRN = len(idxs[0])
        self.IMN = len(content) - self.RN - self.PRN - 4 # number of Imaginary Movements visual cues

        try:
            self.IMdur = int(content[2])
            self.recdur = self.RN * self.Rdur * self.Fs + self.IMN * self.IMdur * self.Fs + self.PRN * self.Prdur * self.Fs # duration of the recording
        except:
            IMdur = list(content[2].split(','))
            self.IMdur = [int(i) for i in IMdur]
            self.IMdur = [np.random.randint(IMdur[0],IMdur[1]) for i in range(self.IMN)]
            self.recdur = self.RN * self.Rdur * self.Fs + sum(self.IMdur) * self.Fs + self.PRN * self.Prdur * self.Fs # duration of the recording

        self.content = np.delete(content,np.s_[:4])
        
        

        if self.mode == 'sync':
            
            self.tim_window = self.IMdur * self.Fs 
            self.vote_window = self.IMdur * self.Fs
            self.step = (self.IMdur + self.Prdur + self.Rdur) * self.Fs
            self.IM_window = 1
            self.IMdur = [self.IMdur] * self.IMN
        
        else:
            
            self.tim_window = tim_window * self.Fs
            self.vote_window = vote_window * self.Fs
            self.overlap = overlap
            self.step = int(self.tim_window * (1 - self.overlap))
            self.IM_window = IM_window
        
        Graphics.__init__(self)
        
    def load_bcis(self, filename):
        with open(filename, 'rb') as train:
            self.bcis = pickle.load(train)
        return self
    
    def begin_stream(self):
        
        print("looking for an EEG stream...")
        self.streams = resolve_stream('type', 'EEG')
        
        # create a new inlet to read from the stream
        self.inlet = StreamInlet(self.streams[0])
    
    def pred_im(self, chunk, cSTR):
    
        self.pred = []
        
        chunk = (np.array(chunk).T)/1000000000
        chunk = chunk.reshape((1,chunk.shape[0],chunk.shape[1]))
               
        for i, bci in enumerate(self.bcis[:-1]):
                
            self.pred.append(bci.predict(chunk))
        
            if self.pred[i] != 0:
                self.vote[i] += 1
            else:
                self.vote[i] -= 1
                    
        self.pred.append(self.bcis[-1].predict(chunk))

        if self.pred[-1] == 1:
            self.vote[-1] += 1
        else:
            self.vote[-1] -= 1
            
        if cSTR % self.vote_window == 0:
            self.pred_decision()
            
            
        
    def pred_decision(self):
        
        if self.mode == 'IMvsall' or self.mode == 'IMvsRest':        
        
            if self.vote[0] <= 0 and self.vote[1] <= 0:
                
                self.prediction.extend([0])
                print("pred:rest")
                
            elif self.vote[0] > 0:
                
                self.prediction.extend([1])
                
                if self.begin:
#                    self.tn.write(('1').encode('ascii'))
                    self.begin = False
                else:
#                    self.tn.write(('4').encode('ascii'))
                    self.cIM += 1
                    
                print("pred:left")
                
            else:
                
                self.prediction.extend([2])
                
                if self.begin:
#                    self.tn.write(('1').encode('ascii'))
                    self.begin = False
                else:
#                    self.tn.write(('3').encode('ascii'))
                    self.cIM += 1
                    
                print("pred:right")
                
        elif self.mode == 'Rvsall':
            
            if self.vote[0] <= 0:
                
                self.prediction.extend([0])
                print("pred:rest")
                
            elif self.vote[1] > 0:
                
                self.prediction.extend([1])
                
                if self.begin:
                    self.tn.write(('1').encode('ascii'))
                    self.begin = False
                else:
                    self.tn.write(('4').encode('ascii'))
                    self.cIM += 1
                    
                print("pred:left")
           
            else:
                
                self.prediction.extend([2])
                
                if self.begin:
                    self.tn.write(('1').encode('ascii'))
                    self.begin = False
                else:
                    self.tn.write(('3').encode('ascii'))
                    self.cIM += 1
                    
                print("pred:right")
                
        else:
  
            self.prediction.extend([self.pred[-1]])
            
            if self.pred[-1] == 0:
                
                print("pred:rest")
                
            elif self.pred[-1] == 1:
                
                if self.begin:
#                    self.tn.write(('1').encode('ascii'))
                    self.begin = False
                else:
#                    self.tn.write(('4').encode('ascii'))
                    self.cIM += 1
                    
                print("pred:left")
                
            else:
                
                if self.begin:
#                    self.tn.write(('1').encode('ascii'))
                    self.begin = False
                else:
#                    self.tn.write(('3').encode('ascii'))
                    self.cIM += 1
                    
                print("pred:right")
                
        self.vote = [0] * len(self.vote)
            
    def main_loop(self):
        
        self.load_bcis('train')
        
        self.vote = [0,0,0]
        self.pred = []
        self.prediction = []
        self.begin = True
        self.cIM = 0
        
        cSTR = 0
        cVC = 0
        cdur = 0
        cIMdur = 0
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
                    try:
                        dur = self.IMdur[cIMdur]
                        cIMdur += 1
                    except:
                        dur = self.IMdur
                    cVC = cVC+1
                    
                elif self.content[cVC] == 'RIGHT':
                    print("RIGHT")
                    self.right_arrow()
                    cdur = 0
                    try:
                        dur = self.IMdur[cIMdur]
                        cIMdur += 1
                    except:
                        dur = self.IMdur
                    cVC = cVC+1
                    
                elif self.content[cVC]=='PREPARE':
                    self.Concentration_Cross()
                    cdur = 0
                    dur = self.Prdur
                    cVC = cVC+1
            
            if cSTR > 0 and cSTR % self.step == 0: #and self.cIM == 0:
                
                t1 = threading.Thread(target = self.pred_im, args=(buffer[-self.tim_window:],cSTR,))
                t1.start() 
                    
#            elif cSTR > 0 and cSTR % self.step == 0:
#                
#                if self.cIM == self.IM_window:
#                    self.cIM = 0
#                else:
#                    self.cIM += 1
        
            cSTR = cSTR + 1
            cdur = cdur + 1
        
#        self.tn.write(('0').encode('ascii'))
        
        return buffer
    
    def save_recording(self, buffer):
        
        LABELS = []
        trig = []
        offset = (self.Rdur + self.Prdur) * self.Fs
        trig += [offset,]
        
        idxs=np.where(self.content=='REST')
        self.content = np.delete(self.content,idxs)
        idxs=np.where(self.content=='PREPARE')
        self.content = np.delete(self.content,idxs)


        try:
            for i, IMdur in enumerate(self.IMdur):

                trig += [IMdur * self.Fs + offset + trig[-1],]
                LABELS += [0] * offset
                if self.content[i] == 'LEFT':
                    LABELS += [1] * IMdur * self.Fs
                else:
                    LABELS += [2] * IMdur * self.Fs
        except:
            for i, visual_cue in enumerate(self.content):

                trig += [self.IMdur * self.Fs + offset + trig[-1],]
                LABELS += [0] * offset
                if visual_cue == 'LEFT':
                    LABELS += [1] * self.IMdur * self.Fs
                else:
                    LABELS += [2] * self.IMdur * self.Fs
                
        LABELS += [0] * self.Rdur * self. Fs
        pred = [[pr] * self.vote_window for pr in self.prediction]
            
                
        trig = np.array(trig)
        trig = np.delete(trig,-1)
        LABELS = np.array(LABELS)
        buffer = np.array(buffer)
        pred = np.array(pred).flatten()
        
        # create matlab files
        sio.savemat('trig.mat', {'trig':trig})
        sio.savemat('rec.mat', {'rec':buffer})
        sio.savemat('LABELS.mat', {'LABELS':LABELS})
        sio.savemat('pred.mat', {'pred':pred})


if __name__ == '__main__':
    b_c_i = rt(mode = 'CSP_OVR', tim_window = 4, vote_window = 8, overlap = 0.5,IM_window = 0)
    b_c_i.load_bcis('train')
    b_c_i.begin_stream()
    buffer = b_c_i.main_loop()
    b_c_i.save_recording(buffer)

    
        
    
