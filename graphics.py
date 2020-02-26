# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:52:03 2020

@author: Kokkinos
"""

from tkinter import *

class Graphics():
    
    def __init__(self):
        
        root = Tk()
        root.state('zoomed')
        root.title("Visual Cues")
        root.configure(background = 'black')
        self.root = root
        
        self.w = Canvas(self.root)
        self.w.pack(fill = "both", expand = True)
        self.w.config(bg = 'black')
        
        self.root.update()
        
        self.x = self.w.winfo_width()/2
        self.y = self.root.winfo_height()/2
        self.recx1 = self.x - 200
        self.recx2 = self.x + 200
        self.recy1 = self.y + 25
        self.recy2 = self.y - 25

                
    def update_all(self):
        self.root.update()
        
    def delete_all(self):
        self.w.delete("all")
        self.root.update()
        
    def left_arrow(self):
        self.w.create_rectangle(self.recx1-1, self.recy1, self.x+25, self.recy2, 
                                fill="grey")
        self.w.create_polygon(self.recx1, self.recy2-25, self.recx1-100, 
                              self.y, self.recx1, self.recy1+25, fill="grey")
        self.root.update()
    
    def right_arrow(self):
        self.w.create_rectangle(self.x-25, self.recy1, self.recx2, self.recy2,
                                fill="grey")
        self.w.create_polygon(self.recx2, self.recy1+25, self.recx2+100, 
                              self.y, self.recx2, self.recy2-25, fill="grey")
        self.root.update()
        
    def Concentration_Cross(self):
        self.w.delete("all")
        self.w.create_rectangle(self.recx1+50, self.recy1, self.recx2-50, self.recy2,
                                fill="grey")
        self.w.create_rectangle(self.x-25, self.y+150, self.x+25, self.y-150,
                                fill="grey")
#        self.w.create_oval(self.x+100, self.y+100, self.x-100, self.y-100, fill="grey")
        
        self.root.update()


