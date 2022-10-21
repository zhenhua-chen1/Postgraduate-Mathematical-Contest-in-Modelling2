#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:00:26 2022

@author: chenzhenhua
"""


import numpy as np
import pandas as pd
from folium import Map
from folium.plugins import HeatMap
 
data=pd.read_excel("热力图.xlsx")
data.dropna(axis=1,how ='any')
data=data.fillna(0)
data=np.array(data)
data[:,2]=1

 
m = Map([ 38., 99.], tiles='stamentoner', zoom_start=2)
 
HeatMap(data).add_to(m)
 
 
m.save('Heatmap.html')#存放路径记得改