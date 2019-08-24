

import cv2
import numpy as np
import glob
import os

#merge captured images!! for printing!!

out = None


f = open("y.txt")

y = f.readline()

for i,fname in enumerate(range(4000)):
    fname = "data/" + str(fname) + ".jpg"
    out = cv2.imread(fname)
    print(fname, out.shape,    y[i])
    
    

print( len(y))
