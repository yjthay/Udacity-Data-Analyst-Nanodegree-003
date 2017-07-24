import os
import pandas

from os import walk

directory = 'C:/Users/YJ/Documents/1) Learning/Udacity - Data Analyst/Submissions/003/baseballdatabank-master/core'
#os.listdir() 
myfile = []

for root, dirs, files in os.walk(directory):
    for f in files:
        myfile.append(os.path.join(root,f))

