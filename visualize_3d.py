import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

NaiveP = [[],[],[]]
LogP = [[],[],[]]
SvcP = [[],[],[]]

file = open(".\classifier_results.txt", "r")
lines = file.readlines()
for i in range(0,len(lines),10):
    current = lines[i:i+10]
    average = 0
    for data in current:
        data = data.split(", ")   
        average = average + float(data[1]) #1: accuracy 2:precision 3:recall     
    average = average/10
    name = data[0].split()
    x,y,name = int(name[2]), int(name[4]), name[0]
    if name == "NaiveBayes":
        NaiveP[0].append(x)
        NaiveP[1].append(y)
        NaiveP[2].append(average)
    elif name == "LogisticRegression":
        LogP[0].append(x)
        LogP[1].append(y)
        LogP[2].append(average)
    elif name == "SVC":
        SvcP[0].append(x)
        SvcP[1].append(y)
        SvcP[2].append(average)
        
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(NaiveP[0], NaiveP[1], NaiveP[2], c="r")
ax.scatter(LogP[0], LogP[1], LogP[2], c="g")
ax.scatter(SvcP[0], SvcP[1], SvcP[2], c="b")

plt.show()