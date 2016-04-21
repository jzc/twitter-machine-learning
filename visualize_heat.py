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
        
xlabels = np.arange(1,6)
ylabels = np.arange(1,11)
z = np.array(LogP[2])
z.resize(len(ylabels), len(xlabels))
fig, ax = plt.subplots()

#plot
heatmap = ax.pcolor(z, cmap=plt.cm.Blues, alpha=0.8)

#turn of frame
ax.set_frame_on(False)

#change label position
ax.set_xticks(np.arange(z.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(z.shape[0]) + 0.5, minor=False)

#set labels
ax.set_xticklabels(xlabels, minor=False)
ax.set_yticklabels(ylabels, minor=False)

#set axis labels
plt.xlabel("ngram size")
plt.ylabel("removed")

#set title
plt.title("Logistic regression accuracy")

#remove ticks
ax = plt.gca()
for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
    
#set colorbar
fig.colorbar(heatmap, cmap=cm.coolwarm)
plt.show()