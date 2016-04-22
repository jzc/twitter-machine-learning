import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import json

NaiveP = [[],[],[]]
LogP = [[],[],[]]
SvcP = [[],[],[]]

file = open(r".\logistic_results.json", "r")
results = []
for line in file:
    parsed = json.loads(line)
    results.append(parsed)

xlabels = np.arange(1,6)
ylabels = np.arange(1,11)
z = np.array([result["average"]["recall"] for result in results])
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