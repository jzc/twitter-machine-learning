import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import json

NaiveP = [[],[],[]]
LogP = [[],[],[]]
SvcP = [[],[],[]]


for i, file_name in enumerate(("logistic", "naive_bayes", "SVC")):
    file = open(r".\data\{}_results.json".format(file_name), "r")
    results = []
    for line in file:
        parsed = json.loads(line)
        results.append(parsed)
    #set axis labels
    for j, stat in enumerate(("accuracy", "precision", "recall")):
        xlabels = np.arange(1,11)
        ylabels = np.arange(1,6)
        z = np.array([result["average"][stat] for result in results])
        z.resize(len(ylabels), len(xlabels))
        ax = plt.subplot(3,3,3*i + j + 1)

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

        if j == 0:
            plt.ylabel("ngram size")
        if i == 2:
            plt.xlabel("removed")

        #set title
        plt.title("{} {}".format(file_name, stat))

        #remove ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        #set colorbar
        plt.colorbar(heatmap)
       
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(8.5, 11)
fig.savefig('test2png.png', dpi=100)
plt.show()