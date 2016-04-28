import json
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import pearsonr
from scipy.optimize import minimize
import numpy as np

classified_dir = r".\classified_tweets"
official_file = r".\official_data.json"

classified_data = []
files = listdir(classified_dir)
files.sort(key=lambda x:x[-7:-5])
for file in files:      
    rel_count = 0
    irr_count = 0
    for line in open(r"{}\{}".format(classified_dir, file), "r", encoding="utf8"):
        parsed = json.loads(line)
        if(parsed["type"] == "REL"):
            rel_count = rel_count + 1
        else:
            irr_count = irr_count + 1
    percent = rel_count / (rel_count + irr_count)
    classified_data.append({"month":parsed["created_at"].split()[1],"type":"official","percent":percent})
    
official_data = [json.loads(line) for line in open(official_file, "r", encoding="utf8")]

#Scale classified data by minimizing square error
f = lambda theta: theta[0] + theta[1] * np.array([d["percent"] for d in classified_data])
g = np.array([d["percent"] for d in official_data])
cost = lambda theta: sum((f(theta) - g)**2)
res = minimize(cost, [0,1], method="BFGS")
print(res.x)

#Show correlation values are the same
print(pearsonr(g, f(res.x))[0]**2)
print(pearsonr(g, f([0,1]))[0]**2)

#Plot data
ax1 = plt.gca()
ax2 = ax1.twinx()
x = range(1,7)
ax1.plot(x, g, "r", label="CDC")
ax1.plot(x, g, "ro")
ax1.plot(x, f([0,1]), "g", label="Model")
ax1.plot(x, f([0,1]), "go")
ax2.plot(x, f([0,1]), "b", label="Scaled Model")
ax2.plot(x, f([0,1]), "bo")
ax2.set_ylim((np.array(ax1.get_ylim()) - res.x[0]) / res.x[1])
ax1.set_ylabel("Percent covered (official)")
ax2.set_ylabel("Percent covered (model)")
plt.xticks(x, ["Sep","Oct","Nov","Dec","Jan","Feb"], rotation="45")
ax1.legend()
ax2.legend()
plt.show()