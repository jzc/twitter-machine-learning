import json
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import pearsonr

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
x = range(1,7)
print(pearsonr([d["percent"] for d in official_data], [d["percent"] for d in classified_data])[0]**2)
plt.plot(x, [d["percent"] for d in official_data], "r", label="CDC")
plt.plot(x, [d["percent"] for d in official_data], "ro")
plt.plot(x, [d["percent"] for d in classified_data], "b", label="Model")
plt.plot(x, [d["percent"] for d in classified_data], "bo")
plt.ylabel("Percent covered")
plt.xticks(x, ["Sep","Oct","Nov","Dec","Jan","Feb"], rotation="45")
plt.legend()
plt.show()