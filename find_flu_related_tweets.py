import json
import gzip
from optparse import OptionParser
from os import listdir
from os.path import basename

parser = OptionParser()
parser.add_option("-i", "--input", dest="input")
(options, args) = parser.parse_args()

out = open(".\S_{}".format(basename(dir)),"w",encoding="utf8")

for file in listdir(options.input):
    for line in gzip.open(r"{}\{}".format(options.input, file), "rt", encoding="utf8"):
        parsed = json.loads(line)
        if "deleted" not in parsed.keys():
            if parsed["lang"] == "en" and  (" flu " in parsed["text"] or " influenza " in parsed["text"]):
                print(parsed["created_at"])
                flu_related.append(parsed)
                out.write(json.JSONEncoder().encode(flu_related) + "\n")