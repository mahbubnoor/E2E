#import os
#import locale
#os.environ["PYTHONIOENCODING"] = "utf-8"
#myLocale=locale.setlocale(category=locale.LC_ALL, locale="en_GB.UTF-8")
import json
import sys

scp_file = sys.argv[1] # "/Data/user_mhb/End2End_bopenslr/End2End_prac_openslr_SE_1/data/BOpenSlr_fbank/bopenslr_fbank.scp"
data_in = sys.argv[2] #"/Data/user_mhb/ESPnet/espnet/egs/basr/asr_practice_1/dump/test/deltafalse/data.json"
data_out = sys.argv[3] #"test_data.json"

feature_map={}
with open(scp_file) as f:
    lines = f.readlines()
    for line in lines:
        k,v = line.split()
        #print(k,v)
        feature_map[k] = v

with open(data_in, encoding='utf-8') as json_file:
    data = json.load(json_file)
    for name in data["utts"]:
        for input_feats in data["utts"][name]["input"]:
            #print(input_feats["feat"])
            input_feats["feat"] = feature_map[name]


with open(data_out, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, indent=4)
