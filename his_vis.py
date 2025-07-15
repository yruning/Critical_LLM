import pandas as pd
import re
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_to_json(filename, trajectory):
    """Save the trajectory to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(trajectory, f, indent=4)
        
def extract_number(sentence):
    result = []
    for i in range(len(sentence)):
        if (sentence[i].isdigit() and i ==0 ) or (sentence[i].isdigit() and sentence[max(i-1,0)].isdigit() == False and sentence[max(i-1,0)] != '.' and sentence[max(i-1,0)] != ',') or sentence[i] == '.' and sentence[min(len(sentence)-1,i+1)].isdigit() and sentence[max(0,i-1)].isdigit() is False:
            rs = []
            k = 0
            while sentence[i+k].isdigit() or sentence[i+k] == '.' or sentence[i+k] == ',':
                if sentence[i+k] != ',':
                    rs.append(sentence[i+k])
                if i+k<len(sentence)-1:
                    k = k+1
                else:
                    break
            
            if rs[len(rs)-1] != '.':
                result.append(''.join(rs))
            else:
                result.append(''.join(rs[:len(rs)-1]))
    return result

Letters = ['A','B','C','D','E','F','G','H','I','J','K']

data_name = 'SVAMP'
model_name ='llama33_70b'
i_type = 'remove'
strategy = 'basic'

with open(data_name +'_data/'+data_name+'_'+i_type+'_'+model_name + '_' + strategy +'_eval.json', "r") as f:
    raw_data = json.load(f)
N = np.min((len(raw_data),300))

fp = 0
tp = 0

"""
for i in range(N):
    print(i)
    m = extract_number(raw_data[i]['ambiguous class'])[0]
"""  
for itr in tqdm(range(N)):
    
    c_c = raw_data[itr]['clear class']
    c_m = raw_data[itr]['ambiguous class']
    #print(itr,c_m)
    l_c = float(re.match(r'\d+', extract_number(c_c)[0]).group())
    l_m = float(re.match(r'\d+', extract_number(c_m)[0]).group())
    """
    if l_c == 5:
        print(itr,c_c)
    if l_m == 5:
        print(itr,c_m)
    """
    if l_c == 1 or l_c == 2:
        fp += 1
        #print(itr,c_c, l_c, extract_number(c_c))
    if l_m == 1 or l_m == 2:
        tp += 1
        #print(itr,c_m, l_m, extract_number(c_m))
p = tp/(tp+fp)
r = tp/N
print('Total:', tp,fp, tp/(tp+fp), tp/N, fp/N, 2*p*r/(p+r))

if i_type == 'redun':
    fp_w = 0
    fp_r = 0
    fp_b = 0
    tp_w = 0
    tp_r = 0
    tp_b = 0
    
    for itr in range(N):
        c_c = raw_data[itr]['clear class']
        c_m = raw_data[itr]['ambiguous class']
        #print(itr)
        l_c = float(re.match(r'\d+', extract_number(c_c)[0]).group())
        l_m = float(re.match(r'\d+', extract_number(c_m)[0]).group())
        if l_c == 1 or l_c == 2:
            if itr < int(N/3):
                fp_r += 1
            else:
                if itr< int(N*2/3):
                    fp_w += 1
                else:
                    fp_b += 1
            #print(itr,c_c, l_c, extract_number(c_c))
        if l_m == 1 or l_m == 2:
            if itr < int(N/3):
                tp_r += 1
            else:
                if itr< int(N*2/3):
                    tp_w += 1
                else:
                    tp_b += 1
    p = tp_w/(tp_w+fp_w)
    r = 3*tp_w/N
    print('Wrong Gas:', tp_w,fp_w, tp_w/(tp_w+fp_w), r, 3*fp_w/N, 2*p*r/(p+r))
    p = tp_r/(tp_r+fp_r)
    r = 3*tp_r/N
    print('Right Gas:', tp_r,fp_r, tp_r/(tp_r+fp_r), r, 3*fp_r/N, 2*p*r/(p+r))
    p = tp_b/(tp_b+fp_b)
    r = 3*tp_b/N
    print('Both Gas:', tp_b,fp_b, tp_b/(tp_b+fp_b), r, 3*fp_b/N, 2*p*r/(p+r))

label_c = Counter(float(extract_number(item['clear class'])[0]) for item in raw_data)
label_m = Counter(float(extract_number(item['ambiguous class'])[0]) for item in raw_data)

if i_type == 'remove':
    all_labels = range(1,6)
else:
    all_labels = range(1,5)

label_c = {k: label_c.get(k, 0) for k in all_labels}
label_m = {k: label_m.get(k, 0) for k in all_labels}

#label_c = {k: v for k, v in label_c.items() if k <= 6}
#label_m = {k: v for k, v in label_m.items() if k <= 6}

plt.figure(figsize=(12, 5))

# First subplot: clear class
plt.subplot(1, 2, 1)
bars1 = plt.bar(label_c.keys(), label_c.values(), color='skyblue')
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('Clear Class Categories')

# Second subplot: ambiguous class
plt.subplot(1, 2, 2)
bars2 = plt.bar(label_m.keys(), label_m.values(), color='orange')
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('Ambiguous Class Categories')
plt.tight_layout()
plt.savefig(data_name+ '_data/'+ data_name + '_'+i_type+'_'+ model_name+'_' + strategy + '_label.png')




    
    