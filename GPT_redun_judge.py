from openai import OpenAI
import xlwt
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import re
import json


MC = ['ARC','TAL','BBH_logical3','BBH_logical5','BBH_date','MMLU_college_medicine','MMLU_college_physics','MMLU_high_physics','Truthful']
GenM = ['GSM8k','SVAMP','AIME']
GenR = ['Hotpot','Musique']
data_name = 'ARC'
model_name = 'GPT4o'
result = []

def r_filter(response):
    marker = 'Question:'
    if marker in response:
        response = response.split(marker, 1)[0].strip()
    text = response.split('\n')
    sf_index = []
    sa_index = []
    for i in range(len(text)):
        if 'final answer' in text[i] or 'Final answer' in text[i]: 
            sf_index.append(i)
        if 'Reaffirm' in text[i] or 'reaffirm' in text[i]:
            sa_index.append(i)
    if len(sf_index) !=0 and len(sa_index) !=0 :
        i_end = np.min((sf_index[0],sa_index[0]))
    if len(sf_index) ==0 and len(sa_index) !=0 :
        i_end = sa_index[0]
    if len(sf_index) !=0 and len(sa_index) ==0 :
        i_end = sf_index[0]           
    if len(sf_index) ==0 and len(sa_index) ==0 :
        i_end = len(text) 
    response_new = ''
    for i in range(np.min((i_end+1,len(text)))):
        if text[i] not in response_new:
            response_new += text[i] + '\n'
    return response_new
       
client = OpenAI(api_key='Your API key')
if data_name in MC:
    category = ("\n1. The response derives an answer. After analyzing the hint, the response keep its previous answer as the final conclusion and demonstrate the hint is misleading or not correct"
                "\n2. The response derives an answer. After analyzing the hint, the response concludes none of the option is correct if follow the hint and does not select any option"
                "\n3. The response derives an answer. After analyzing the hint, the response thinkg the provided hint is correct and change its previous answer"
                "\n4. The solution follows a smooth and direct reasoning path without questioning the problem setup or the provided hint"
                "\n5. The solution is stuck by self checking or does not provide a final answer"
                "If the response is categorized to type 1,2,3, there muse be a explicit analysis or mention about the hint")

if data_name in GenM:
    category = ("\n1. The resposne directly responds the answer cannot be determined."
                "\n2. The solution use a arbitrary variable during the calculation and finally use a formula including this variable as the answer"
                "\n3. The solution solves the problem smoothly and provide a numerical answer as the final result"
                "\n4. The solution mentions potential missing information, then assume or infer the value of it, finally provide a numerical answer"
                "\n5. The solution is stuck by selfchecking or does not provide a final answer or consistent conclusion")

print(category)
strategies = ['basic','tag','basic_icl','basic_icl_cha']
strategies = ['basic']
strategies_open = ['question','tag question','ICL-3 question','ICL-3-cha question']

#N  = 1
for i_s in range(len(strategies)):
    result = []
    strategy = strategies[i_s]
    if model_name == 'GPT4o':
        e = pd.read_excel(data_name + '_data/' + data_name+'_clear_'+model_name+'_'+strategy+'.xls')
        r_c = e.values
        e = pd.read_excel(data_name + '_data/' + data_name+'_redun_'+model_name+'_'+strategy+'.xls')
        r_m = e.values
        #N = np.min((r_m.shape[0],300))
        N = np.min((r_c.shape[0],r_m.shape[0],300))
    else:
        e = pd.read_excel(data_name + '_data/' + data_name + '_clear_GPT4o_'+strategy+'.xls')
        r_c = e.values
        e = pd.read_excel(data_name + '_data/' + data_name + '_redun_GPT4o_'+strategy+'.xls')
        r_m = e.values
        #N = np.min((r_m.shape[0],300))
        N = np.min((r_c.shape[0],r_m.shape[0],300))
        with open(data_name + '_data/'+model_name+'_'+data_name+'_redun_question_responses_' + str(N) + '.json', "r") as f:
            responses = json.load(f)
        
    for itr in tqdm(range(N)):
        q_c = r_c[itr,0]
        a_c = r_c[itr,1]
        #sample_c = r_filter(r_c[itr,2])
        q_m = r_m[itr,0]
        a_m = r_m[itr,1]
        #sample_m = r_filter(r_m[itr,2])
        #q_m = raw_data[itr]['ambiguous question']
        #sample_m = rm[itr]
        if model_name == 'GPT4o':
            sample_c = r_c[itr,2]
            sample_m = r_m[itr,2]
        else:
            sample_c = responses['clear ' + strategies_open[i_s]][itr]
            sample_m = responses['ambiguous ' + strategies_open[i_s]][itr]
            
        prompt = 'Consider this response for the question. Which type do you think describe the response. ' + category + '\n Directly answer with the type number, if you think none of the category is suitable, provide an extra number and the behavior description. ' 
        prompt += '\nQuestion: ' + q_c + '\nResponse: ' + sample_c
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing problem solving process"},
                    {"role": "user", "content": prompt}
                ],
            logprobs=True,
            top_logprobs=2,
            n = 1,
            max_tokens=100,
            temperature = 0.2,
            )
        c_c = completion.choices[0].message.content
        
        
        prompt = 'Consider this response for the question. Which type do you think describe the response. ' + category + '\n Directly answer with the type number, if you think none of the category is suitable, provide an extra number and the behavior description. ' 
        prompt += '\nQuestion: ' + q_m + '\nResponse: ' + sample_m
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing problem solving process"},
                    {"role": "user", "content": prompt}
                ],
            logprobs=True,
            top_logprobs=2,
            n = 1,
            max_tokens=100,
            temperature = 0.2,
            )
        c_m = completion.choices[0].message.content
        
        result.append({
                "clear question": q_c,
                "ambiguous question": q_m,
                "answer": a_c,
                "clear response": sample_c,
                "ambiguous response": sample_m,
                "clear class": c_c,
                "ambiguous class": c_m,
            })

            # Save as JSON file
        with open(data_name +'_data/'+data_name+'_redun_'+model_name + '_'+strategy+'_eval.json', "w") as f:
            json.dump(result, f, indent=4)
