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
data_name = 'SVAMP'
model_name = 'dsr1_7b_rT'
#strategy = 'basic'
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

#My            
client = OpenAI(api_key='Your API key')
if data_name in MC:
    category = ("\n1. Explicitly demonstrate the correct answer is not provided in the option. Finally use none of the above as the answer" 
    "\n2. Provide an extra option or propose an extra statement as the final answer, if the response finally use a extra statement which is not provided by the original options or change the content of the original option as the answer after several reasoning steps, still belong to this type. "
    "\n3. Mention none of the option is correct and decide to select the least wrong one as the answer, so the resposne still select one option as the final answer"
    "\n4. The solution follows a smooth and direct reasoning path without questioning the problem setup"
    "\n5. The solution includes some contradiction, but still select one option without any modification as the final answer. The answer could be wrong"
    "\n6. The solution is stuck by selfchecking, finally select an option as the answer or does not get conclusion"
    "We do not care about whether the reasoning path is correct or not, we only analyze how the reasoning path evolves."
    "The provided response is assumed to be complete, for any response does not provide a final result or conclusion should be considered as type 6")

if data_name in GenM:
    category = ("\n1. The final conclusion of the response is the answer can not be determined due to the insufficient information. The response does not provide a final numrical result for the question."
                "\n2. The solution use a arbitrary variable during the calculation and finally use a formula including this variable as the answer"
                "\n3. The solution solves the problem and provide a numerical answer as the final result or use some assumption to provide a numerical result"
                "When the response is long, focus on the final conlcusion, which is near the end of response. If the response provides a numerical value in the final answer, then the response is belong to type 3.")

print(category)
strategies = ['basic','tag','basic_icl','basic_icl_cha']
strategies_open = ['question','tag question','ICL-3 question','ICL-3-cha question']

for i_s in range(len(strategies)):
    result = []
    strategy = strategies[i_s]
    if model_name == 'GPT4o':
        e = pd.read_excel(data_name + '_data/' + data_name+'_clear_'+model_name+'_'+strategy+'.xls')
        r_c = e.values
        e = pd.read_excel(data_name + '_data/' + data_name+'_ambig_'+model_name+'_'+strategy+'.xls')
        r_m = e.values
        #N = np.min((r_m.shape[0],300))
        N = np.min((r_c.shape[0],r_m.shape[0],300))
    else:
        with open(data_name + '_data/'+data_name+'_remove_question.json', "r") as f:
            rq = json.load(f)
        N = len(rq)
        with open(data_name + '_data/'+model_name+'_'+data_name+'_remove_question_responses_' + str(N) + '.json', "r") as f:
            responses = json.load(f)
        
    for itr in tqdm(range(N)):
        if model_name == 'GPT4o':
            q_c = r_c[itr][0]
            q_m = r_m[itr][0]
            a_c = r_c[itr][1]
            sample_c = r_c[itr,2]
            sample_m = r_m[itr,2]
        else:
            
            q_c = rq[itr]['clear question']
            a_c = rq[itr]['answer']
            #sample_c = r_filter(r_c[itr,2])
            q_m = rq[itr]['ambiguous question']
            sample_c = responses['clear ' + strategies_open[i_s]][itr]
            sample_m = responses['ambiguous ' + strategies_open[i_s]][itr]
            
        prompt = 'Consider this response for the question. Which type do you think describe the response. ' + category + '\n Directly answer with the type number, if you think none of the category is suitable, provide an extra number and the behavior description. ' 
        if data_name in MC:
            prompt += '\nQuestion:' + q_c + '\nResponse: ' + sample_c
        else:
            prompt += '\nResponse: ' + sample_c
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
        if data_name in MC:
            prompt += '\nQuestion:' + q_m + '\nResponse: ' + sample_m
        else:
            prompt += '\nResponse: ' + sample_m
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
        with open(data_name +'_data/'+data_name+'_remove_'+model_name + '_'+strategy+'_eval.json', "w") as f:
            json.dump(result, f, indent=4)
