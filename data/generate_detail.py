import os
import openai
import json
from tqdm import tqdm
import time
import re
import csv  
from typing import Any, List, Mapping, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
from transformers.generation.utils import GenerationConfig
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json

openai.api_key = "sk-nllqSQZXUcE0KocTB28904481846487bA2FfF82a9039Ca03" #要更换成自已的API KEY
openai.api_base = "https://api.132999.xyz/v1"

prompt = "Please write down 20 specific items  \
from shopping malls, outputting their detailed  \
names, types, and one-sentence descriptions.When  \
the type belongs to multiple categories, separate \
different categories with |. Please output in the following format: \
Item: item name \n \
Type: item types (split with |) \n \
Description: item description in one sentence \n \
Item: item name \n \
Type: item types (split with |) \n \
Description: item description in one sentence \n \
..."

model_path = "/home/liuyijun27/pretrain_models/baichuan-inc/Baichuan2-13B-Chat-4bits"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_path)

def save_data(products, output_file="new_items.csv"):
    # 定义CSV文件的列头和数据  
    name_set = set()
    fields = ["id", "title", "genre", "description"]  
    rows = []  
    pattern = r"\d+\."  
    replacement = ""  
    
    # 生成CSV文件的数据行  
    for i, item in enumerate(products, start=0):  
        title = re.sub(pattern, replacement, item["title"])
        if(title in name_set):
            continue
        row = [i, title, item["genre"], item["description"]]  
        rows.append(row)  
        name_set.add(title)

    # 写入CSV文件  
    with open(output_file, mode="w", newline="") as file:  
        writer = csv.writer(file)  
        writer.writerow(fields)  # 写入列头  
        writer.writerows(rows)  # 写入数据行

def chat(prompt):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content":prompt}
    ]
)
    answer = response.choices[0].message.content
    return answer

def chat_local(prompt):
    messages = []
    messages.append({"role": "user", "content": prompt})
    answer = model.chat(tokenizer, messages)
    return answer


import pandas as pd  
import re
# 读取CSV文件  
df = pd.read_csv('item.csv')  
for index, row in tqdm(df.iterrows(),total=df.shape[0]):  
    if(row['item_detail']==row['title']+ "'s detail"):
        item_name = row['title']
        prompt = f"Given the product name: {item_name}, please provide detailed information about this product, including various product parameters and the specific price information, directly output the detailed information content (no more than 50 words): "
        try:
            print("\nPrompt: ",prompt)
            respones = chat_local(prompt)
            print("Out: ",respones)
            df.iat[index, df.columns.get_loc('item_detail')] = respones.replace("\n","")
        except Exception as e:
            # 捕获其他类型的异常
            print("发生了错误:", e)
            time.sleep(5)

df.to_csv('item.csv', index=False)