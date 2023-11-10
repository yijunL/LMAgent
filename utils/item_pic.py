import requests
import json
import base64
from PIL import Image
from io import BytesIO
import csv
import numpy as np

csv.field_size_limit(10000000)

def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
        
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=Q2hbfxDGa0t0WebCLQt6AXxg&client_secret=8tQWRlDSKPjbwzxqKhqhpeQB1cafeNyZ"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

def get_base64(prompt):
    """
    get pic (base64) from SD API
    input: prompt
    output: base64_data
    """
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/text2image/sd_xl?access_token=" + get_access_token()
    
    # name = " iPhone 11"
    # description = " The latest smartphone from Apple with a powerful processor and improved cameras"
    # prompt =  name + ", " + description
    payload = json.dumps({
        "prompt": prompt,
        # "negative_prompt": "white",
        "size": "768x768", # "1024x1024",
        "steps": 50, 
        "n": 1,
        "sampler_index": "DPM++ SDE Karras" 
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response)
    base64_data = response.json()["data"][0]["b64_image"]
    # print(base64_data)
    # if "data" in response.json().keys():
    return base64_data
    
def base642img(base64_data):
    """
    show one image
    input: one base64_data
    output: show image
    """
    binary_data = base64.b64decode(base64_data)
    image_stream = BytesIO(binary_data)
    image = Image.open(image_stream)
    image.show()

def base64_csv2img(base64_path, id):
    """
    show image from csv
    input: csv path ;  item_id
    output: show image
    """
    base64_dic = {}
    with open(base64_path, "r", encoding='utf-8', newline="") as file:
        reader = csv.reader(file)
        # next(reader)  # Skip the header line
        for row in reader:
            item_id, one_base64 = row
            base64_dic[int(item_id)] = {
                "base64": one_base64
            }
    
    ids = list(base64_dic.keys())
    print(len(ids))

    test1 = base64_dic[id]
    binary_data = base64.b64decode(test1["base64"])
    image_stream = BytesIO(binary_data)
    image = Image.open(image_stream)
    image.show()

def main():
    file_path = "data/item.csv"
    base64_path = "data/base64.csv"

    # base64_csv2img(base64_path, 659)  # for test
    items = {}
    with open(file_path, "r", encoding='utf-8', newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header line
        for row in reader:
            item_id, title, genre, description, detail = row
            items[int(item_id)] = {
                "name": title.strip(),
                "genre": genre,
                "description": description.strip(),
                "detail": detail.strip(),
                "inter_cnt":0,
                "mention_cnt":0
            }

    # print(items[0])
    continu = 657
    for i in range(len(items.keys())):
        if i > continu:
            one_name = items[i]["name"]
            one_description = items[i]["description"]
            one_prompt = one_name + ", " + one_description
            # print(one_prompt)
            
            one_base64 = get_base64(one_prompt)
            if one_base64 == "":
                continue
            one_csv = [str(i), one_base64]
            with open(base64_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(one_csv)
                file.close()
            print(i)


if __name__ == '__main__':
    main()