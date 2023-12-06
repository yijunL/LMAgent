import cv2
import base64
import json
import logging
from typing import Any, Dict, Optional, List, Tuple
import re
import itertools
import random
from llm import *
from yacs.config import CfgNode
import os
from langchain.chat_models import ChatOpenAI

COLOR_TAG = 0


# logger
def set_logger(log_file, name="default"):
    """
    Set logger.
    Args:
        log_file (str): log file path
        name (str): logger name
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the 'log' folder if it doesn't exist
    log_folder = os.path.join(output_folder, "log")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Create the 'message' folder if it doesn't exist
    message_folder = os.path.join(output_folder, "message")
    if not os.path.exists(message_folder):
        os.makedirs(message_folder)
    log_file = os.path.join(log_folder, log_file)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


# json
def load_json(json_file: str, encoding: str = "utf-8") -> Dict:
    with open(json_file, "r", encoding=encoding) as fi:
        data = json.load(fi)
    return data


def save_json(
    json_file: str,
    obj: Any,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: Optional[int] = None,
    **kwargs,
) -> None:
    with open(json_file, "w", encoding=encoding) as fo:
        json.dump(obj, fo, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def bytes_to_json(data: bytes) -> Dict:
    return json.loads(data)


def dict_to_json(data: Dict) -> str:
    return json.dumps(data)


# cfg
def load_cfg(cfg_file: str, new_allowed: bool = True) -> CfgNode:
    """
    Load config from file.
    Args:
        cfg_file (str): config file path
        new_allowed (bool): whether to allow new keys in config
    """
    with open(cfg_file, "r") as fi:
        cfg = CfgNode.load_cfg(fi)
    cfg.set_new_allowed(new_allowed)
    return cfg


def add_variable_to_config(cfg: CfgNode, name: str, value: Any) -> CfgNode:
    """
    Add variable to config.
    Args:
        cfg (CfgNode): config
        name (str): variable name
        value (Any): variable value
    """
    cfg.defrost()
    cfg[name] = value
    cfg.freeze()
    return cfg


def merge_cfg_from_list(cfg: CfgNode, cfg_list: list) -> CfgNode:
    """
    Merge config from list.
    Args:
        cfg (CfgNode): config
        cfg_list (list): a list of config, it should be a list like
        `["key1", "value1", "key2", "value2"]`
    """
    cfg.defrost()
    cfg.merge_from_list(cfg_list)
    cfg.freeze()
    return cfg


def extract_item_names(observation: str, action: str = "RECOMMENDER") -> List[str]:
    """
    Extract item names from observation
    Args:
        observation: observation from the environment
        action: action type, RECOMMENDER or SOCIAL
    """
    item_names = []
    if observation.find("<") != -1:
        matches = re.findall(r"<(.*?)>", observation)
        item_names = []
        for match in matches:
            item_names.append(match)
    elif observation.find(";") != -1:
        item_names = observation.split(";")
        item_names = [item.strip(" '\"") for item in item_names]
    elif action == "RECOMMENDER":
        matches = re.findall(r'"([^"]+)"', observation)
        for match in matches:
            item_names.append(match)
    elif action == "SOCIAL":
        matches = re.findall(r'[<"]([^<>"]+)[">]', observation)
        for match in matches:
            item_names.append(match)
    return item_names


def layout_img(background, img, place: Tuple[int, int]):
    """
    Place the image on a specific position on the background
    Args:
        background: background image
        img: the specified image
        place: [top, left]
    """
    back_h, back_w, _ = background.shape
    height, width, _ = img.shape
    for i, j in itertools.product(range(height), range(width)):
        if img[i, j, 3]:
            background[place[0] + i, place[1] + j] = img[i, j, :3]


def get_avatar1(idx, height=100):
    """
    Retrieve the avatar for the specified index and encode it as a byte stream suitable for display in a text box.
    Args:
        idx (int): The index of the avatar, used to determine the path to the avatar image.
    """
    img = cv2.imread(f"./asset/img/v_1/{idx}.png")
    base64_str = cv2.imencode(".png", img)[1].tostring()
    avatar = "data:image/png;base64," + base64.b64encode(base64_str).decode("utf-8")
    msg = f'<img src="{avatar}" style="width: 100%; height: {height}%; margin-right: 50px;">'
    return msg


def get_avatar2(idx):
    """
    Retrieve the avatar for the specified index and encode it as a Base64 data URI.
    Args:
        idx (int): The index of the avatar, used to determine the path to the avatar image.
    """
    img = cv2.imread(f"./asset/img/v_1/{idx}.png")
    base64_str = cv2.imencode(".png", img)[1].tostring()
    return "data:image/png;base64," + base64.b64encode(base64_str).decode("utf-8")

def highlight_items(new_content: str):

    for name in [
        "David",
        "Miller",
        "Smith",
        "Eve",
        "Tommie",
        "Jake",
        "Lily",
        "Alice",
        "Sophia",
        "Rachel",
        "Lei",
        "Max",
        "Emma",
        "Ella",
        "Sen",
        "James",
        "Ben",
        "Isabella",
        "Mia",
        "Henry",
        "Charlotte",
        "Olivia",
        "Michael",
        "Jiaqi Li"
    ]:
        html_span = "<span style=\"color: red;\">" + name + "</span>"
        new_content = new_content.replace(name, html_span)
    new_content = new_content.replace("['", '<span style=\"color: #06A279;\">[\'')
    new_content = new_content.replace("']", "']</span>")
    return new_content

def html_format(orig_content: str):
    """
    Convert the original content to HTML format.
    Args:
        orig_content (str): The original content.
    """
    new_content = orig_content.replace("<", "")
    new_content = new_content.replace(">", "")
    # for name in [
    #     "Eve",
    #     "Tommie",
    #     "Jake",
    #     "Lily",
    #     "Alice",
    #     "Sophia",
    #     "Rachel",
    #     "Lei",
    #     "Max",
    #     "Emma",
    #     "Ella",
    #     "Sen",
    #     "James",
    #     "Ben",
    #     "Isabella",
    #     "Mia",
    #     "Henry",
    #     "Charlotte",
    #     "Olivia",
    #     "Michael",
    # ]:
    #     html_span = "<span style=\"color: red;\">" + name + "</span>"
    #     new_content = new_content.replace(name, html_span)
    # new_content = new_content.replace("['", '<span style=\"color: #06A279;\">[\'')
    # new_content = new_content.replace("']", "']</span>")
    return new_content


# border: 0;
def chat_format(msg: Dict, COLOR_TAG=0):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """
    if(COLOR_TAG==0):
        color = "#FAE1D1"
    else:
        color = "#A0E0FF"

    html_text = ""
    avatar = get_avatar2(msg["agent_id"])
    html_text += (
        f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
    )
    html_text += f'<img src="{avatar}" style="width: 20%; height: 20%; border: solid white; background-color: white; border-radius: 25px; margin-right: 3px;">'
    html_text += f'<div style="background-color: {color}; color: black; padding: 10px; border-radius: 10px; max-width: 75%;font-family: 微软雅黑, sans-serif; font-size: 10px; ">'

    if("**##" in msg["content"]):   # visualization when being recommended or buying or checking.
        raw_text = msg["content"].split("**##")[0]
        pics = msg["content"].split("**##")[1]
        pics = "data:image/png;base64," + pics

        html_text += f'{highlight_items(raw_text)}'
        html_text += f'</div></div><div style="display: flex; justify-content: space-between; margin-bottom: 5px;">'
        html_text += f'<img src="{pics}" style="margin-left: 21%; width: 50%; height: 50%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;"></div>'

        # temp_html = open("tmp.html","w")
        # temp_html.write(html_text)
        # temp_html.close()

    else:
        html_text += f'{highlight_items(msg["content"])}'
        html_text += f"</div></div>"


    return html_text


def rec_format(msg: Dict):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """
    html_text = ""
    avatar = get_avatar2(msg["agent_id"])
    html_text += (
        f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
    )
    html_text += f'<img src="{avatar}" style="width: 20%; height: 20%; border: solid white; background-color: white; border-radius: 25px; margin-right: 3px;">'
    html_text += f'<div style="background-color: #DFEED5; color: black; padding: 10px; border-radius: 10px; max-width: 75%;font-family: 微软雅黑, sans-serif; font-size: 10px; ">'

    if("**##" in msg["content"]):   # visualization when being recommended or buying or checking.

        pic_per_raw = 3
        raw_text = msg["content"].split("**##")[0]
        pro_desc = eval('['+re.findall(r'\[(.*?)\]', raw_text)[0]+']')

        add_margin = "margin-left: 21%;"
        pic_scale = 50
        text_scale = pic_scale
        if("is recommended" in raw_text):
            pic_scale = 100/pic_per_raw-2
            text_scale = pic_scale
            add_margin = ""
            html_text += f'"{highlight_items(raw_text.split("is recommended")[0])}" is recommended:'
        elif("details:" in raw_text):
            text_scale = 80
            pro_desc = eval('['+re.findall(r'\[(.*?)\]', raw_text)[1]+']')
            raw_text = raw_text.split("details:")[0]+"details:"
            html_text += f'{highlight_items(raw_text)}'
        else:
            html_text += f'{highlight_items(raw_text)}'
        html_text += f"</div></div>"


        pics = msg["content"].split("**##")[1].split("##**")[:-1]

        pics = ["data:image/png;base64," + pi for pi in pics]

        html_text += ""
        html_text += (
            f'<div style="display: flex; justify-content: space-between; margin-bottom: 5px;">'
        )
        accu_pic = []
        for i in range(len(pics)):
            html_text += f'<img src="{pics[i]}" style="width: {pic_scale}%;{add_margin} height: {pic_scale}%; border: solid white; background-color: white; border-radius: 25px; margin-right: 5px;">'
            accu_pic.append(i)
            
            if((i+1)%pic_per_raw==0):
                html_text += f"</div>"
                html_text += (
                    f'<div style="display: flex; justify-content: space-between; margin-bottom: 5px;">'
                )
                for ac in accu_pic:
                    html_text += f'<div style="background-color: #D9E8F5;text-align: center; color: black;display: flex; padding: 2px;font-family: 微软雅黑, sans-serif; font-size: 10px; border-radius: 10px; width: {text_scale}%;">{pro_desc[ac]}</div>'
                html_text += f'</div><br><div style="display: flex; justify-content: space-between; margin-bottom: 5px; margin-right: 5px;">'
                accu_pic = []
            
        if(len(accu_pic)!=0):
            html_text += f"</div>"
            html_text += (
                f'<div style="display: flex; justify-content: space-between; margin-bottom: 5px;">'
            )
            for ac in accu_pic:
                html_text += f'<div style="background-color: #D9E8F5;{add_margin}text-align: center; color: black;display: flex; padding: 2px; border-radius: 10px; width: {pic_scale}%; margin-right: 5px;">{pro_desc[ac]}</div>'
            html_text += f"</div>"
            accu_pic = []
            
        # temp_html = open("tmp.html","w")
        # temp_html.write(html_text)
        # temp_html.close()
    else:
        html_text += f'{highlight_items(msg["content"])}'
        html_text += f"</div></div>"


    return html_text


def social_format(msg: Dict):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """
    html_text = "<br>"
    avatar = get_avatar2(msg["agent_id"])
    html_text += (
        f'<div style="display: flex; align-items: center; margin-bottom: 1px;">'
    )
    html_text += f'<img src="{avatar}" style="width: 20%; height: 20%; border: solid white; background-color: white; border-radius: 25px; margin-right: 3px;">'
    html_text += f'<div style="background-color: #DFEED5; color: black; padding: 10px; border-radius: 10px; max-width: 75%;font-family: 微软雅黑, sans-serif; font-size: 10px; ">'
    
    if("**##" in msg["content"]):   # visualization when being recommended or buying or checking.
        raw_text = msg["content"].split("**##")[0]
        pics = msg["content"].split("**##")[1]
        pics = "data:image/png;base64," + pics

        html_text += f'{highlight_items(raw_text)}'
        html_text += f'</div></div><div style="display: flex; justify-content: space-between; margin-bottom: 5px;">'
        html_text += f'<img src="{pics}" style="margin-left: 21%; width: 50%; height: 50%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;"></div>'

        # temp_html = open("tmp.html","w")
        # temp_html.write(html_text)
        # temp_html.close()

    else:
        html_text += f'{highlight_items(msg["content"])}'
        html_text += f"</div></div>"

    return html_text


def social_history_format(msg: str):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """

    user = int(msg.split("**^^")[0])
    content = msg.split("**^^")[1]


    html_text = ""
    avatar = get_avatar2(user)
    html_text += (
        f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
    )
    html_text += f'<img src="{avatar}" style="width: 20%; height: 20%; border: solid white; background-color: white; border-radius: 25px; margin-right: 3px;">'
    html_text += f'<div style="background-color: #DFEED5; color: black; padding: 10px; border-radius: 10px; max-width: 75%;font-family: 微软雅黑, sans-serif; font-size: 10px; ">'
    if("**##" in content):   # visualization when being recommended or buying or checking.
        raw_text = content.split("**##")[0]
        pics = content.split("**##")[1]
        pics = "data:image/png;base64," + pics

        html_text += f'{highlight_items(raw_text)}'
        html_text += f'</div></div><div style="display: flex; justify-content: space-between; margin-bottom: 5px;">'
        html_text += f'<img src="{pics}" style="margin-left: 21%; width: 50%; height: 50%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;"></div>'

        # temp_html = open("tmp.html","w")
        # temp_html.write(html_text)
        # temp_html.close()

    else:
        html_text += f'{highlight_items(msg["content"])}'
        html_text += f"</div></div>"

    return html_text


def round_format(round: int, agent_name: str, agent_id: int, agent_feat: dict):
    """
    Convert the round information to HTML format.
    Args:
        round (int): The round number.
        agent_name (str): The agent name.
    """
    # round_info = ""
    # round_info += f'<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 15px; color: #000000; font-weight: bold;">'
    # round_info += f"{round}  &nbsp;&nbsp;  Actor: {agent_name}"
    # round_info += f"</div>"
    # return round_info
    return f"{round}  &nbsp;&nbsp;  Actor: {agent_name}"

def user_format(agent_name: str, agent_id: int, agent_feat: dict):
    """
    Convert the round information to HTML format.
    Args:
        round (int): The round number.
        agent_name (str): The agent name.
    """
    sex = '♂' if agent_feat[agent_id]["traits"] == 'male' else '♀'

    html_text = ""
    avatar = get_avatar2(agent_id)
    html_text += (
        f'<div style="display: flex; align-items: center; margin-bottom: 3px;">'
    )
    html_text += f'<img src="{avatar}" style="width: 20%; height: 20%; border: solid white; background-color: white; border-radius: 25px; margin-right: 5px;">'
    html_text += f'<div style="color: black; padding: 10px; border-radius: 10px; max-width: 75%;">'
    html_text += f'<div style="text-align: left; font-family: 微软雅黑, sans-serif; font-size: 15px; font-weight: bold;">{agent_name}&nbsp;{sex}</div>'
    # html_text += "<br>"
    html_text += f'<div style="text-align: left; font-family: 微软雅黑, sans-serif; font-size: 10px; ">{agent_feat[agent_id]["status"]}&nbsp;&nbsp;{agent_feat[agent_id]["age"]} years old</div></div></div>'

    # html_text += "<br>"
    html_text += f'<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 10px;"><b>Interest:</b>&nbsp;<p>{agent_feat[agent_id]["interest"]}</p></div>'
    html_text += f'<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 10px;"><b>Feature:</b>&nbsp;<p>{agent_feat[agent_id]["feature"]}</p></div>'
    # html_text += f'<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 10px;"><b>Relationships:</b>&nbsp;<p>{agent_feat[agent_id]["relationships"]}</p></div>'

    # round_info = ""
    # round_info += f'<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 15px; color: #000000; font-weight: bold;">'
    # round_info += f"{round}  &nbsp;&nbsp;  Actor: {agent_name}"
    # round_info += f"</div>"
    # return round_info
    temp_html = open("tmp.html","w")
    temp_html.write(html_text)
    temp_html.close()
    return html_text

def ensure_dir(dir_path):
    """
    Make sure the directory exists, if it does not exist, create it
    Args:
        dir_path (str): The directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_id(dir_name):
    ensure_dir(dir_name)
    existed_id = set()
    for f in os.listdir(dir_name):
        existed_id.add(f.split("-")[0])
    id = random.randint(1, 999999999)
    while id in existed_id:
        id = random.randint(1, 999999999)
    return id


def get_llm(config, logger, api_key):
    """
    Get the large language model.
    Args:
        config (CfgNode): The config.
        logger (Logger): The logger.
        api_key (str): The API key.
    """
    if config["llm"] == "gpt-4":
        LLM = ChatOpenAI(
            max_tokens=config["max_token"],
            temperature=config["temperature"],
            openai_api_key=api_key,
            openai_api_base="https://api.132999.xyz/v1",
            model="gpt-4-1106-preview",
            max_retries=config["max_retries"]
        )
    elif config["llm"] == "gpt-3.5-16k":
        LLM = ChatOpenAI(
            max_tokens=config["max_token"],
            temperature=config["temperature"],
            openai_api_key=api_key,
            openai_api_base="https://api.132999.xyz/v1",
            model="gpt-3.5-turbo-16k",
            max_retries=config["max_retries"]
        )
    elif config["llm"] == "gpt-3.5":
        LLM = ChatOpenAI(
            max_tokens=config["max_token"],
            temperature=config["temperature"],
            openai_api_key=api_key,
            openai_api_base="https://api.132999.xyz/v1",
            model="gpt-3.5-turbo",
            max_retries=config["max_retries"]
        )
    elif config["llm"] == "gpt-3.5-1106":
        LLM = ChatOpenAI(
            max_tokens=config["max_token"],
            temperature=config["temperature"],
            openai_api_key=api_key,
            openai_api_base="https://api.132999.xyz/v1",
            model="gpt-3.5-turbo-1106",
            max_retries=config["max_retries"]
        )
    elif config["llm"] == "custom":
        LLM = CustomLLM(max_token=2048, logger=logger)
        # LLM = CustomLLM()
    return LLM


def is_chatting(agent, agent2):
    """Determine if agent1 and agent2 is chatting"""
    name = agent.name
    agent_name2 = agent2.name
    return (
        (agent2.event.target_agent)
        and (agent.event.target_agent)
        and (name in agent2.event.target_agent)
        and (agent_name2 in agent.event.target_agent)
    )

def get_feature_description(feature):
    """Get description of given features."""
    descriptions = {
        "Watcher": "Enjoy browsing shopping systems.",
        "Explorer": "Like to search for products heard of before, also like check product's detail in the shopping system.",
        "Critic": "Demanding high standards for products, like to criticize the products.",
        "Chatter": "Enjoy chatting with friends, trust friends' recommendations.",
        "Poster": "Enjoy publicly posting on social media and sharing content and insights with more people."
    }
    features = feature.split(";")
    descriptions_list = [descriptions[feature] for feature in features if feature in descriptions]
    return " ".join(descriptions_list)

def count_files_in_directory(target_directory:str):
    """Count the number of files in the target directory"""
    return len(os.listdir(target_directory))

def get_avatar_url(id:int,gender:str,type:str="origin",role=False):
    if role:
        target='/asset/img/avatar/role/'+gender+'/'
        return target+str(id%10)+'.png'
    target='/asset/img/avatar/'+type+"/"+gender+'/'
    return target+str(id%10)+'.png'