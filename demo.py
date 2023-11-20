import cv2
import time
import gradio as gr
from typing import Dict, List

from simulator import Simulator
from utils.message import Message
from utils.utils import (
    layout_img,
    get_avatar1,
    get_avatar2,
    html_format,
    chat_format,
    rec_format,
    social_format,
    social_history_format,
    round_format,
    user_format,
    highlight_items,
)

def height_list_map(ls):
    return [int(i*(923.0/1104.0)) for i in ls]
def weight_list_map(ls):
    return [int(i*(1448.0/1756.0)) for i in ls]

class Demo:
    """
    the UI configurations of Demonstration
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.simulator = Simulator(config, logger)
        self.round = 0
        self.cur_image = None
        self.cur_log = ""
        self.cur_chat = ""
        self.cur_rec = ""
        self.cur_post = ""
        self.cur_webcast = ""
        self.cur_round = ""
        self.cur_user = ""
        self.play = False
        self.sleep_time = 3
        self.css_path = "./asset/css/styles.css"
        self.init_round_info = '<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 15px; color: #000000; font-weight: bold;">&nbsp;&nbsp; Waiting to start !</div>'

    def init_background(self):
        background = cv2.imread("./asset/img/v_1/background3.png")
        back_h, back_w, _ = background.shape

        small_height_list = [
            350,
            130,
            130,
            130,
            350,
            130,
            350,
            350,
            520,
            520,
            520,
            720,
            720,
            720,
            720,
            900,
            900,
            900,
            900,
            900,
        ]
        small_weight_list = [
            700,
            500,
            850,
            1300,
            350,
            150,
            1000,
            1400,
            100,
            500,
            1000,
            350,
            750,
            1200,
            1500,
            200,
            500,
            850,
            1050,
            1300,
        ]

        small_height_list = height_list_map(small_height_list)
        small_weight_list = weight_list_map(small_weight_list)

        small_coordinate = list(zip(small_height_list, small_weight_list))
        for id in range(20):
            img = cv2.imread(f"./asset/img/v_1/s_{id}.png", cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dsize=None, fx=0.063, fy=0.063)  # 0.063  0.1
            layout_img(background, img, small_coordinate[id])

        return background

    def reset(self):
        # reset simulator
        del self.simulator
        self.simulator = Simulator(self.config, self.logger)
        self.simulator.load_simulator()
        # reset image and text
        background = self.init_background()
        return [
            cv2.cvtColor(background, cv2.COLOR_BGR2RGB),
            "",
            "",
            "",
            "",
            self.init_round_info,
        ]

    def format_message(self, messages: List[Message]):
        _format = [
            {
                "original_content": "",
                "content": "",
                "agent_id": messages[idx].agent_id,
                "action": messages[idx].action,
                "msg_id": idx,
            }
            for idx in range(len(messages))
        ]

        for idx in range(len(messages)):
            _format[idx]["original_content"] = "[{}]: {}".format(
                self.agent_dict[messages[idx].agent_id], messages[idx].content
            )
            _format[idx]["content"] = html_format(messages[idx].content)

        return _format

    def generate_img_once(self, data: List[Dict]):
        background = self.init_background()
        big_height_list = [
            300,
            80,
            80,
            80,  # small-50
            300,
            80,
            300,
            300,
            470,
            470,
            470,
            670,
            670,
            670,
            670,
            850,
            850,
            850,
            850,
            850,
        ]
        big_weight_list = [
            670,
            470,
            820,
            1270,  # small-30
            320,
            120,
            970,
            1370,
            70,
            470,
            970,
            320,
            720,
            1170,
            1470,
            170,
            470,
            820,
            1020,
            1270,
        ]

        icon_height_list = [
            280,
            60,
            60,
            60,  # big-20
            280,
            60,
            280,
            280,
            450,
            450,
            450,
            650,
            650,
            650,
            650,
            830,
            830,
            830,
            830,
            830,
        ]
        icon_weight_list = [
            790,
            590,
            940,
            1390,  # big+120
            440,
            240,
            1090,
            1490,
            190,
            590,
            1090,
            440,
            840,
            1290,
            1590,
            290,
            590,
            940,
            1140,
            1390,
        ]
        big_height_list = height_list_map(big_height_list)
        big_weight_list = weight_list_map(big_weight_list)
        icon_height_list = height_list_map(icon_height_list)
        icon_weight_list = weight_list_map(icon_weight_list)

        big_coordinate = list(zip(big_height_list, big_weight_list))
        icon_coordinate = list(zip(icon_height_list, icon_weight_list))

        for idx in range(len(data)):
            img = cv2.imread(
                "./asset/img/v_1/b_{}.png".format(data[idx]["agent_id"]),
                cv2.IMREAD_UNCHANGED,
            )
            img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
            layout_img(background, img, big_coordinate[data[idx]["agent_id"]])

            if data[idx]["action"] == "SHOPPING":
                img = cv2.imread("./asset/img/v_1/recsys.png", cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
                layout_img(background, img, icon_coordinate[data[idx]["agent_id"]])
            elif data[idx]["action"] == "POST":
                img = cv2.imread("./asset/img/v_1/social.png", cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
                layout_img(background, img, icon_coordinate[data[idx]["agent_id"]])
            elif data[idx]["action"] == "CHAT":
                img = cv2.imread("./asset/img/v_1/chat.png", cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)
                layout_img(background, img, icon_coordinate[data[idx]["agent_id"]])

        return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    
    def get_head_html(self, pic_desc="", color="#000000"):

        html_text = f'<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 15px; color: {color}; font-weight: bold;">{pic_desc}</div>'

        # html_text = ""
        # html_text += (
        #     f'<div style="display: flex; align-items: center; border-radius: 10px; border: 1px solid black; margin-bottom: 10px;">'
        # )
        # # html_text += f'<img src="{avatar}" alt="Transparent Image" style="background-color: transparent; width: 5%; height: 5%; margin-right: 20px; margin-left: 10px; margin-bottom: 5px; border-radius: 5px;">'
        # html_text += f'<p>{pic_desc}</p>'
        # html_text += f'<div style="color: black; padding: 10px; max-width: 80%;"></div></div>'
        
        return html_text

    def generate_text_once(self, data: List[Dict], round: int):
        log = self.cur_log
        chat_log = ""
        rec_log = ""
        social_log = self.cur_post
        round_log = ""
        webcast_log = self.cur_webcast
        for msg in data:
            if(msg["action"] == "POST_HISTORY" and msg["content"]!=""):
                for single_post in msg["content"].split("**&&")[:-1]:
                    social_log += social_history_format(single_post)
            else:
                log += highlight_items(msg["content"].split("**##")[0])
                # log += '\n\n'
                log += "<br><br>"
                if msg["action"] == "CHAT":
                    chat_log = chat_format(msg)
                elif msg["action"] == "SHOPPING":
                    rec_log = rec_format(msg)
                elif msg["action"] == "POST":
                    social_log += social_format(msg)
                elif msg["action"] == "WEBCAST":
                    webcast_log += social_format(msg)
            round_log = self.get_head_html(round_format(round, self.agent_dict[msg["agent_id"]], msg["agent_id"], self.agent_feat_dict),)
            user_log = user_format(self.agent_dict[msg["agent_id"]], msg["agent_id"], self.agent_feat_dict)
        return log, chat_log, rec_log, social_log, webcast_log, round_log, user_log

    def generate_output(self):
        """
        generate new image and message of next step
        :return: [new image, new message]
        """
        self.round = self.round + 1
        for i in range(self.agent_num):
            next_message = self.simulator.one_step(i)
            self.cur_post = ""
            self.cur_webcast = ""
            data = self.format_message(next_message)
            for d in data:
                time.sleep(self.sleep_time)
                img = self.generate_img_once([d])
                log, chat_log, rec_log, social_log, webcast_log, round_log, user_log = self.generate_text_once(
                    [d], self.round
                )
                yield [img, log, chat_log, rec_log, social_log, webcast_log, round_log, user_log]

    def execute_reset(self):
        self.play = False
        (
            self.cur_image,
            self.cur_log,
            self.cur_chat,
            self.cur_rec,
            self.cur_post,
            self.cur_webcast,
            self.cur_round,
            self.cur_user,
        ) = self.reset()
        return (
            self.cur_image,
            self.cur_log,
            self.cur_chat,
            self.cur_rec,
            self.cur_post,
            self.cur_webcast,
            self.cur_round,
            self.cur_user,
        )

    def execute_play(self):
        self.play = True
        self.simulator.play()
        while self.play:
            for output in self.generate_output():
                (
                    self.cur_image,
                    self.cur_log,
                    self.cur_chat,
                    self.cur_rec,
                    self.cur_post,
                    self.cur_webcast,
                    self.cur_round,
                    self.cur_user,
                ) = output
                if self.play:
                    yield self.cur_image, self.cur_log, self.cur_chat, self.cur_rec, self.cur_post, self.cur_webcast, self.cur_round,self.cur_user,
                else:
                    return self.reset()
                time.sleep(self.sleep_time)

    def launch_demo(self):
        with gr.Blocks(theme="soft", title="ConsAgent Demo", css=self.css_path) as demo:
            with gr.Row(elem_classes=["row-container"]):
                with gr.Column(scale=1, min_width=0, elem_classes=["column-container-left"]):
                    with gr.Row(elem_classes=["white-background", "rounded-corners"]):
                        with gr.Row(elem_classes=["right-up-margin","deep-background","rounded-corners"]):
                            round_output = gr.HTML(
                                value=self.init_round_info,
                                elem_classes=[
                                    "roundbox_size",
                                    "textbox",
                                    "textbox-font",
                                    "rounded-corners",
                                    "deep-background",
                                ],
                            )
                        # with gr.Row(elem_classes=["right-up-margin"]):
                        #     round_output = gr.HTML(value=self.get_head_html("&nbsp; Waiting to start !"))
                        with gr.Row():
                            user_output = gr.HTML(
                                value="",
                                show_label=False,
                                elem_classes=[
                                    "userbox_size",
                                    "textbox",
                                    "textbox-font",
                                    "light-background",
                                    "rounded-corners",
                                ],
                            )
                    with gr.Row(elem_classes=["white-background", "rounded-corners"]):
                        with gr.Tab("Chatting"):
                            chat_output = gr.HTML(
                                value="",
                                show_label=False,
                                elem_classes=[
                                    "textbox_size",
                                    "scrollable-textbox",
                                    "textbox-font",
                                    "rounded-corners",
                                ],
                            )
                        with gr.Tab("Moments"):
                            soc_output = gr.HTML(
                                value="",
                                show_label=False,
                                elem_classes=[
                                    "textbox_size",
                                    "scrollable-textbox",
                                    "textbox-font",
                                    "rounded-corners",
                                ],
                            )


                with gr.Column(scale=3, min_width=0, elem_classes=["column-container"]):

                    background = self.init_background()
                    image_output = gr.Image(
                        value=cv2.cvtColor(background, cv2.COLOR_BGR2RGB),
                        label="Demo",
                        show_label=False,
                    )
                    with gr.Row(variant="panel", elem_classes=["button-container"]):
                        with gr.Column(scale=1, min_width=0, elem_classes=["column-container"]):
                            play_btn = gr.Button(
                                "Play",
                                variant="primary",
                                elem_id="play_btn",
                                elem_classes=["btn_font", "btn_size"],
                            )
                        with gr.Column(scale=1, min_width=0, elem_classes=["column-container"]):
                            reset_btn = gr.Button(
                                "Reset",
                                variant="primary",
                                elem_id="reset_btn",
                                elem_classes=["btn_font", "btn_size"],
                            )


                with gr.Column(scale=1, min_width=0, elem_classes=["column-container-right"]):

                    with gr.Row(elem_classes=["white-background", "rounded-corners"]):
                        # with gr.Row(elem_classes=["right-up-margin"]):
                        #     rec_pic = gr.HTML(value=self.get_head_html("Shopping System"))
                        with gr.Tab("Shopping"):
                        # with gr.Row(elem_classes=["margin"]):
                            rec_output = gr.HTML(
                                value="",
                                show_label=False,
                                elem_classes=[
                                    "shopbox_size",
                                    "scrollable-textbox",
                                    "light-background",
                                    "textbox-font",
                                    "rounded-corners",
                                ],
                            )
                        with gr.Tab("Webcast"):
                            web_output = gr.HTML(
                                value="",
                                show_label=False,
                                elem_classes=[
                                    "shopbox_size",
                                    "scrollable-textbox",
                                    "light-background",
                                    "textbox-font",
                                    "rounded-corners",
                                ],
                            )

                    with gr.Row(elem_classes=["white-background", "rounded-corners"]):
                        with gr.Row(elem_classes=["right-up-margin"]):
                            log_pic = gr.HTML(value=self.get_head_html("Logs"))
                        with gr.Row(elem_classes=["margin"]):
                            log_output = gr.HTML(
                                value="",
                                show_label=False,
                                elem_classes=[
                                    "logbox_size",
                                    "scrollable-textbox",
                                    "textbox-font",
                                    "light-background",
                                    "rounded-corners",
                                    # "border",
                                ],
                            )


            play_btn.click(
                fn=self.execute_play,
                inputs=None,
                outputs=[
                    image_output,
                    log_output,
                    chat_output,
                    rec_output,
                    soc_output,
                    web_output,
                    round_output,
                    user_output,
                ],
                show_progress=False,
            )
            reset_btn.click(
                fn=self.execute_reset,
                inputs=None,
                outputs=[
                    image_output,
                    log_output,
                    chat_output,
                    rec_output,
                    soc_output,
                    web_output,
                    round_output,
                    user_output,
                ],
                show_progress=False,
            )

        self.simulator.load_simulator()
        self.agent_num = len(self.simulator.agents.keys())
        self.agent_dict = {
            agent.id: agent.name for id, agent in self.simulator.agents.items()
        }

        self.agent_feat_dict = {
            agent.id: {"age": agent.age,
                       "gender": agent.gender,
                       "traits": agent.gender,
                       "status": agent.status,
                       "interest": agent.interest,
                       "relationships": agent.relationships,
                       "feature": agent.feature,} for id, agent in self.simulator.agents.items()
        }

        demo.style={"background-color": "lightblue"}
        demo.queue(concurrency_count=1, max_size=1).launch(height="100%", width="100%")
