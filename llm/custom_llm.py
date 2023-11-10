from typing import Any, List, Mapping, Optional
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
# from transformers.generation.utils import GenerationConfig
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json

# model_path = "/home/liuyijun27/pretrain_models/ZhipuAI/chatglm2-6b"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cuda')
# model = model.eval()


# model_path = "/home/liuyijun27/pretrain_models/qwen/Qwen-7B-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     bf16=True,
#     trust_remote_code=True
# ).eval()
# model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
# model = model.eval()

# model_path = "/home/liuyijun27/pretrain_models/baichuan-inc/Baichuan2-13B-Chat-4bits"
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained(model_path)

# model_path = "/home/liuyijun27/pretrain_models/ZhipuAI/chatglm3-6b"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cuda')
# model = model.eval()

# model_path = "/home/liuyijun27/pretrain_models/baichuan-inc/Baichuan2-13B-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained(model_path)

class CustomLLM(LLM):
    max_token: int = 2048
    URL: str = "https://api.132999.xyz/v1"
    headers: dict = {"Content-Type": "application/json"}
    payload: dict = {"prompt": "", "history": []}
    logger: Any

    @property
    def _llm_type(self) -> str:
        return "CustomLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # print("*"*20)
        # print("\nPrompt: ",prompt)
        # messages = []
        # messages.append({"role": "user", "content": prompt})
        # response = model.chat(tokenizer, messages)
        # response, history = model.chat(tokenizer, prompt, history=[])
        print("\nResponse: ",response)
        
        return response
    
    def get_num_tokens(self, text):
        res = tokenizer(text)
        return len(res["input_ids"])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "local_model_path": model_path,
            "max_token": self.max_token,
            "URL": self.URL,
            "headers": self.headers,
            "payload": self.payload,
        }

