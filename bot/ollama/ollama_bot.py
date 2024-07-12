# encoding:utf-8

import time
import openai
from bot.bot import Bot
from bot.ollama.ollama_ai_session import OllamaAISession
from bot.ollama.ollama_ai_image import OllamaImage
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf, load_config
from openai import OpenAI 
from zhipuai import ZhipuAI
import json
from common import const
from bot.session_manager import Session
from bridge.bridge import Bridge

# Ollama对话模型API
class OLLAMAAIBot(Bot):
    def __init__(self):
        super().__init__()
        self.sessions = SessionManager(OllamaAISession, model=conf().get("model") or "OLLAMA_AI")
        self.args = {
            "model": conf().get("model") or "ollama3",  # 对话模型的名称
            "temperature": conf().get("temperature", 0.9),  # 值在(0,1)之间(智谱AI 的温度不能取 0 或者 1)
            "top_p": conf().get("top_p", 0.7),  # 值在(0,1)之间(智谱AI 的 top_p 不能取 0 或者 1)
        }
        self.client = OpenAI(base_url=conf().get("ollama_ai_api_base"), api_key=conf().get("ollama_ai_api_key"))
        
    def update_model_in_config(self, model_name):  # 确保包含'self'参数
        config_path = '/home/team1/testcb/chatgpt-on-wechat/config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config['model'] = model_name
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        load_config()  
        logger.warning(f"切换成功, ignore")
        print(f"配置已更新，新的 model 为：{model_name}")


    def reply(self, query, context=None):
        # acquire reply content
        if context.type == ContextType.TEXT:
            self.client = OpenAI(base_url=conf().get("ollama_ai_api_base"), api_key=conf().get("ollama_ai_api_key"))
            logger.info("[OLLAMA_AI] query={}".format(query))

            session_id = context["session_id"]
            reply = None
            clear_memory_commands = conf().get("clear_memory_commands", ["清除记忆"])
            if query in clear_memory_commands:
                self.sessions.clear_session(session_id)
                reply = Reply(ReplyType.INFO, "记忆已清除")
            elif query == "清除所有":
                self.sessions.clear_all_session()
                reply = Reply(ReplyType.INFO, "所有人记忆已清除")
            elif query == "更新配置":
                load_config()
                reply = Reply(ReplyType.INFO, "配置已更新")
            elif query == "切换模型llama3":   
                self.update_model_in_config('llama3')
                load_config()
                Bridge().reset_bot()
                reply = Reply(ReplyType.INFO, "模型已更新")  
            elif query == "切换模型qwen2":   
                self.update_model_in_config('qwen2')
                load_config()
                Bridge().reset_bot()
                reply = Reply(ReplyType.INFO, "模型已更新") 
            elif query == "切换模型chinese":   
                self.update_model_in_config('chinese')
                load_config()
                Bridge().reset_bot()
                reply = Reply(ReplyType.INFO, "模型已更新") 
            elif query == "切换模型med":   
                self.update_model_in_config('med')
                load_config()
                Bridge().reset_bot()
                reply = Reply(ReplyType.INFO, "模型已更新") 
            elif query == "切换模型wizardlm2":   
                self.update_model_in_config('wizardlm2')
                load_config()
                Bridge().reset_bot()
               
                reply = Reply(ReplyType.INFO, "模型已更新") 
            elif query == "查看模型列表":   
                reply = Reply(ReplyType.INFO,"可用模型列表：llama3，qwen2，chinese，med，wizardlm2") 
                
                
            if reply:
                return reply
            session = self.sessions.session_query(query, session_id)
            logger.debug("[OLLAMA_AI] session query={}".format(session.messages))

            api_key = context.get("openai_api_key") or openai.api_key
            model = context.get("gpt_model")
            new_args = None
            if model:
                new_args = self.args.copy()
                new_args["model"] = model
            # if context.get('stream'):
            #     # reply in stream
            #     return self.reply_text_stream(query, new_query, session_id)

            reply_content = self.reply_text(session, api_key, args=new_args)
            logger.debug(
                "[OLLAMA_AI] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                    session.messages,
                    session_id,
                    reply_content["content"],
                    reply_content["completion_tokens"],
                )
            )
            if reply_content["completion_tokens"] == 0 and len(reply_content["content"]) > 0:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
            elif reply_content["completion_tokens"] > 0:
                self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
                reply = Reply(ReplyType.TEXT, reply_content["content"])
            else:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
                logger.debug("[OLLAMA_AI] reply {} used 0 tokens.".format(reply_content))
            return reply
        elif context.type == ContextType.IMAGE_CREATE:
            ok, retstring = self.create_img(query, 0)
            reply = None
            if ok:
                reply = Reply(ReplyType.IMAGE_URL, retstring)
            else:
                reply = Reply(ReplyType.ERROR, retstring)
            return reply

        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply

    def reply_text(self, session: OllamaAISession, api_key=None, args=None, retry_count=0) -> dict:
        """
        call openai's ChatCompletion to get the answer
        :param session: a conversation session
        :param session_id: session id
        :param retry_count: retry count
        :return: {}
        """
        try:
            # if conf().get("rate_limit_chatgpt") and not self.tb4chatgpt.get_token():
            #     raise openai.error.RateLimitError("RateLimitError: rate limit exceeded")
            # if api_key == None, the default openai.api_key will be used
            if args is None:
                args = self.args
            # response = openai.ChatCompletion.create(api_key=api_key, messages=session.messages, **args)
            response = self.client.chat.completions.create(messages=session.messages, **args)
            # logger.debug("[OLLAMA_AI] response={}".format(response))
            # logger.info("[OLLAMA_AI] reply={}, total_tokens={}".format(response.choices[0]['message']['content'], response["usage"]["total_tokens"]))

            return {
                "total_tokens": response.usage.total_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "content": response.choices[0].message.content,
            }
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            if isinstance(e, openai.error.RateLimitError):
                logger.warn("[OLLAMA_AI] RateLimitError: {}".format(e))
                result["content"] = "提问太快啦，请休息一下再问我吧"
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.error.Timeout):
                logger.warn("[OLLAMA_AI] Timeout: {}".format(e))
                result["content"] = "我没有收到你的消息"
                if need_retry:
                    time.sleep(5)
            elif isinstance(e, openai.error.APIError):
                logger.warn("[OLLAMA_AI] Bad Gateway: {}".format(e))
                result["content"] = "请再问我一次"
                if need_retry:
                    time.sleep(10)
            elif isinstance(e, openai.error.APIConnectionError):
                logger.warn("[OLLAMA_AI] APIConnectionError: {}".format(e))
                result["content"] = "我连接不到你的网络"
                if need_retry:
                    time.sleep(5)
            else:
                logger.exception("[OLLAMA_AI] Exception: {}".format(e), e)
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[OLLAMA_AI] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, api_key, args, retry_count + 1)
            else:
                return result