import sys
import os
import asyncio
import httpx
import tiktoken  # 用于计算 token 数量（确保安装：`pip install tiktoken`）

# 添加路径
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

class llm_client:
    def __init__(self, base_url="https://api.key77qiqi.cn/v1", api_keys=None, model='gpt-4o-mini-2024-07-18'):
        """
        初始化 LLM 客户端
        :param base_url: API 的基础地址
        :param api_keys: 包含多个 API Key 的列表
        :param model: 使用的模型名称,gpt-4o,gpt-4o-mini,gpt-4o-mini-2024-07-18,gpt-3.5-turbo-0125,gpt-4o-2024-11-20,o3-2025-04-16
        """
        self.base_url = base_url
        # 如果未提供 api_keys,则使用一个默认的 API Key
        self.api_keys = api_keys or []
        self.model = model  # 模型名称
        # self.model = "gpt-3.5-turbo-16k"
        self.api_key_index = 0  # 用于跟踪当前正在使用的 API Key

    def switch_api_key(self):
        """
        切换到下一个 API Key。如果达到列表末尾,则从头开始循环。
        """
        self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)
        print(f"切换到新的 API Key: {self.api_keys[self.api_key_index]}")

    async def response(self, question):
        """
        向 API 发送请求并返回响应。
        :param question: 用户提出的问题
        :param mode: 模式（例如对 prompt 的不同处理方式）
        :return: 模型生成的响应内容
        """
        url = f"{self.base_url}/chat/completions"
        retries = 100  # 设置最大重试次数

        # 第一步：计算 `question` 的 token 数量，并判断是否超出阈值
        encoder = tiktoken.encoding_for_model(self.model)
        # tokens = encoder.encode(question)
        
        # max_token_threshold = 50000
        # if len(tokens) > max_token_threshold:
        #     print(f"⚠️ 输入的 token 数量超出阈值 ({max_token_threshold})，将截断内容...")
        #     # 截断 token 到阈值长度，并解码为字符串
        #     truncated_tokens = tokens[:max_token_threshold]
        #     question = encoder.decode(truncated_tokens)
        #     # print(f"截断后的 `question` 是: {question}")
            
        for attempt in range(retries):
            current_api_key = self.api_keys[self.api_key_index]
            headers = {
                "Authorization": f"Bearer {current_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": 1
            }

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
                    
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()  # 如果状态码不是 2xx，则抛出异常
                    result = response.json()
                    print("生成内容token长度:",len(encoder.encode(result["choices"][0]["message"]["content"])))
                    return result["choices"][0]["message"]["content"]  # 返回结果

            except httpx.ConnectTimeout:
                print(f"Attempt {attempt + 1}: 连接超时,正在重试...")
                self.switch_api_key()  # 切换到下一个 API Key
            except httpx.ReadTimeout:
                print(f"Attempt {attempt + 1}: 读取超时,正在重试...")
                self.switch_api_key()  # 切换到下一个 API Key
            except httpx.ConnectError as exc:
                print(f"Attempt {attempt + 1}: 连接错误 {exc},正在重试...")
                self.switch_api_key()  # 切换到下一个 API Key
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 403:
                    print(f"Attempt {attempt + 1}: 权限不足 (403 Forbidden)，可能 API Key 无效或无访问权限，切换 API Key 并重试...")
                    self.switch_api_key()  # 切换到下一个 API Key
                elif exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    wait_time = int(retry_after) if retry_after is not None else 5  # 默认等待时间（5秒）
                    print(f"Attempt {attempt + 1}: 请求频率过高 (429 Too Many Requests)，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)  # 等待指定时间后重试
                elif exc.response.status_code == 400:
                    error_details = exc.response.json()  # 获取详细的错误信息
                    print(f"Attempt {attempt + 1}: 客户端请求错误 (400 Bad Request): {error_details}")
                    self.switch_api_key()  # 切换到下一个 API Key
                elif exc.response.status_code == 401:
                    error_details = exc.response.json()  # 获取详细的错误信息
                    print(f"Attempt {attempt + 1}: 认证失败 (401 Unauthorized): {error_details}")
                    print("API Key 可能无效或已被撤销，尝试切换 API Key 并重试...")
                    self.switch_api_key()  # 切换到下一个 API Key
                elif exc.response.status_code == 413:  # 对 413 错误进行处理
                    print(f"Attempt {attempt + 1}: 请求内容过大 (413 Request Entity Too Large)")
                    self.switch_api_key()  # 切换到下一个 API Key
                    # self.max_tokens = max(self.max_tokens - 2000, 1000)  # 减少 max_tokens，但保留最低值为 1000
                    # print(f"已调整 max_tokens: {self.max_tokens}")
                elif 500 <= exc.response.status_code < 600:
                    print(f"Attempt {attempt + 1}: 服务器错误 {exc.response.status_code},正在重试...")
                    self.switch_api_key()  # 切换到下一个 API Key
                else:
                    print(f"HTTP 错误: {exc.response.status_code}。")
                    raise
            except Exception as exc:
                print(f"未知错误: {exc},正在重试...")
                self.switch_api_key()  # 切换到下一个 API Key

            # 如果达到最大重试次数,则抛出异常
            if attempt == retries - 1:
                print(f"最大重试次数已到 ({retries})")
                raise Exception("请求失败：达到最大重试次数")
