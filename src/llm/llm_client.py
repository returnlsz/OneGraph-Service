import sys
import os
import asyncio
import httpx

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
        self.api_key_index = 0  # 用于跟踪当前正在使用的 API Key

    def switch_api_key(self):
        """
        切换到下一个 API Key。如果达到列表末尾,则从头开始循环。
        """
        if len(self.api_keys) == 0:
            print("⚠️ 没有可用的 API Key")
            return
            
        self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)
        print(f"切换到新的 API Key: {self.api_keys[self.api_key_index][:10]}...")  # 只显示前10个字符

    async def response(self, question):
        """
        向 API 发送请求并返回响应。
        :param question: 用户提出的问题
        :return: 模型生成的响应内容
        """
        url = f"{self.base_url}/chat/completions"
        retries = 5  # 减少重试次数，避免无限循环
            
        for attempt in range(retries):
            if len(self.api_keys) == 0:
                raise Exception("没有可用的 API Key")
                
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
                "max_tokens": 12800,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": 1
            }

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:  # 减少超时时间
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()  # 如果状态码不是 2xx，则抛出异常
                    result = response.json()
                    
                    return result["choices"][0]["message"]["content"]  # 返回结果

            except httpx.ConnectTimeout:
                print(f"Attempt {attempt + 1}: 连接超时,正在重试...")
                self.switch_api_key()
            except httpx.ReadTimeout:
                print(f"Attempt {attempt + 1}: 读取超时,正在重试...")
                self.switch_api_key()
            except httpx.ConnectError as exc:
                print(f"Attempt {attempt + 1}: 连接错误 {exc},正在重试...")
                self.switch_api_key()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 403:
                    print(f"Attempt {attempt + 1}: 权限不足 (403 Forbidden)，切换 API Key 并重试...")
                    self.switch_api_key()
                elif exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    wait_time = int(retry_after) if retry_after is not None else 5
                    print(f"Attempt {attempt + 1}: 请求频率过高 (429 Too Many Requests)，等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                elif exc.response.status_code == 400:
                    try:
                        error_details = exc.response.json()
                        print(f"Attempt {attempt + 1}: 客户端请求错误 (400 Bad Request): {error_details}")
                    except:
                        print(f"Attempt {attempt + 1}: 客户端请求错误 (400 Bad Request)")
                    self.switch_api_key()
                elif exc.response.status_code == 401:
                    try:
                        error_details = exc.response.json()
                        print(f"Attempt {attempt + 1}: 认证失败 (401 Unauthorized): {error_details}")
                    except:
                        print(f"Attempt {attempt + 1}: 认证失败 (401 Unauthorized)")
                    print("API Key 可能无效或已被撤销，尝试切换 API Key 并重试...")
                    self.switch_api_key()
                elif exc.response.status_code == 413:
                    print(f"Attempt {attempt + 1}: 请求内容过大 (413 Request Entity Too Large)")
                    self.switch_api_key()
                elif 500 <= exc.response.status_code < 600:
                    print(f"Attempt {attempt + 1}: 服务器错误 {exc.response.status_code},正在重试...")
                    self.switch_api_key()
                else:
                    print(f"HTTP 错误: {exc.response.status_code}")
                    raise
            except Exception as exc:
                error_msg = str(exc)
                print(f"Attempt {attempt + 1}: 未知错误: {error_msg[:200]}...")
                self.switch_api_key()
                
                # 添加延迟避免快速重试
                await asyncio.sleep(1)

        # 如果达到最大重试次数,则抛出异常
        print(f"最大重试次数已到 ({retries})")
        raise Exception("请求失败：达到最大重试次数")