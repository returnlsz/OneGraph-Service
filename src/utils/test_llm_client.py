#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Client 多线程并发测试脚本
测试在高并发情况下的性能和稳定性
"""

import asyncio
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import json
from dataclasses import dataclass
from datetime import datetime
import sys

# 导入你的LLM客户端
# 添加项目根目录到Python路径
sys.path.append('/home/lsz/OneGraph-Service')
from src.llm.llm_client import llm_client


@dataclass
class TestResult:
    """测试结果数据类"""
    thread_id: int
    question_id: int
    question: str
    success: bool
    response_time: float
    response_length: int = 0
    error_message: str = ""
    timestamp: str = ""


class ConcurrentLLMTester:
    def __init__(self):
        """初始化并发测试器"""
        # 测试用的 API Keys（请替换为你的真实 API Keys）
        self.api_keys = [
            "sk-SgiEuM72oCrNUpDZ9b87F351103e4d218d69B42e36C859Df"
        ]
        
        # 生成大量测试问题
        self.test_questions = self._generate_test_questions()
        
        # 测试结果存储
        self.results: List[TestResult] = []
        self.results_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_time': 0,
            'min_response_time': float('inf'),
            'max_response_time': 0,
            'errors': {}
        }

    def _generate_test_questions(self) -> List[str]:
        """生成大量测试问题"""
        questions = []
        
        # 基础问题模板
        basic_questions = [
            "你好，请介绍一下自己。",
            "什么是人工智能？",
            "解释一下机器学习的概念。",
            "Python和Java有什么区别？",
            "什么是深度学习？",
            "请解释一下神经网络。",
            "什么是自然语言处理？",
            "区块链技术是什么？",
            "什么是云计算？",
            "解释一下大数据的概念。"
        ]
        
        # 编程相关问题
        programming_questions = [
            "请写一个Python快速排序算法。",
            "如何在JavaScript中实现异步编程？",
            "解释一下数据库索引的作用。",
            "什么是RESTful API？",
            "如何优化SQL查询性能？",
            "解释一下面向对象编程的特点。",
            "什么是设计模式？举几个例子。",
            "如何处理并发编程中的竞态条件？",
            "解释一下HTTP和HTTPS的区别。",
            "什么是微服务架构？",
            "如何实现负载均衡？",
            "解释一下缓存的作用和类型。",
            "什么是容器化技术？",
            "如何进行代码版本控制？",
            "解释一下敏捷开发方法论。"
        ]
        
        # 数学和科学问题
        math_science_questions = [
            "解释一下概率论的基本概念。",
            "什么是线性代数？",
            "如何计算矩阵的逆？",
            "解释一下微积分的应用。",
            "什么是统计学中的假设检验？",
            "解释一下量子力学的基本原理。",
            "什么是相对论？",
            "如何理解熵的概念？",
            "解释一下DNA的结构。",
            "什么是进化论？"
        ]
        
        # 生活常识问题
        general_questions = [
            "如何保持健康的生活方式？",
            "请推荐一些好书。",
            "如何学习一门新语言？",
            "旅行时需要注意什么？",
            "如何管理时间？",
            "怎样培养良好的习惯？",
            "如何处理压力？",
            "请介绍一下健康饮食。",
            "如何提高工作效率？",
            "怎样建立良好的人际关系？"
        ]
        
        # 创意和分析问题
        creative_questions = [
            "如果你是一个城市规划师，你会如何设计一个理想的城市？",
            "分析一下未来10年科技发展的趋势。",
            "如何解决环境污染问题？",
            "设计一个解决交通拥堵的方案。",
            "如何促进教育公平？",
            "分析社交媒体对社会的影响。",
            "如何应对人口老龄化问题？",
            "设计一个可持续发展的商业模式。",
            "如何提高公众的科学素养？",
            "分析人工智能对就业市场的影响。"
        ]
        
        # 技术深度问题
        technical_questions = [
            "详细解释TCP/IP协议栈的工作原理。",
            "如何设计一个高可用的分布式系统？",
            "解释一下MapReduce算法的原理和应用。",
            "如何实现一个高性能的缓存系统？",
            "详细分析B+树索引的优势。",
            "如何设计一个秒杀系统？",
            "解释一下CAP定理及其应用。",
            "如何实现数据库的主从复制？",
            "详细说明HTTPS的握手过程。",
            "如何设计一个推荐系统？"
        ]
        
        # 合并所有问题
        all_question_categories = [
            basic_questions,
            programming_questions, 
            math_science_questions,
            general_questions,
            creative_questions,
            technical_questions
        ]
        
        # 生成足够多的问题（至少200个）
        for category in all_question_categories:
            questions.extend(category)
        
        # 如果问题不够，重复添加并稍作变化
        while len(questions) < 200:
            base_questions = questions[:50]  # 取前50个问题
            for i, q in enumerate(base_questions):
                # 添加一些变化
                variations = [
                    f"请详细{q}",
                    f"简单{q}",
                    f"从不同角度{q}",
                    f"用例子说明{q}",
                    f"比较分析{q}"
                ]
                questions.append(random.choice(variations))
                if len(questions) >= 200:
                    break
        
        return questions

    def create_client(self) -> llm_client:
        """创建LLM客户端实例"""
        return llm_client(
            base_url="https://api.key77qiqi.cn/v1",
            api_keys=self.api_keys,
            model='gpt-4o-mini-2024-07-18'
        )

    async def single_request(self, thread_id: int, question_id: int, question: str) -> TestResult:
        """执行单个请求"""
        client = self.create_client()
        
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            response = await client.response(question)
            end_time = time.time()
            
            result = TestResult(
                thread_id=thread_id,
                question_id=question_id,
                question=question,
                success=True,
                response_time=end_time - start_time,
                response_length=len(response),
                timestamp=timestamp
            )
            
            print(f"✅ 线程{thread_id}-问题{question_id}: 成功 ({result.response_time:.2f}s)")
            
        except Exception as e:
            end_time = time.time()
            result = TestResult(
                thread_id=thread_id,
                question_id=question_id,
                question=question,
                success=False,
                response_time=end_time - start_time,
                error_message=str(e),
                timestamp=timestamp
            )
            
            print(f"❌ 线程{thread_id}-问题{question_id}: 失败 - {str(e)[:100]}")
        
        return result

    def worker_thread(self, thread_id: int, questions_per_thread: int) -> List[TestResult]:
        """工作线程函数"""
        print(f"🚀 启动线程 {thread_id}, 处理 {questions_per_thread} 个问题")
        
        results = []
        
        # 为每个线程创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for i in range(questions_per_thread):
                # 随机选择问题
                question_id = i
                question = random.choice(self.test_questions)
                
                # 执行异步请求
                result = loop.run_until_complete(
                    self.single_request(thread_id, question_id, question)
                )
                results.append(result)
                
                # 随机延迟，避免过于密集的请求
                time.sleep(random.uniform(0.1, 0.5))
                
        except Exception as e:
            print(f"❌ 线程 {thread_id} 发生错误: {e}")
        finally:
            loop.close()
        
        print(f"✅ 线程 {thread_id} 完成，处理了 {len(results)} 个请求")
        return results

    def update_stats(self, results: List[TestResult]):
        """更新统计信息"""
        with self.results_lock:
            self.results.extend(results)
            
            for result in results:
                self.stats['total_requests'] += 1
                
                if result.success:
                    self.stats['successful_requests'] += 1
                    self.stats['total_time'] += result.response_time
                    self.stats['min_response_time'] = min(
                        self.stats['min_response_time'], 
                        result.response_time
                    )
                    self.stats['max_response_time'] = max(
                        self.stats['max_response_time'], 
                        result.response_time
                    )
                else:
                    self.stats['failed_requests'] += 1
                    error_type = type(Exception(result.error_message)).__name__
                    self.stats['errors'][error_type] = self.stats['errors'].get(error_type, 0) + 1

    def run_concurrent_test(self, num_threads: int = 100, requests_per_thread: int = 3):
        """运行并发测试"""
        print("=" * 80)
        print(f"🚀 开始多线程并发测试")
        print(f"📊 线程数: {num_threads}")
        print(f"📊 每线程请求数: {requests_per_thread}")
        print(f"📊 总请求数: {num_threads * requests_per_thread}")
        print(f"📊 可用问题数: {len(self.test_questions)}")
        print("=" * 80)
        
        # 确认开始测试
        user_input = input("⚠️ 这将产生大量API调用，确认继续？(y/N): ")
        if user_input.lower() != 'y':
            print("❌ 测试已取消")
            return
        
        start_time = time.time()
        
        # 使用线程池执行并发测试
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            future_to_thread = {
                executor.submit(self.worker_thread, thread_id, requests_per_thread): thread_id
                for thread_id in range(num_threads)
            }
            
            # 收集结果
            completed_threads = 0
            for future in as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                try:
                    results = future.result()
                    self.update_stats(results)
                    completed_threads += 1
                    
                    print(f"📈 进度: {completed_threads}/{num_threads} 线程完成")
                    
                except Exception as e:
                    print(f"❌ 线程 {thread_id} 异常: {e}")
        
        end_time = time.time()
        total_test_time = end_time - start_time
        
        # 打印详细统计
        self.print_detailed_stats(total_test_time)
        
        # 保存结果到文件
        self.save_results_to_file()

    def print_detailed_stats(self, total_test_time: float):
        """打印详细统计信息"""
        print("\n" + "=" * 80)
        print("📊 并发测试结果统计")
        print("=" * 80)
        
        success_rate = (self.stats['successful_requests'] / self.stats['total_requests'] * 100) if self.stats['total_requests'] > 0 else 0
        avg_response_time = (self.stats['total_time'] / self.stats['successful_requests']) if self.stats['successful_requests'] > 0 else 0
        
        print(f"⏱️  总测试时间: {total_test_time:.2f} 秒")
        print(f"📊 总请求数: {self.stats['total_requests']}")
        print(f"✅ 成功请求: {self.stats['successful_requests']}")
        print(f"❌ 失败请求: {self.stats['failed_requests']}")
        print(f"📈 成功率: {success_rate:.2f}%")
        print(f"⚡ 平均响应时间: {avg_response_time:.2f} 秒")
        
        if self.stats['successful_requests'] > 0:
            print(f"🚀 最快响应: {self.stats['min_response_time']:.2f} 秒")
            print(f"🐌 最慢响应: {self.stats['max_response_time']:.2f} 秒")
            print(f"📊 QPS (每秒请求数): {self.stats['successful_requests'] / total_test_time:.2f}")
        
        # 错误统计
        if self.stats['errors']:
            print(f"\n❌ 错误类型统计:")
            for error_type, count in self.stats['errors'].items():
                print(f"   • {error_type}: {count} 次")
        
        # 响应时间分布
        if self.results:
            successful_results = [r for r in self.results if r.success]
            if successful_results:
                response_times = [r.response_time for r in successful_results]
                response_times.sort()
                
                print(f"\n⏱️ 响应时间分布:")
                print(f"   • P50: {response_times[len(response_times)//2]:.2f}s")
                print(f"   • P90: {response_times[int(len(response_times)*0.9)]:.2f}s")
                print(f"   • P95: {response_times[int(len(response_times)*0.95)]:.2f}s")
                print(f"   • P99: {response_times[int(len(response_times)*0.99)]:.2f}s")

    def save_results_to_file(self):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_concurrent_test_results_{timestamp}.json"
        
        # 准备保存的数据
        save_data = {
            'test_info': {
                'timestamp': timestamp,
                'total_requests': self.stats['total_requests'],
                'successful_requests': self.stats['successful_requests'],
                'failed_requests': self.stats['failed_requests']
            },
            'statistics': self.stats,
            'detailed_results': [
                {
                    'thread_id': r.thread_id,
                    'question_id': r.question_id,
                    'question': r.question[:100] + '...' if len(r.question) > 100 else r.question,
                    'success': r.success,
                    'response_time': r.response_time,
                    'response_length': r.response_length,
                    'error_message': r.error_message,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ]
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 测试结果已保存到: {filename}")
        except Exception as e:
            print(f"\n❌ 保存结果失败: {e}")

    def run_stress_test(self):
        """运行压力测试 - 多种并发级别"""
        print("🔥 开始压力测试 - 多种并发级别")
        
        test_configs = [
            (10, 2),   # 10线程，每线程2请求
            (25, 2),   # 25线程，每线程2请求  
            (50, 2),   # 50线程，每线程2请求
            (100, 1),  # 100线程，每线程1请求
        ]
        
        for threads, requests in test_configs:
            print(f"\n🧪 测试配置: {threads} 线程 × {requests} 请求")
            
            # 重置统计
            self.results = []
            self.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_time': 0,
                'min_response_time': float('inf'),
                'max_response_time': 0,
                'errors': {}
            }
            
            self.run_concurrent_test(threads, requests)
            
            # 等待一段时间再进行下一轮测试
            print("⏳ 等待 30 秒后进行下一轮测试...")
            time.sleep(30)


def main():
    """主函数"""
    print("🚀 LLM Client 多线程并发测试工具")
    print("=" * 80)
    
    tester = ConcurrentLLMTester()
    
    print("⚠️ 重要提醒:")
    print("1. 请确保在代码中设置了正确的 API Keys")
    print("2. 高并发测试会产生大量 API 调用，请注意费用")
    print("3. 请确保 API 服务有足够的配额和频率限制")
    print()
    
    while True:
        print("请选择测试模式:")
        print("1. 标准并发测试 (100线程)")
        print("2. 自定义并发测试")
        print("3. 压力测试 (多种并发级别)")
        print("4. 退出")
        
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '1':
            tester.run_concurrent_test(num_threads=100, requests_per_thread=2)
        elif choice == '2':
            try:
                threads = int(input("请输入线程数: "))
                requests = int(input("请输入每线程请求数: "))
                tester.run_concurrent_test(num_threads=threads, requests_per_thread=requests)
            except ValueError:
                print("❌ 请输入有效的数字")
        elif choice == '3':
            tester.run_stress_test()
        elif choice == '4':
            print("👋 测试结束")
            break
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()