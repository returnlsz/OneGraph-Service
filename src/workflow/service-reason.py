import pandas as pd
import numpy as np
from lxml import etree
import io
import json
import asyncio
import aiofiles
from pathlib import Path
from collections import defaultdict
from tqdm.asyncio import tqdm
import logging
from typing import List, Dict, Any
import pyarrow.parquet as pq
import re
import sys
import traceback
import time
import pickle
import hashlib
import argparse

# 导入你的LLM客户端
sys.path.append('/home/lsz/OneGraph-Service')
from src.llm.llm_client import llm_client

# 增加递归限制
sys.setrecursionlimit(10000)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QA_PROMPT = """
Please answer the question based on the given knowledge and using the knowledge you already possess. Please provide the final answer in the form of options.
Question: 
{question}
Knowledge: 
{knowledge}
Answer:
"""

QA_PROMPT_WO_KNOWLEDGE = """
Please answer the question using the knowledge you already possess. Please provide the final answer in the form of options.
Question: 
{question}
Answer:
"""

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str):
        # 如果是相对路径，转换为基于用户主目录的绝对路径
        if not Path(checkpoint_dir).is_absolute():
            checkpoint_dir = str(Path.home() / checkpoint_dir)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def get_checkpoint_path(self, dataset_name: str, file_name: str) -> Path:
        """获取检查点文件路径"""
        return self.checkpoint_dir / f"{dataset_name}_{file_name}_checkpoint.pkl"
    
    def get_progress_path(self, dataset_name: str, file_name: str) -> Path:
        """获取进度文件路径"""
        return self.checkpoint_dir / f"{dataset_name}_{file_name}_progress.json"
    
    def save_checkpoint(self, dataset_name: str, file_name: str, 
                       processed_samples: List[Dict], batch_index: int, 
                       total_samples: int, stats: Dict):
        """保存检查点"""
        checkpoint_data = {
            'processed_samples': processed_samples,
            'batch_index': batch_index,
            'total_samples': total_samples,
            'stats': stats,
            'timestamp': time.time()
        }
        
        checkpoint_path = self.get_checkpoint_path(dataset_name, file_name)
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # 同时保存进度信息（JSON格式，便于查看）
            progress_info = {
                'dataset_name': dataset_name,
                'file_name': file_name,
                'processed_count': len(processed_samples),
                'batch_index': batch_index,
                'total_samples': total_samples,
                'progress_percentage': (len(processed_samples) / total_samples * 100) if total_samples > 0 else 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'stats': stats
            }
            
            progress_path = self.get_progress_path(dataset_name, file_name)
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(progress_info, f, indent=2, ensure_ascii=False)
                
            logger.info(f"检查点已保存: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def load_checkpoint(self, dataset_name: str, file_name: str) -> Dict:
        """加载检查点"""
        checkpoint_path = self.get_checkpoint_path(dataset_name, file_name)
        
        if not checkpoint_path.exists():
            return None
            
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"检查点已加载: {checkpoint_path}")
            logger.info(f"已处理样本数: {len(checkpoint_data['processed_samples'])}/{checkpoint_data['total_samples']}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None
    
    def has_checkpoint(self, dataset_name: str, file_name: str) -> bool:
        """检查是否存在检查点"""
        return self.get_checkpoint_path(dataset_name, file_name).exists()
    
    def remove_checkpoint(self, dataset_name: str, file_name: str):
        """删除检查点（完成处理后）"""
        checkpoint_path = self.get_checkpoint_path(dataset_name, file_name)
        progress_path = self.get_progress_path(dataset_name, file_name)
        
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if progress_path.exists():
                progress_path.unlink()
            logger.info(f"检查点已清理: {dataset_name}_{file_name}")
        except Exception as e:
            logger.error(f"清理检查点失败: {e}")
    
    def list_checkpoints(self) -> List[Dict]:
        """列出所有检查点"""
        checkpoints = []
        for progress_file in self.checkpoint_dir.glob("*_progress.json"):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_info = json.load(f)
                checkpoints.append(progress_info)
            except Exception as e:
                logger.error(f"读取进度文件失败 {progress_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x.get('timestamp', ''))

class TripleConverter:
    def __init__(self, api_base_url, api_keys, api_model, process_batch_size=20, 
                 max_concurrent=50, max_triples_per_sample=100, checkpoint_dir=None):
        self.llm = llm_client(
            base_url=api_base_url,
            api_keys=api_keys,
            model=api_model
        )
        self.process_batch_size = process_batch_size
        self.max_concurrent = max_concurrent
        self.max_triples_per_sample = max_triples_per_sample
        # 创建信号量来限制并发数
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 添加统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_count': 0,
            'timeout_count': 0
        }
        
        # 初始化检查点管理器
        if checkpoint_dir is None:
            checkpoint_dir = "./checkpoints"
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    def safe_get_value(self, data, key, default=""):
        """安全地获取值，处理各种数据类型"""
        try:
            if data is None:
                return default
            
            if isinstance(data, dict):
                value = data.get(key, default)
            elif hasattr(data, key):
                value = getattr(data, key)
            else:
                return default
            
            # 处理numpy数组和pandas Series
            if hasattr(value, 'iloc'):
                # pandas Series
                return str(value.iloc[0]) if len(value) > 0 else default
            elif hasattr(value, '__len__') and not isinstance(value, str):
                # 数组类型
                if len(value) > 0:
                    return str(value[0])
                else:
                    return default
            elif pd.isna(value):
                return default
            else:
                return str(value) if value is not None else default
                
        except Exception as e:
            logger.debug(f"Error getting value for key {key}: {e}")
            return default
    
    def safe_check_exists(self, data, key):
        """安全地检查键是否存在且有值"""
        try:
            if data is None:
                return False
            
            if isinstance(data, dict):
                value = data.get(key)
            elif hasattr(data, key):
                value = getattr(data, key)
            else:
                return False
            
            if value is None:
                return False
            
            # 处理pandas/numpy类型
            if hasattr(value, '__len__') and not isinstance(value, str):
                return len(value) > 0
            elif pd.isna(value):
                return False
            else:
                return True
                
        except Exception:
            return False

    def format_question_with_options(self, sample: Dict) -> str:
        """格式化问题和选项"""
        question = self.safe_get_value(sample, 'question')
        
        # 获取所有可能的选项字段
        options = []
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # 支持更多选项
        
        for letter in option_letters:
            option_value = self.safe_get_value(sample, letter)
            if option_value and option_value.strip():  # 只添加非空选项
                options.append(f"{letter}. {option_value}")
        
        # 组合问题和选项
        if options:
            formatted_question = question + "\n" + "\n".join(options)
        else:
            formatted_question = question
            
        return formatted_question
        
    def extract_triples_from_sample(self, sample: Dict) -> List[Dict]:
        """从样本中提取所有三元组，并在此处进行数量限制"""
        all_triples = []
        
        try:
            # 提取 relatedTriples
            if self.safe_check_exists(sample, 'relatedTriples'):
                related_triples = sample['relatedTriples']
                
                # 处理不同的数据类型
                if isinstance(related_triples, str):
                    try:
                        related_triples = json.loads(related_triples)
                    except Exception as e:
                        logger.debug(f"JSON parse error for relatedTriples: {e}")
                        related_triples = []
                elif hasattr(related_triples, 'tolist'):
                    related_triples = related_triples.tolist()
                elif not isinstance(related_triples, list):
                    related_triples = [related_triples] if related_triples else []
                
                for triple in related_triples:
                    if isinstance(triple, dict):
                        h = self.safe_get_value(triple, 'head') or self.safe_get_value(triple, 'Head')
                        r = self.safe_get_value(triple, 'relation') or self.safe_get_value(triple, 'Relation')
                        t = self.safe_get_value(triple, 'tail') or self.safe_get_value(triple, 'Tail')
                        
                        if h and r and t:
                            all_triples.append({'h': str(h), 'r': str(r), 't': str(t)})
            
            # 提取 enrich_triples
            if self.safe_check_exists(sample, 'enrich_triples'):
                enrich_triples = sample['enrich_triples']
                
                if isinstance(enrich_triples, str):
                    try:
                        enrich_triples = json.loads(enrich_triples)
                    except Exception as e:
                        logger.debug(f"JSON parse error for enrich_triples: {e}")
                        enrich_triples = []
                elif hasattr(enrich_triples, 'tolist'):
                    enrich_triples = enrich_triples.tolist()
                elif not isinstance(enrich_triples, list):
                    enrich_triples = [enrich_triples] if enrich_triples else []
                
                for triple in enrich_triples:
                    if isinstance(triple, dict):
                        h = self.safe_get_value(triple, 'head') or self.safe_get_value(triple, 'Head')
                        r = self.safe_get_value(triple, 'relation') or self.safe_get_value(triple, 'Relation')
                        t = self.safe_get_value(triple, 'tail') or self.safe_get_value(triple, 'Tail')
                        
                        if h and r and t:
                            all_triples.append({'h': str(h), 'r': str(r), 't': str(t)})
            
            # 提取 hierarchical_enhancements
            if self.safe_check_exists(sample, 'hierarchical_enhancements'):
                hierarchical_enhancements = sample['hierarchical_enhancements']
                
                if isinstance(hierarchical_enhancements, str):
                    try:
                        hierarchical_enhancements = json.loads(hierarchical_enhancements)
                    except Exception as e:
                        logger.debug(f"JSON parse error for hierarchical_enhancements: {e}")
                        hierarchical_enhancements = []
                elif hasattr(hierarchical_enhancements, 'tolist'):
                    hierarchical_enhancements = hierarchical_enhancements.tolist()
                elif not isinstance(hierarchical_enhancements, list):
                    hierarchical_enhancements = [hierarchical_enhancements] if hierarchical_enhancements else []
                
                for triple in hierarchical_enhancements:
                    if isinstance(triple, dict):
                        h = self.safe_get_value(triple, 'head') or self.safe_get_value(triple, 'Head')
                        r = self.safe_get_value(triple, 'relation') or self.safe_get_value(triple, 'Relation')
                        t = self.safe_get_value(triple, 'tail') or self.safe_get_value(triple, 'Tail')
                        
                        if h and r and t:
                            all_triples.append({'h': str(h), 'r': str(r), 't': str(t)})
            
        except Exception as e:
            logger.error(f"Error extracting triples: {e}")
            return []
        

        # 4. 从relatedTriples和hierarchical_enhancements分别选取指定数量
        def extract_limited_triples(field_name):
            if not self.safe_check_exists(sample, field_name):
                return []
            data = sample[field_name]
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    return []
            elif hasattr(data, 'tolist'):
                data = data.tolist()
            elif not isinstance(data, list):
                data = [data] if data else []
            
            triples = []
            for triple in data:
                if isinstance(triple, dict):
                    h = self.safe_get_value(triple, 'head') or self.safe_get_value(triple, 'Head')
                    r = self.safe_get_value(triple, 'relation') or self.safe_get_value(triple, 'Relation')
                    t = self.safe_get_value(triple, 'tail') or self.safe_get_value(triple, 'Tail')
                    if h and r and t:
                        triples.append({'h': str(h), 'r': str(r), 't': str(t)})
            return triples[:self.max_triples_per_sample]
        
        related_limited = extract_limited_triples('relatedTriples')
        hierarchical_limited = extract_limited_triples('hierarchical_enhancements')
        enrich_limited = extract_limited_triples('enrich_triples')
        
        all_triples = related_limited + hierarchical_limited + enrich_limited
        all_triples = all_triples[:self.max_triples_per_sample * 2]
        print("Triple nums:",len(all_triples))

        # # 在这里进行数量限制 - 总入口限制
        # if len(all_triples) > self.max_triples_per_sample:
        #     logger.info(f"Limiting triples from {len(all_triples)} to {self.max_triples_per_sample}")
        #     # 可以选择不同的策略：
        #     # 1. 简单截取前N个
        #     # all_triples = all_triples[:self.max_triples_per_sample]
            
        #     # 2. 或者随机采样（可选）
        #     # import random
        #     # all_triples = random.sample(all_triples, self.max_triples_per_sample)
            
        #     # 3. 或者按重要性排序后截取（可选，需要定义重要性指标）
            
        
        return all_triples

    def triples_to_edgetable(self, triples: List[Dict]) -> str:
        """将三元组转换为边表格式 - 移除数量限制"""
        if not triples:
            return ""
            
        result = ""
        try:
            for triple in triples:  # 处理所有triples，不做限制
                h = str(triple.get('h', '')).strip()
                r = str(triple.get('r', '')).strip()
                t = str(triple.get('t', '')).strip()
                
                if h and r and t:
                    # 清理字符串中的特殊字符
                    h_clean = h.replace('(', '').replace(')', '').replace(',', '')
                    r_clean = r.replace('(', '').replace(')', '').replace(',', '')
                    t_clean = t.replace('(', '').replace(')', '').replace(',', '')
                    
                    result += f"({h_clean},{r_clean},{t_clean})\n"
                    
        except Exception as e:
            logger.error(f"Error in triples_to_edgetable: {e}")
            return ""
            
        return result.strip()

    def triples_to_nodesequence(self, triples: List[Dict]) -> str:
        """将三元组转换为节点序列格式 - 移除数量限制"""
        if not triples:
            return ""
            
        try:
            # 构建关系字典
            relation_dict = defaultdict(set)  # 使用set避免重复
            all_nodes = set()
            
            for triple in triples:  # 处理所有triples
                h = str(triple.get('h', '')).strip()
                t = str(triple.get('t', '')).strip()
                if h and t:
                    relation_dict[h].add(t)
                    all_nodes.add(h)
                    all_nodes.add(t)
            
            # 使用广度优先搜索来构建路径，避免递归
            def bfs_paths(start_node, max_depth=5):
                """使用BFS构建路径，限制最大深度"""
                paths = []
                queue = [(start_node, [start_node], 0)]  # (当前节点, 当前路径, 深度)
                visited_paths = set()
                
                while queue:  # 移除路径数量限制
                    current, path, depth = queue.pop(0)
                    
                    if depth >= max_depth:  # 限制深度
                        continue
                        
                    path_str = "->".join(path)
                    if path_str in visited_paths:
                        continue
                    visited_paths.add(path_str)
                    
                    # 如果当前节点没有出边，或者达到一定深度，添加路径
                    if current not in relation_dict or depth >= 3:
                        if len(path) > 1:  # 只添加长度大于1的路径
                            paths.append(path_str)
                    
                    # 添加子节点到队列
                    if current in relation_dict:
                        for next_node in relation_dict[current]:  # 处理所有子节点
                            if next_node not in path:  # 避免循环
                                new_path = path + [next_node]
                                queue.append((next_node, new_path, depth + 1))
                
                return paths
            
            # 获取起始节点（入度为0的节点，或者出现频率高的节点）
            in_degree = defaultdict(int)
            for h in relation_dict:
                for t in relation_dict[h]:
                    in_degree[t] += 1
            
            # 找到入度为0的节点作为起始点
            start_nodes = [node for node in all_nodes if in_degree[node] == 0]
            if not start_nodes:
                # 如果没有入度为0的节点，选择出度最高的几个节点
                start_nodes = sorted(relation_dict.keys(), 
                                   key=lambda x: len(relation_dict[x]), 
                                   reverse=True)
            
            # 生成路径
            all_paths = set()
            for start in start_nodes:  # 处理所有起始节点
                paths = bfs_paths(start)
                all_paths.update(paths)
            
            return "\n".join(list(all_paths))  # 返回所有路径
            
        except Exception as e:
            logger.error(f"Error in triples_to_nodesequence: {e}")
            return ""

    def triples_to_graphml(self, triples: List[Dict]) -> str:
        """将三元组转换为GraphML格式 - 移除数量限制"""
        if not triples:
            return ""
            
        try:
            # 创建GraphML根元素
            graphml = etree.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
            graph = etree.SubElement(graphml, "graph", id="G", edgedefault="directed")
            
            # 收集所有节点
            nodes = set()
            valid_triples = []
            
            for triple in triples:  # 处理所有triples
                h = str(triple.get('h', '')).strip()
                t = str(triple.get('t', '')).strip()
                r = str(triple.get('r', '')).strip()
                
                if h and t and r:
                    nodes.add(h)
                    nodes.add(t)
                    valid_triples.append((h, t, r))
            
            # 为节点创建ID映射
            node_id_map = {node: f"n{i}" for i, node in enumerate(nodes)}
            
            # 添加节点
            for node, node_id in node_id_map.items():
                node_element = etree.SubElement(graph, "node", id=node_id)
                etree.SubElement(node_element, "data", key="label").text = node[:100]  # 限制长度
            
            # 添加边
            for h, t, r in valid_triples:  # 处理所有有效triples
                if h in node_id_map and t in node_id_map:
                    edge = etree.SubElement(graph, "edge", 
                                          source=node_id_map[h], 
                                          target=node_id_map[t])
                    etree.SubElement(edge, "data", key="label").text = r[:50]  # 限制长度
            
            return etree.tostring(graphml, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("UTF-8")
            
        except Exception as e:
            logger.error(f"Error in triples_to_graphml: {e}")
            return ""

    def triples_to_syntaxtree(self, triples: List[Dict], k=None) -> str:
        """将三元组转换为语法树格式 - 移除数量限制"""
        if not triples:
            return ""
            
        try:
            # 统计实体出现频率
            entity_count = defaultdict(int)
            for triple in triples:  # 处理所有triples
                h = str(triple.get('h', '')).strip()
                t = str(triple.get('t', '')).strip()
                if h:
                    entity_count[h] += 1
                if t:
                    entity_count[t] += 1
            
            if not entity_count:
                return ""
            
            # 按频率排序
            sorted_entities = sorted(entity_count.items(), key=lambda x: x[1], reverse=True)
            
            # 如果没有指定k，则处理所有实体，否则使用指定的k值
            entities_to_process = sorted_entities if k is None else sorted_entities[:k]
            
            result = ""
            for i, (entity, count) in enumerate(entities_to_process):
                result += f"Tree {i+1} (Center: {entity}, Frequency: {count}):\n"
                
                # 找到以该实体为中心的三元组
                center_triples = []
                for triple in triples:  # 处理所有triples
                    h = str(triple.get('h', '')).strip()
                    t = str(triple.get('t', '')).strip()
                    r = str(triple.get('r', '')).strip()
                    
                    if (h == entity or t == entity) and h and t and r:
                        center_triples.append((h, r, t))
                
                # 构建树结构
                result += f"  Root: {entity}\n"
                
                # 出边（作为头实体）
                outgoing = [(r, t) for h, r, t in center_triples if h == entity]
                for r, t in outgoing:  # 处理所有出边
                    result += f"    -> {t} (via {r})\n"
                
                # 入边（作为尾实体）
                incoming = [(h, r) for h, r, t in center_triples if t == entity]
                for h, r in incoming:  # 处理所有入边
                    result += f"    <- {h} (via {r})\n"
                
                result += "\n"
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error in triples_to_syntaxtree: {e}")
            return ""

    async def triples_to_naturallanguage(self, triples: List[Dict]) -> str:
        """将三元组转换为自然语言格式 - 移除数量限制"""
        if not triples:
            return ""
            
        try:
            # 构建三元组字符串，处理所有triples
            triple_strs = []
            for triple in triples:  # 处理所有triples
                h = str(triple.get('h', '')).strip()
                r = str(triple.get('r', '')).strip()
                t = str(triple.get('t', '')).strip()
                
                if h and r and t:
                    triple_strs.append(f"({h},{r},{t})")
            
            if not triple_strs:
                return ""
                
            triples_text = '\n'.join(triple_strs)
            query = f'请对下列三元组信息做一段简要的总结：\n{triples_text}'
            
            # response = await self._safe_llm_call_with_retry(query, "summary", 1)
            response = None

            return response if response else self._fallback_natural_language(triples)
            
        except Exception as e:
            logger.error(f"Error in natural language conversion: {e}")
            return self._fallback_natural_language(triples)
    
    def _fallback_natural_language(self, triples: List[Dict]) -> str:
        """回退的自然语言转换方法 - 移除数量限制"""
        try:
            result = ""
            for triple in triples:  # 处理所有triples
                h = str(triple.get('h', '')).strip()
                r = str(triple.get('r', '')).strip()
                t = str(triple.get('t', '')).strip()
                
                if h and r and t:
                    result += f"{h} {r} {t}. "
            return result.strip()
        except:
            return ""

    async def _safe_llm_call_with_retry(self, prompt: str, task_id: str, max_retries: int = 100) -> str:
        """带重试机制的安全LLM调用 - 优化版本"""
        async with self.semaphore:  # 限制并发数
            self.stats['total_requests'] += 1
            
            for attempt in range(max_retries):
                try:
                    # 减少超时时间，更快失败
                    response = await asyncio.wait_for(
                        self._safe_llm_call(prompt, task_id), 
                        timeout=120.0
                    )
                    
                    if response:
                        self.stats['successful_requests'] += 1
                        return response
                    else:
                        raise Exception("Empty response")
                        
                except asyncio.TimeoutError:
                    self.stats['timeout_count'] += 1
                    logger.warning(f"Timeout on attempt {attempt + 1} for task {task_id}")
                    if attempt < max_retries - 1:
                        # 减少等待时间
                        wait_time = 0.5 * (attempt + 1)
                        await asyncio.sleep(wait_time)
                    else:
                        self.stats['failed_requests'] += 1
                        logger.error(f"All attempts failed for task {task_id}")
                        return ""
                        
                except Exception as e:
                    self.stats['retry_count'] += 1
                    logger.error(f"Error on attempt {attempt + 1} for task {task_id}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.2)  # 减少等待时间
                    else:
                        self.stats['failed_requests'] += 1
                        return ""
            
            return ""

    async def _safe_llm_call(self, prompt: str, task_id: str) -> str:
        """安全的LLM调用"""
        try:
            # 直接调用你的LLM客户端的response方法
            response = await self.llm.response(prompt)
            return str(response) if response else ""
            
        except Exception as e:
            logger.error(f"LLM call failed for task {task_id}: {e}")
            raise  # 重新抛出异常，让上层处理重试

    def extract_answer_from_response(self, response: str) -> str:
        """从LLM响应中提取答案"""
        if not response:
            return ""
            
        try:
            # 尝试提取选择题答案 (A, B, C, D)
            answer_pattern = r'\b([ABCD])\b'
            matches = re.findall(answer_pattern, response)
            if matches:
                return matches[-1]  # 返回最后一个匹配的答案
            
            # 如果没有找到标准答案格式，返回原始响应的前100个字符
            return response.strip()[:100]
        except:
            return ""

    async def process_direct_qa(self, sample: Dict) -> Dict:
        """处理直接问答（不使用知识）"""
        try:
            sample_id = self.safe_get_value(sample, 'id')
            
            # 格式化问题和选项
            formatted_question = self.format_question_with_options(sample)
            
            # 使用不带知识的prompt
            qa_prompt = QA_PROMPT_WO_KNOWLEDGE.format(question=formatted_question)
            
            # 将qa_prompt以追加方式写入文件，包含时间戳和样本ID
            import datetime
            with open('/home/lsz/OneGraph-Service/temp.txt', 'a', encoding='utf-8') as f:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] Sample ID: {sample_id}_direct\n")
                f.write(qa_prompt + '\n')
                f.write('='*80 + '\n\n')

            response = await self._safe_llm_call_with_retry(
                qa_prompt, 
                f"{sample_id}_direct",
                3
            )
            
            prediction = self.extract_answer_from_response(response)
            
            result = {
                'origin_prediction': prediction,
                'origin_response': response
            }
            
            # 只在有预测结果时打印
            if prediction:
                print(f"Sample {sample_id} - direct: {prediction}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing direct QA for sample {sample_id}: {e}")
            return {
                'origin_prediction': "",
                'origin_response': ""
            }

    async def process_single_format(self, sample: Dict, format_name: str, triples: List[Dict]) -> Dict:
        """处理单个格式 - 独立的异步函数"""
        try:
            sample_id = self.safe_get_value(sample, 'id')
            
            # 转换知识格式 - triples已经在入口处限制了数量
            if format_name == 'edgetable':
                knowledge = self.triples_to_edgetable(triples)
            elif format_name == 'nodesequence':
                knowledge = self.triples_to_nodesequence(triples)
            elif format_name == 'code':
                knowledge = self.triples_to_graphml(triples)
            elif format_name == 'syntaxtree':
                knowledge = self.triples_to_syntaxtree(triples)
            elif format_name == 'naturallanguage':
                knowledge = await self.triples_to_naturallanguage(triples)
            else:
                knowledge = ""
            
            result = {
                f'{format_name}_knowledge': knowledge,
                f'{format_name}_prediction': "",
                f'{format_name}_response': ""
            }
            
            # 进行QA预测
            if knowledge and knowledge.strip():
                # 格式化问题和选项
                formatted_question = self.format_question_with_options(sample)
                
                qa_prompt = QA_PROMPT.format(
                    question=formatted_question,  # 使用格式化后的问题（包含选项）
                    knowledge=knowledge
                )
                
                # 将qa_prompt以追加方式写入文件，包含时间戳和样本ID
                import datetime
                with open('/home/lsz/OneGraph-Service/temp.txt', 'a', encoding='utf-8') as f:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"[{timestamp}] Sample ID: {sample_id}_{format_name}\n")
                    f.write(qa_prompt + '\n')
                    f.write('='*80 + '\n\n')

                response = await self._safe_llm_call_with_retry(
                    qa_prompt, 
                    f"{sample_id}_{format_name}",
                    3
                )
                
                prediction = self.extract_answer_from_response(response)
                result[f'{format_name}_prediction'] = prediction
                result[f'{format_name}_response'] = response
                
                # 只在有预测结果时打印
                if prediction:
                    print(f"Sample {sample_id} - {format_name}: {prediction}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing format {format_name} for sample {sample_id}: {e}")
            return {
                f'{format_name}_knowledge': "",
                f'{format_name}_prediction': "",
                f'{format_name}_response': ""
            }

    async def process_sample(self, sample: Dict, formats: List[str]) -> Dict:
        """处理单个样本 - 真正的并发版本"""
        try:
            sample_id = self.safe_get_value(sample, 'id')
            
            # 提取三元组 - 在这里进行总的数量限制
            triples = self.extract_triples_from_sample(sample)
            
            # 准备基础结果 - 动态获取所有选项字段
            result = {
                'id': sample_id,
                'question': self.safe_get_value(sample, 'question'),
                'answer': self.safe_get_value(sample, 'answer'),
                'triple_count': len(triples)
            }
            
            # 动态添加选项字段
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            for letter in option_letters:
                option_value = self.safe_get_value(sample, letter)
                if option_value:  # 只添加存在的选项
                    result[letter] = option_value
            
            # 准备所有任务
            tasks = []
            
            # 添加直接问答任务（如果在格式列表中）
            if '直接问答' in formats:
                tasks.append(self.process_direct_qa(sample))
            
            # 添加其他格式的任务
            knowledge_formats = [f for f in formats if f != '直接问答']
            if knowledge_formats:
                format_tasks = [
                    self.process_single_format(sample, format_name, triples)
                    for format_name in knowledge_formats
                ]
                tasks.extend(format_tasks)
            
            # 并发处理所有任务
            if tasks:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 合并结果
                for task_result in task_results:
                    if isinstance(task_result, dict):
                        result.update(task_result)
                    else:
                        logger.error(f"Task processing failed for sample {sample_id}: {task_result}")
            
            return result
            
        except Exception as e:
            sample_id = self.safe_get_value(sample, 'id')
            logger.error(f"Error processing sample {sample_id}: {e}")
            # 返回基本信息，即使处理失败
            basic_result = {
                'id': sample_id,
                'question': self.safe_get_value(sample, 'question'),
                'answer': self.safe_get_value(sample, 'answer'),
                'triple_count': 0
            }
            
            # 动态添加选项字段
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            for letter in option_letters:
                option_value = self.safe_get_value(sample, letter)
                if option_value:
                    basic_result[letter] = option_value
            
            return basic_result

    def print_stats(self):
        """打印统计信息"""
        total = self.stats['total_requests']
        if total > 0:
            success_rate = (self.stats['successful_requests'] / total) * 100
            logger.info(f"API调用统计 - 总数: {total}, 成功: {self.stats['successful_requests']}, "
                       f"失败: {self.stats['failed_requests']}, 成功率: {success_rate:.1f}%, "
                       f"超时: {self.stats['timeout_count']}, 重试: {self.stats['retry_count']}")

    def is_file_completed(self, output_path: Path, expected_samples: int) -> bool:
        """检查文件是否已完成处理"""
        if not output_path.exists():
            return False
        
        try:
            df = pd.read_parquet(output_path)
            actual_samples = len(df)
            
            # 如果实际样本数等于预期样本数，认为已完成
            if actual_samples >= expected_samples:
                logger.info(f"文件 {output_path.name} 已完成处理 ({actual_samples}/{expected_samples})")
                return True
            else:
                logger.info(f"文件 {output_path.name} 部分完成 ({actual_samples}/{expected_samples})")
                return False
                
        except Exception as e:
            logger.error(f"检查文件完成状态失败 {output_path}: {e}")
            return False

    async def process_dataset(self, input_path: str, output_path: str, 
                            formats: List[str] = ['直接问答', 'edgetable', 'nodesequence', 'code', 'syntaxtree', 'naturallanguage']):
        """处理整个数据集 - 带检查点功能的版本"""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 验证格式参数
        valid_formats = ['直接问答', 'edgetable', 'nodesequence', 'code', 'syntaxtree', 'naturallanguage']
        invalid_formats = [f for f in formats if f not in valid_formats]
        if invalid_formats:
            logger.error(f"无效的格式: {invalid_formats}")
            logger.info(f"支持的格式: {valid_formats}")
            return
        
        logger.info(f"将处理以下格式: {formats}")
        
        # 找到所有parquet文件
        parquet_files = list(input_path.glob("*.parquet"))
        
        for parquet_file in parquet_files:
            logger.info(f"Processing {parquet_file}")
            
            try:
                # 读取parquet文件
                df = pd.read_parquet(parquet_file)
                samples = df.to_dict('records')
                total_samples = len(samples)
                
                logger.info(f"Found {total_samples} samples in {parquet_file.name}")
                
                # 检查是否已完成
                output_file = output_path / parquet_file.name
                if self.is_file_completed(output_file, total_samples):
                    logger.info(f"跳过已完成的文件: {parquet_file.name}")
                    continue
                
                # 检查是否有检查点
                dataset_name = input_path.name
                file_name = parquet_file.stem
                
                processed_samples = []
                start_batch_index = 0
                
                checkpoint_data = self.checkpoint_manager.load_checkpoint(dataset_name, file_name)
                if checkpoint_data:
                    processed_samples = checkpoint_data['processed_samples']
                    start_batch_index = checkpoint_data['batch_index']
                    self.stats.update(checkpoint_data['stats'])
                    
                    logger.info(f"从检查点恢复: 已处理 {len(processed_samples)} 样本，从批次 {start_batch_index} 开始")
                    
                    # 如果有已处理的样本，先保存到输出文件
                    if processed_samples:
                        processed_df = pd.DataFrame(processed_samples)
                        processed_df.to_parquet(output_file, index=False)
                        logger.info(f"已恢复 {len(processed_samples)} 个已处理样本")
                
                # 计算剩余需要处理的样本
                remaining_samples = samples[len(processed_samples):]
                
                if not remaining_samples:
                    logger.info(f"文件 {parquet_file.name} 已完全处理完成")
                    # 清理检查点
                    self.checkpoint_manager.remove_checkpoint(dataset_name, file_name)
                    continue
                
                logger.info(f"剩余需要处理的样本数: {len(remaining_samples)}")
                
                # 分批处理剩余样本
                checkpoint_interval = max(1, self.process_batch_size // 2)  # 每处理一定数量的批次保存检查点
                
                for i in tqdm(range(0, len(remaining_samples), self.process_batch_size), 
                             desc=f"Processing {parquet_file.name}"):
                    batch = remaining_samples[i:i + self.process_batch_size]
                    current_batch_index = start_batch_index + (i // self.process_batch_size)
                    
                    # 并发处理整个批次
                    batch_start_time = time.time()
                    
                    batch_tasks = [
                        self.process_sample(sample, formats)
                        for sample in batch
                    ]
                    
                    # 等待批次中所有样本处理完成
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # 处理结果
                    valid_results = []
                    for result in batch_results:
                        if isinstance(result, dict):
                            valid_results.append(result)
                        else:
                            logger.error(f"Sample processing failed: {result}")
                            # 添加一个基本的错误结果
                            error_result = {
                                'id': f"error_{len(valid_results)}",
                                'question': "",
                                'A': "", 'B': "", 'C': "", 'D': "",
                                'answer': "",
                                'triple_count': 0
                            }
                            valid_results.append(error_result)
                    
                    # 添加到已处理样本列表
                    processed_samples.extend(valid_results)
                    
                    batch_time = time.time() - batch_start_time
                    
                    # 保存批次结果到文件
                    if processed_samples:
                        batch_df = pd.DataFrame(processed_samples)
                        batch_df.to_parquet(output_file, index=False)
                    
                    # 定期保存检查点
                    if (current_batch_index + 1) % checkpoint_interval == 0:
                        self.checkpoint_manager.save_checkpoint(
                            dataset_name, file_name, processed_samples, 
                            current_batch_index + 1, total_samples, self.stats
                        )
                    
                    # 打印批次统计
                    batch_num = current_batch_index + 1
                    samples_per_sec = len(batch) / batch_time if batch_time > 0 else 0
                    logger.info(f"批次 {batch_num} 完成 - 处理 {len(valid_results)} 样本, "
                              f"总进度 {len(processed_samples)}/{total_samples}, "
                              f"耗时 {batch_time:.1f}s, 速度 {samples_per_sec:.1f} 样本/秒")
                    
                    # 打印API调用统计
                    self.print_stats()
                    
                    # 添加批次间的短暂延迟，避免API限流
                    await asyncio.sleep(0.5)
                
                # 处理完成，清理检查点
                self.checkpoint_manager.remove_checkpoint(dataset_name, file_name)
                logger.info(f"完成处理 {parquet_file.name}，共 {len(processed_samples)} 样本")
                
            except Exception as e:
                logger.error(f"Error processing file {parquet_file}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # 即使出错也保存检查点
                try:
                    dataset_name = input_path.name
                    file_name = parquet_file.stem
                    if 'processed_samples' in locals():
                        self.checkpoint_manager.save_checkpoint(
                            dataset_name, file_name, processed_samples, 
                            current_batch_index if 'current_batch_index' in locals() else 0, 
                            total_samples if 'total_samples' in locals() else 0, 
                            self.stats
                        )
                        logger.info("已保存错误前的检查点")
                except Exception as checkpoint_error:
                    logger.error(f"保存错误检查点失败: {checkpoint_error}")
                
                continue

    def list_progress(self):
        """列出所有检查点的进度"""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            logger.info("没有找到检查点")
            return
        
        logger.info("当前检查点进度:")
        logger.info("-" * 80)
        
        for checkpoint in checkpoints:
            dataset_name = checkpoint.get('dataset_name', 'Unknown')
            file_name = checkpoint.get('file_name', 'Unknown')
            processed_count = checkpoint.get('processed_count', 0)
            total_samples = checkpoint.get('total_samples', 0)
            progress_percentage = checkpoint.get('progress_percentage', 0)
            timestamp = checkpoint.get('timestamp', 'Unknown')
            
            logger.info(f"数据集: {dataset_name}")
            logger.info(f"文件: {file_name}")
            logger.info(f"进度: {processed_count}/{total_samples} ({progress_percentage:.1f}%)")
            logger.info(f"时间: {timestamp}")
            logger.info("-" * 40)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Triple Converter with configurable formats')
    
    parser.add_argument('--formats', nargs='+', 
                       choices=['直接问答', 'edgetable', 'nodesequence', 'code', 'syntaxtree', 'naturallanguage'],
                       default=['直接问答', 'edgetable', 'nodesequence', 'code', 'syntaxtree', 'naturallanguage'],
                       help='选择要处理的格式')
    
    parser.add_argument('--input-path', type=str, 
                       default="/disk0/lsz/datasets/ceval/ceval-exam-enrich-triples",
                       help='输入数据路径')
    
    parser.add_argument('--output-path', type=str,
                       default="/disk0/lsz/datasets/ceval/ceval-exam-retrieve-enrich-prediction-gpt-4o",
                       help='输出数据路径')
    
    parser.add_argument('--batch-size', type=int, default=100,
                       help='批处理大小')
    
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='最大并发数')
    
    parser.add_argument('--max-triples', type=int, default=10,
                       help='每个样本retrieve和enrich三元组数量,例如设置为10则包含10个retrieve和10个enrich三元组，共20个三元组')
    
    parser.add_argument('--checkpoint-dir', type=str, default="./checkpoints",
                       help='检查点目录')
    
    parser.add_argument('--list-progress', action='store_true',
                       help='只列出当前进度，不执行处理')
    
    # 新增参数
    parser.add_argument('--api-base-url', type=str, default="",
                       help='API base URL')
    
    parser.add_argument('--api-keys', nargs='+', default=[""],
                       help='API keys (可以提供多个)')
    
    parser.add_argument('--api-model', type=str, default="",
                       help='API model name')
    
    return parser.parse_args()

async def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 从命令行参数获取配置
    api_base_url = args.api_base_url
    api_keys = args.api_keys
    api_model = args.api_model
    
    # 初始化转换器
    converter = TripleConverter(
        api_base_url=api_base_url,
        api_keys=api_keys,
        api_model=api_model,
        process_batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        max_triples_per_sample=args.max_triples,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # 如果只是查看进度，则显示后退出
    if args.list_progress:
        converter.list_progress()
        return
    
    # 显示当前进度
    converter.list_progress()
    
    # 显示选择的格式
    logger.info(f"选择的处理格式: {args.formats}")
    
    # 获取所有子文件夹
    input_base = Path(args.input_path)
    subdirs = [d for d in input_base.iterdir() if d.is_dir()]
    
    # 处理每个子文件夹
    for subdir in subdirs:
        logger.info(f"开始处理数据集: {subdir.name}")
        
        output_dir = Path(args.output_path) / subdir.name
        
        try:
            dataset_start_time = time.time()
            
            await converter.process_dataset(
                input_path=str(subdir),
                output_path=str(output_dir),
                formats=args.formats
            )
            
            dataset_time = time.time() - dataset_start_time
            logger.info(f"完成处理数据集 {subdir.name}，总耗时 {dataset_time:.1f} 秒")
            
            # 打印最终统计
            converter.print_stats()
            
        except Exception as e:
            logger.error(f"Error processing dataset {subdir.name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

if __name__ == "__main__":
    # 设置事件循环策略以支持更好的并发性能
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())

# usage
# python script.py \
#   --input-path "/path/to/input" \
#   --output-path "/path/to/output" \
#   --max-triples 10 \
#   --batch-size 100 \
#   --max-concurrent 10 \
#   --api-base-url "https://api.example.com" \
#   --api-keys "key1" "key2" "key3" \
#   --api-model "gpt-4o" \