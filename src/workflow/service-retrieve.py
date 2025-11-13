import os
import pandas as pd
import asyncio
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import torch
import multiprocessing as mp
from queue import Queue
import time
import argparse

# 设置多进程启动方法为spawn，解决CUDA多进程问题
mp.set_start_method('spawn', force=True)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

# 导入LLM客户端
from llm.llm_client import llm_client

# 主领域到子领域路径映射
MAIN_FIELD_TO_PATH = {
    'Natural Sciences': 'natural_sciences',
    'Engineering & Technology': 'engineering_technology',
    'Medicine & Health': 'medicine_health',
    'Agriculture': 'agriculture',
    'Social Sciences': 'social_sciences',
    'Humanities': 'humanities',
    'Other': 'other'
}

# 子领域名称映射：原始名称 -> 标准化名称（用于文件路径）
SUBFIELD_NAME_MAPPING = {
    # Natural Sciences
    'mathematics': 'mathematics',
    'physics': 'physics', 
    'chemistry': 'chemistry',
    'astronomy': 'astronomy',
    'earth sciences': 'earth_sciences',
    'biological sciences': 'biological_sciences',
    
    # Engineering & Technology
    'mechanical engineering': 'mechanical_engineering',
    'electrical engineering': 'electrical_engineering',
    'computer science & technology': 'computer_science_technology',
    'materials science & engineering': 'materials_science_engineering',
    'civil engineering': 'civil_engineering',
    'environmental engineering': 'environmental_engineering',
    'aerospace engineering': 'aerospace_engineering',
    
    # Medicine & Health
    'basic medicine': 'basic_medicine',
    'clinical medicine': 'clinical_medicine',
    'pharmacy': 'pharmacy',
    'traditional Chinese medicine': 'traditional_chinese_medicine',
    'public health & preventive medicine': 'public_health_preventive_medicine',
    'nursing': 'nursing',
    
    # Agriculture
    'agronomy': 'agronomy',
    'horticulture': 'horticulture',
    'forestry': 'forestry',
    'veterinary medicine': 'veterinary_medicine',
    'agricultural resources & environment': 'agricultural_resources_environment',
    
    # Social Sciences
    'economics': 'economics',
    'law': 'law',
    'education': 'education',
    'sociology': 'sociology',
    'political science': 'political_science',
    'management': 'management',
    'journalism & communication': 'journalism_communication',
    
    # Humanities
    'linguistics': 'linguistics',
    'literature': 'literature',
    'philosophy': 'philosophy',
    'history': 'history',
    'arts': 'arts',
    
    # Other - 特殊处理
    'other': 'other'
}

# Prompt模板
PROMPT_TEMPLATE = """<<INST>>
<<SYS>>
You are an expert knowledge classifier with deep understanding across multiple academic disciplines.
<</SYS>>

<<instruction>>
Your task: Analyze a given question and identify which specific sub-field(s) are required to answer it properly.

Available Sub-fields:
Natural Sciences Sub-fields:
• mathematics • physics • chemistry • astronomy • earth sciences • biological sciences

Engineering & Technology Sub-fields:
• mechanical engineering • electrical engineering • computer science & technology • materials science & engineering • civil engineering • environmental engineering • aerospace engineering

Medicine & Health Sub-fields:
• basic medicine • clinical medicine • pharmacy • traditional Chinese medicine • public health & preventive medicine • nursing

Agriculture Sub-fields:
• agronomy • horticulture • forestry • veterinary medicine • agricultural resources & environment

Social Sciences Sub-fields:
• economics • law • education • sociology • political science • management • journalism & communication

Humanities Sub-fields:
• linguistics • literature • philosophy • history • arts

• other (for questions not fitting the above sub-fields)

Instructions:
1. Read the question carefully
2. Identify what specific knowledge domains are needed
3. Select the most relevant sub-field(s) from the list above
4. Explain your reasoning briefly
5. Provide your final answer using the exact sub-field names in square brackets

Required Output Format:
Reasoning: [Your brief explanation]
Answer: [sub-field1][sub-field2][...] (use exact names from the list above)

<</instruction>>

### Your Turn
Input: {Question}
<</INST>>"""

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Process CEVAL dataset with triple enrichment')
    
    # 必需参数
    parser.add_argument('--ceval_data_path', type=str, required=True,
                       help='Path to CEVAL dataset directory')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--triple_data_path', type=str, required=True,
                       help='Path to triple data directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to sentence transformer model')
    
    # API相关参数
    parser.add_argument('--api_base_url', type=str, default='https://api.key77qiqi.cn/v1',
                       help='Base URL for the API (default: https://api.key77qiqi.cn/v1)')
    parser.add_argument('--api_keys', type=str, nargs='+', default=None,
                       help='API keys (can specify multiple keys separated by spaces)')
    parser.add_argument('--api_model', type=str, default='gpt-4o-mini-2024-07-18',
                       help='API model name (default: gpt-4o-mini-2024-07-18)')
    
    # 可选参数
    parser.add_argument('--other_data_path', type=str, default=None,
                       help='Path to other domain data file (CSV)')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                       help='Comma-separated GPU IDs to use (default: 0,1,2,3)')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of worker threads (default: 4)')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for processing samples (default: 10)')
    parser.add_argument('--num_embedding_workers', type=int, default=4,
                       help='Number of parallel workers for embedding computation (default: 4)')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of top similar triples to retrieve (default: 100)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    return parser.parse_args()

def compute_embeddings_worker(args):
    """计算嵌入的工作函数，用于多进程处理"""
    subfield, triple_texts, model_path, gpu_id = args
    
    try:
        # 设置当前进程使用的GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # 确保CUDA初始化在子进程中进行
        torch.cuda.set_device(0)  # 使用相对GPU ID 0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Worker on GPU {gpu_id}: Loading model for {subfield}")
        model = SentenceTransformer(model_path, device=device)
        
        logger.info(f"Worker on GPU {gpu_id}: Computing embeddings for {subfield} ({len(triple_texts)} triples)")
        start_time = time.time()
        
        # 分批计算嵌入以避免内存问题
        batch_size = 1000  # 根据GPU内存调整
        all_embeddings = []
        
        for i in range(0, len(triple_texts), batch_size):
            batch_texts = triple_texts[i:i+batch_size]
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        end_time = time.time()
        logger.info(f"Worker on GPU {gpu_id}: Completed embeddings for {subfield} in {end_time - start_time:.2f}s")
        
        return subfield, embeddings
        
    except Exception as e:
        logger.error(f"Error computing embeddings for {subfield} on GPU {gpu_id}: {e}")
        return subfield, None

class GlobalTripleManager:
    """全局三元组管理器，预加载所有三元组数据"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.triple_data = {}  # {subfield: [triples]}
            self.triple_embeddings = {}  # {subfield: embeddings}
            self.model = None
            self.initialized = False
            self.config = None
    
    def initialize(self, config):
        """初始化并预加载所有三元组数据"""
        if self.initialized:
            return
            
        self.config = config
        logger.info("Initializing global triple manager...")
        
        # 主进程中加载一个模型用于后续查询
        self.model = SentenceTransformer(config.model_path)
        
        # 使用映射中的所有子领域（原始名称）
        all_subfields = list(SUBFIELD_NAME_MAPPING.keys())
        
        # 第一阶段：加载所有三元组数据
        logger.info("Phase 1: Loading all triple data...")
        triple_texts_dict = {}
        
        for subfield in all_subfields:
            logger.info(f"Loading triples for subfield: {subfield}")
            triples = self._load_triples_for_subfield(subfield)
            if triples:
                self.triple_data[subfield] = triples
                triple_texts_dict[subfield] = [triple['text'] for triple in triples]
                logger.info(f"Loaded {len(triples)} triples for {subfield}")
            else:
                logger.warning(f"No triples found for subfield: {subfield}")
        
        # 第二阶段：并行计算所有嵌入
        logger.info("Phase 2: Computing embeddings in parallel...")
        self._compute_embeddings_parallel(triple_texts_dict, config.model_path)
        
        self.initialized = True
        logger.info("Global triple manager initialization completed!")
    
    def _compute_embeddings_parallel(self, triple_texts_dict: Dict[str, List[str]], model_path: str):
        """并行计算所有子领域的嵌入"""
        # 准备任务参数
        tasks = []
        available_gpus = [int(gpu) for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
        
        for i, (subfield, triple_texts) in enumerate(triple_texts_dict.items()):
            if triple_texts:  # 只处理有数据的子领域
                gpu_id = available_gpus[i % len(available_gpus)]
                tasks.append((subfield, triple_texts, model_path, gpu_id))
        
        logger.info(f"Starting {len(tasks)} embedding computation tasks across {len(available_gpus)} GPUs")
        
        # 使用进程池并行计算嵌入
        with ProcessPoolExecutor(max_workers=self.config.num_embedding_workers) as executor:
            # 提交所有任务
            future_to_subfield = {
                executor.submit(compute_embeddings_worker, task): task[0] 
                for task in tasks
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_subfield):
                subfield = future_to_subfield[future]
                try:
                    result_subfield, embeddings = future.result()
                    completed += 1
                    
                    if embeddings is not None:
                        self.triple_embeddings[result_subfield] = embeddings
                        logger.info(f"Successfully computed embeddings for {result_subfield} ({completed}/{len(tasks)})")
                    else:
                        logger.error(f"Failed to compute embeddings for {result_subfield} ({completed}/{len(tasks)})")
                        
                except Exception as e:
                    logger.error(f"Exception computing embeddings for {subfield}: {e}")
                    completed += 1
        
        logger.info(f"Completed embedding computation for {len(self.triple_embeddings)} subfields")
    
    def _load_triples_for_subfield(self, subfield: str) -> List[Dict]:
        """加载指定子领域的三元组数据"""
        triples = []
        
        # 特殊处理other子领域
        if subfield == 'other':
            return self._load_other_triples()
        
        # 根据子领域找到对应的文件路径
        main_field = self._get_main_field_for_subfield(subfield)
        if not main_field:
            logger.warning(f"Cannot find main field for subfield: {subfield}")
            return triples
            
        main_field_path = MAIN_FIELD_TO_PATH[main_field]
        
        # 使用映射获取标准化的子领域名称
        if subfield not in SUBFIELD_NAME_MAPPING:
            logger.warning(f"Subfield {subfield} not found in mapping")
            return triples
            
        subfield_clean = SUBFIELD_NAME_MAPPING[subfield]
        subfield_dir = os.path.join(self.config.triple_data_path, main_field_path, subfield_clean)
        
        logger.debug(f"Looking for triples in directory: {subfield_dir}")
        
        # 首先尝试找_all.csv文件
        all_file = os.path.join(subfield_dir, f"{subfield_clean}_all.csv")
        
        if os.path.exists(all_file):
            try:
                logger.debug(f"Loading from file: {all_file}")
                df = pd.read_csv(all_file)
                for _, row in df.iterrows():
                    try:
                        # 先尝试大写的列名
                        head = str(row['Head'])
                        relation = str(row['Relation'])
                        tail = str(row['Tail'])
                    except KeyError:
                        # 如果大写失败，尝试小写的列名
                        head = str(row['head'])
                        relation = str(row['relation'])
                        tail = str(row['tail'])
                    
                    triples.append({
                        'head': head,
                        'relation': relation,
                        'tail': tail,
                        'text': f"{head} {relation} {tail}"
                    })
            except Exception as e:
                logger.error(f"Error loading {all_file}: {e}")
        else:
            # 如果没有_all.csv文件，加载该目录下所有csv文件
            if os.path.exists(subfield_dir):
                logger.debug(f"Directory exists, loading all CSV files from: {subfield_dir}")
                csv_files = [f for f in os.listdir(subfield_dir) if f.endswith('.csv')]
                logger.debug(f"Found CSV files: {csv_files}")
                
                for file in csv_files:
                    try:
                        file_path = os.path.join(subfield_dir, file)
                        logger.debug(f"Loading from file: {file_path}")
                        df = pd.read_csv(file_path)
                        for _, row in df.iterrows():
                            try:
                                # 先尝试大写的列名
                                head = str(row['Head'])
                                relation = str(row['Relation'])
                                tail = str(row['Tail'])
                            except KeyError:
                                # 如果大写失败，尝试小写的列名
                                head = str(row['head'])
                                relation = str(row['relation'])
                                tail = str(row['tail'])
                            
                            triples.append({
                                'head': head,
                                'relation': relation,
                                'tail': tail,
                                'text': f"{head} {relation} {tail}"
                            })
                    except Exception as e:
                        logger.error(f"Error loading {os.path.join(subfield_dir, file)}: {e}")
            else:
                logger.warning(f"Directory does not exist: {subfield_dir}")
        
        return triples
    
    def _load_other_triples(self) -> List[Dict]:
        """特殊处理加载other子领域的三元组数据"""
        triples = []
        
        if self.config.other_data_path and os.path.exists(self.config.other_data_path):
            try:
                logger.debug(f"Loading other triples from: {self.config.other_data_path}")
                df = pd.read_csv(self.config.other_data_path)
                for _, row in df.iterrows():
                    triples.append({
                        'head': str(row['Head']),
                        'relation': str(row['Relation']),
                        'tail': str(row['Tail']),
                        'text': f"{row['Head']} {row['Relation']} {row['Tail']}"
                    })
                logger.info(f"Successfully loaded {len(triples)} triples for 'other' from {self.config.other_data_path}")
            except Exception as e:
                logger.error(f"Error loading other triples from {self.config.other_data_path}: {e}")
        else:
            if self.config.other_data_path:
                logger.warning(f"Other data file does not exist: {self.config.other_data_path}")
            else:
                logger.warning("No other data path specified")
        
        return triples
    
    def _get_main_field_for_subfield(self, subfield: str) -> str:
        """根据子领域获取主领域"""
        natural_sciences = ['mathematics', 'physics', 'chemistry', 'astronomy', 'earth sciences', 'biological sciences']
        engineering_tech = ['mechanical engineering', 'electrical engineering', 'computer science & technology', 
                           'materials science & engineering', 'civil engineering', 'environmental engineering', 'aerospace engineering']
        medicine_health = ['basic medicine', 'clinical medicine', 'pharmacy', 'traditional Chinese medicine', 
                          'public health & preventive medicine', 'nursing']
        agriculture = ['agronomy', 'horticulture', 'forestry', 'veterinary medicine', 'agricultural resources & environment']
        social_sciences = ['economics', 'law', 'education', 'sociology', 'political science', 'management', 'journalism & communication']
        humanities = ['linguistics', 'literature', 'philosophy', 'history', 'arts']
        
        if subfield in natural_sciences:
            return 'Natural Sciences'
        elif subfield in engineering_tech:
            return 'Engineering & Technology'
        elif subfield in medicine_health:
            return 'Medicine & Health'
        elif subfield in agriculture:
            return 'Agriculture'
        elif subfield in social_sciences:
            return 'Social Sciences'
        elif subfield in humanities:
            return 'Humanities'
        else:
            return 'Other'
    
    def retrieve_top_k_triples(self, question: str, subfields: List[str], top_k: int = 100) -> List[Dict]:
        """检索与问题最相关的top-k三元组"""
        all_triples = []
        all_embeddings = []
        
        # 收集所有相关子领域的三元组和嵌入
        for subfield in subfields:
            if subfield in self.triple_data and subfield in self.triple_embeddings:
                triples = self.triple_data[subfield]
                embeddings = self.triple_embeddings[subfield]
                all_triples.extend(triples)
                all_embeddings.extend(embeddings)
        
        if not all_triples:
            logger.warning(f"No triples found for subfields: {subfields}")
            return []
        
        # 计算问题嵌入
        question_embedding = self.model.encode([question])
        all_embeddings = np.array(all_embeddings)
        
        # 计算相似度
        similarities = cosine_similarity(question_embedding, all_embeddings)[0]
        
        # 获取top-k最相似的三元组
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        top_k_triples = []
        for idx in top_k_indices:
            triple = all_triples[idx].copy()
            triple['similarity'] = float(similarities[idx])
            top_k_triples.append(triple)
        
        return top_k_triples

def parse_subfields_from_response(response: str) -> List[str]:
    """从LLM响应中解析子领域 - 改进版本"""
    logger.debug(f"Raw LLM response: {response}")
    
    # 定义所有可能的子领域（使用原始名称）
    all_subfields = list(SUBFIELD_NAME_MAPPING.keys())
    
    subfields = []
    response_lower = response.lower()
    
    # 方法1: 尝试多种正则表达式模式
    patterns = [
        r'$$([^$$]+)\]',  # 标准方括号
        r'answer:\s*$$([^$$]+)\]',  # Answer: [...]
        r'answer:\s*([^.\n]+)',  # Answer: ... (没有方括号)
        r'final answer[:\s]*$$([^$$]+)\]',  # Final answer: [...]
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            # 清理匹配的内容
            match_clean = match.strip()
            if 'explanation' not in match_clean.lower() and 'reasoning' not in match_clean.lower():
                # 分割多个子领域（可能用逗号、分号或其他分隔符分隔）
                potential_fields = re.split(r'[,;|&]+|\]\s*\[', match_clean)
                for field in potential_fields:
                    field_clean = field.strip().strip('[]')
                    if field_clean and len(field_clean) > 2:  # 避免空字符串或过短的匹配
                        subfields.append(field_clean)
    
    # 方法2: 如果正则表达式失败，尝试直接在文本中搜索子领域名称
    if not subfields:
        logger.warning("Regex extraction failed, trying direct text matching")
        for subfield in all_subfields:
            # 检查子领域是否在响应中出现
            if subfield.lower() in response_lower:
                # 进一步验证：确保不是作为其他词的一部分出现
                pattern = r'\b' + re.escape(subfield.lower()) + r'\b'
                if re.search(pattern, response_lower):
                    subfields.append(subfield)
    
    # 方法3: 如果仍然没有找到，尝试关键词匹配
    if not subfields:
        logger.warning("Direct matching failed, trying keyword matching")
        keyword_mapping = {
            'math': 'mathematics',
            'mathematical': 'mathematics',
            'physics': 'physics',
            'physical': 'physics',
            'chemistry': 'chemistry',
            'chemical': 'chemistry',
            'biology': 'biological sciences',
            'biological': 'biological sciences',
            'computer': 'computer science & technology',
            'programming': 'computer science & technology',
            'software': 'computer science & technology',
            'medicine': 'basic medicine',
            'medical': 'basic medicine',
            'health': 'basic medicine',
            'economic': 'economics',
            'legal': 'law',
            'literature': 'literature',
            'history': 'history',
            'language': 'linguistics',
            'art': 'arts',
            'engineering': 'mechanical engineering',  # 默认工程类别
        }
        
        for keyword, subfield in keyword_mapping.items():
            if keyword in response_lower:
                subfields.append(subfield)
                break  # 只取第一个匹配的关键词
    
    # 去重并验证
    valid_subfields = []
    for subfield in subfields:
        if subfield.lower() in [sf.lower() for sf in all_subfields]:
            # 找到精确匹配的子领域名称
            for valid_sf in all_subfields:
                if subfield.lower() == valid_sf.lower():
                    if valid_sf not in valid_subfields:
                        valid_subfields.append(valid_sf)
                    break
    
    # 如果仍然没有找到任何子领域，默认返回'other'
    if not valid_subfields:
        logger.warning("No valid subfields found, defaulting to 'other'")
        valid_subfields = ['other']
    
    logger.debug(f"Extracted subfields: {valid_subfields}")
    return valid_subfields

async def process_sample(sample: Dict, llm, triple_manager: GlobalTripleManager, config) -> Dict:
    """处理单个样本"""
    question = sample['question']
    
    # Step 1: 获取子领域
    prompt = PROMPT_TEMPLATE.format(Question=question)
    try:
        response = await llm.response(prompt)
        subfields = parse_subfields_from_response(response)
        logger.debug(f"Question: {question[:50]}... -> Subfields: {subfields}")
    except Exception as e:
        logger.error(f"Error getting subfields for question: {question[:50]}... Error: {e}")
        subfields = ['other']  # 默认分类
    
    # Step 2: 检索相关三元组（使用预加载的数据）
    try:
        related_triples = triple_manager.retrieve_top_k_triples(question, subfields, config.top_k)
        logger.debug(f"Retrieved {len(related_triples)} triples for question")
    except Exception as e:
        logger.error(f"Error retrieving triples: {e}")
        related_triples = []
    
    # 添加新字段
    sample['subdomains'] = subfields
    sample['relatedTriples'] = related_triples
    
    return sample

async def process_batch(samples: List[Dict], llm, triple_manager: GlobalTripleManager, config) -> List[Dict]:
    """批量处理样本"""
    tasks = []
    for sample in samples:
        task = process_sample(sample, llm, triple_manager, config)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_samples = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error processing sample {i}: {result}")
            # 添加默认值
            sample = samples[i].copy()
            sample['subdomains'] = ['other']
            sample['relatedTriples'] = []
            processed_samples.append(sample)
        else:
            processed_samples.append(result)
    
    return processed_samples

def process_subject_worker(subject: str, worker_id: int, config) -> bool:
    """工作线程处理单个学科"""
    try:
        logger.info(f"Worker {worker_id}: Processing subject: {subject}")
        
        # 每个工作线程创建自己的LLM客户端，使用配置参数
        llm = llm_client(
            base_url=config.api_base_url,
            api_keys=config.api_keys,
            model=config.api_model
        )
        triple_manager = GlobalTripleManager()
        
        # 读取测试数据
        subject_path = os.path.join(config.ceval_data_path, subject)
        test_files = [f for f in os.listdir(subject_path) if f.startswith('test') and f.endswith('.parquet')]
        
        if not test_files:
            logger.warning(f"Worker {worker_id}: No test files found for subject: {subject}")
            return False
        
        # 处理每个测试文件
        for test_file in test_files:
            file_path = os.path.join(subject_path, test_file)
            df = pd.read_parquet(file_path)
            
            # 将样本分批处理
            samples = df.to_dict('records')
            processed_samples = []
            
            for i in range(0, len(samples), config.batch_size):
                batch = samples[i:i+config.batch_size]
                logger.info(f"Worker {worker_id}: Processing batch {i//config.batch_size + 1}/{(len(samples)-1)//config.batch_size + 1} for {subject}/{test_file}")
                
                # 使用异步处理批次
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    batch_results = loop.run_until_complete(process_batch(batch, llm, triple_manager, config))
                    processed_samples.extend(batch_results)
                finally:
                    loop.close()
            
            # 保存处理后的数据
            output_dir = os.path.join(config.output_path, subject)
            os.makedirs(output_dir, exist_ok=True)
            
            output_df = pd.DataFrame(processed_samples)
            output_file = os.path.join(output_dir, test_file)
            output_df.to_parquet(output_file, index=False)
            
            logger.info(f"Worker {worker_id}: Saved processed data to: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Worker {worker_id}: Error processing subject {subject}: {e}")
        return False

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    logger.info(f"Using GPUs: {args.gpu_ids}")
    
    start_time = time.time()
    
    # 初始化全局三元组管理器
    logger.info("Starting initialization...")
    triple_manager = GlobalTripleManager()
    triple_manager.initialize(args)
    
    init_time = time.time()
    logger.info(f"Initialization completed in {init_time - start_time:.2f} seconds")
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 获取所有学科
    subjects = [d for d in os.listdir(args.ceval_data_path) if os.path.isdir(os.path.join(args.ceval_data_path, d))]
    
    logger.info(f"Found {len(subjects)} subjects to process")
    
    # 使用线程池并行处理学科
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        future_to_subject = {
            executor.submit(process_subject_worker, subject, i % args.max_workers, args): subject 
            for i, subject in enumerate(subjects)
        }
        
        # 等待完成并收集结果
        completed = 0
        for future in as_completed(future_to_subject):
            subject = future_to_subject[future]
            try:
                success = future.result()
                completed += 1
                if success:
                    logger.info(f"Successfully completed {subject} ({completed}/{len(subjects)})")
                else:
                    logger.error(f"Failed to process {subject} ({completed}/{len(subjects)})")
            except Exception as e:
                logger.error(f"Exception processing {subject}: {e}")
                completed += 1
    
    total_time = time.time()
    logger.info(f"Processing completed in {total_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()

# uasge:
# python src/workflow/service-retrieve.py \
    # --api_base_url "https://your-custom-api.com/v1" \
    # --api_keys "key1" "key2" "key3" \
    # --api_model "gpt-4-turbo" \
    # --ceval_data_path "/disk0/lsz/datasets/ceval/ceval-exam" \
    # --output_path "/disk0/lsz/datasets/ceval/ceval-exam-added-triples" \
    # --triple_data_path "/disk0/lsz/OneGraph/sorted_data_v2" \
    # --model_path "/disk0/lsz/PLMs/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
    # --other_data_path "/home/lsz/OneGraph/sorted_data_v1/other_all.csv" \
    # --gpu_ids "0,1,2,3" \
    # --max_workers 4 \
    # --batch_size 10 \
    # --num_embedding_workers 4 \
    # --top_k 100 \
    # --log_level INFO