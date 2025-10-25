import os
import asyncio
import pandas as pd
from pathlib import Path
from datasets import Dataset, load_dataset
import logging
from typing import List, Dict, Any
import json
import sys
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from llm.llm_client import llm_client

# 设置日志
def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

class TripleEnricher:
    def __init__(self, input_dir: str, output_dir: str, 
                 base_url: str = "https://api.key77qiqi.cn/v1", 
                 api_keys: List[str] = None, 
                 model: str = 'gpt-4o-mini-2024-07-18',
                 prompt_template: str = None):
        # 初始化LLM客户端
        self.llm = llm_client(base_url=base_url, api_keys=api_keys, model=model)
        
        # 路径配置
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API配置（用于日志记录）
        self.base_url = base_url
        self.api_keys = api_keys
        self.model = model
        
        # 使用自定义prompt模板或默认模板
        self.prompt_template = prompt_template or self._get_default_prompt_template()
        
        self.logger = logging.getLogger(__name__)
        
        # 记录API配置信息
        self.logger.info(f"Initialized LLM client with:")
        self.logger.info(f"  - Base URL: {self.base_url}")
        self.logger.info(f"  - Model: {self.model}")
        self.logger.info(f"  - API Keys count: {len(api_keys) if api_keys else 0}")

    def _get_default_prompt_template(self) -> str:
        """获取默认的prompt模板"""
        return """<<INST>>
<<SYS>>
You are a linguist with expertise in semantic analysis and knowledge graph enhancement. You can understand user queries and identify relevant semantic information in triples. Your goal is to enhance the semantic connections between queries and triples by applying similarity, symmetry, transitivity properties to triples, and hierarchical relationships to entities.
<</SYS>>

<<instruction>>
You will be given a question and a set of triples. Your task is to:
1. Use your internal knowledge to identify triples and entities that are relevant to the question
2. Apply semantic enhancement to the relevant triples using similarity, symmetry, and transitivity properties
3. Apply hierarchical enhancement to the relevant entities  
4. Output the enhanced triples in the specified format using []

Enhancement Properties:

**Similarity**: For a given triple (e1, r1, e2), you can derive [e1, r2, e2] where r2 is semantically different from r1 but maintains the same directional connection between e1 and e2.

**Symmetry**: For a given triple (e1, r1, e2), you can derive [e2, r2, e1] where r2 is semantically different from r1 and reverses the connection direction.

**Transitivity**: Given triples (e1, r1, e2) and (e2, r2, e3) that share entity e2, you can derive [e1, r3, e3] where r3 combines the semantic meanings of r1 and r2.

**Hierarchy**: For relevant entities, add appropriate hierarchical relationships using these ontology relations:
- Hypernym_isA: A is a type of B
- Hypernym_locateAt: A is located at B  
- Hypernym_mannerOf: A is a specific way of doing B
- Induction_belongTo: A belongs to category B
- Inclusion_isPartOf: A is a part of B
- Inclusion_madeOf: A is made of B
- Inclusion_derivedFrom: A is derived from B
- Inclusion_hasContext: A is used in the context of B

Procedure:
Step 1: Analyze the given question and triples. Identify which triples and entities are relevant to answering the question based on your internal knowledge.

Step 2: Apply similarity and symmetry properties to enhance the relevant triples.

Step 3: Apply transitivity property to create new multi-hop connections between relevant entities.

Step 4: Apply hierarchical enhancement to relevant entities by adding appropriate ontology relationships.

Step 5: Output all enhanced triples in the specified format using [].

Output Format:
Enhanced Triples:
[entity1, relation, entity2]
[entity1, relation, entity2]
...

Hierarchical Enhancements:
[entity, ontology_relation, ontology_concept]
[entity, ontology_relation, ontology_concept]
...

<</instruction>>

### Your Turn
Input:
Question:
{question}
Triples:
{triples}

<</INST>>"""

    def get_subject_folders(self) -> List[str]:
        """获取所有subject文件夹"""
        subjects = []
        for item in self.input_dir.iterdir():
            if item.is_dir():
                subjects.append(item.name)
        return sorted(subjects)

    def load_subject_data(self, subject: str) -> Dataset:
        """加载指定subject的数据"""
        subject_dir = self.input_dir / subject
        
        try:
            # 使用huggingface datasets库加载parquet文件
            dataset = load_dataset("parquet", data_dir=str(subject_dir), split="test")
            self.logger.info(f"Loaded {len(dataset)} samples for subject: {subject}")
            return dataset
        except Exception as e:
            self.logger.error(f"Error loading data for subject {subject}: {e}")
            return Dataset.from_dict({})

    def format_triples_from_list(self, related_triples: List[Dict]) -> str:
        """将三元组列表格式化为字符串"""
        if not related_triples:
            return "No triples available."
        
        formatted_triples = []
        for triple in related_triples:
            try:
                head = triple.get('head', '')
                relation = triple.get('relation', '')
                tail = triple.get('tail', '')
                similarity = triple.get('similarity', 0.0)
                
                # 格式化三元组
                triple_str = f"[{head}, {relation}, {tail}] (similarity: {similarity:.3f})"
                formatted_triples.append(triple_str)
            except Exception as e:
                self.logger.warning(f"Error formatting triple {triple}: {e}")
                continue
        
        return "\n".join(formatted_triples)

    async def enrich_sample(self, question: str, related_triples: List[Dict], delay: float = 0.1) -> str:
        """使用LLM增强单个样本的三元组"""
        try:
            # 格式化三元组
            triples_str = self.format_triples_from_list(related_triples)
            
            # 构建prompt - 使用安全的字符串格式化
            formatted_prompt = self.prompt_template.format(
                question=question,
                triples=triples_str
            )
            
            # 调用LLM API
            response = await self.llm.response(formatted_prompt)
            
            # 添加延迟避免API限制
            if delay > 0:
                await asyncio.sleep(delay)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error enriching sample: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def parse_enriched_response(self, response: str) -> Dict[str, Any]:
        """解析LLM返回的增强三元组"""
        try:
            # 简单的解析逻辑，可以根据实际返回格式调整
            enhanced_triples = []
            hierarchical_enhancements = []
            
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if "Enhanced Triples:" in line:
                    current_section = "enhanced"
                    continue
                elif "Hierarchical Enhancements:" in line:
                    current_section = "hierarchical"
                    continue
                
                # 解析三元组格式 [entity1, relation, entity2]
                if line.startswith('[') and line.endswith(']'):
                    try:
                        # 移除方括号并分割
                        content = line[1:-1]
                        parts = [part.strip() for part in content.split(',')]
                        if len(parts) == 3:
                            triple_dict = {
                                "head": parts[0],
                                "relation": parts[1],
                                "tail": parts[2],
                                "source": "enriched"
                            }
                            
                            if current_section == "enhanced":
                                enhanced_triples.append(triple_dict)
                            elif current_section == "hierarchical":
                                hierarchical_enhancements.append(triple_dict)
                    except Exception as e:
                        self.logger.warning(f"Error parsing triple line '{line}': {e}")
                        continue
            
            return {
                "enhanced_triples": enhanced_triples,
                "hierarchical_enhancements": hierarchical_enhancements,
                "raw_response": response
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing enriched response: {e}")
            return {
                "enhanced_triples": [],
                "hierarchical_enhancements": [],
                "raw_response": response
            }

    async def process_subject(self, subject: str, delay: float = 0.1):
        """处理单个subject的所有数据"""
        self.logger.info(f"Processing subject: {subject}")
        
        # 加载数据
        dataset = self.load_subject_data(subject)
        if len(dataset) == 0:
            self.logger.warning(f"No data found for subject: {subject}")
            return
        
        # 检查必要的字段是否存在
        required_fields = ['question', 'relatedTriples']
        sample = dataset[0]
        missing_fields = [field for field in required_fields if field not in sample]
        if missing_fields:
            self.logger.error(f"Missing fields in {subject}: {missing_fields}")
            return
        
        # 处理每个样本
        enriched_data = []
        
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                question = sample['question']
                related_triples = sample['relatedTriples']
                
                self.logger.info(f"Processing {subject} sample {idx + 1}/{len(dataset)}")
                self.logger.debug(f"Question: {question[:100]}...")
                self.logger.debug(f"Related triples count: {len(related_triples) if related_triples else 0}")
                
                # 增强三元组
                enriched_response = await self.enrich_sample(question, related_triples, delay)
                parsed_enrichment = self.parse_enriched_response(enriched_response)
                
                # 创建新的样本数据
                new_sample = dict(sample)  # 复制原始数据
                new_sample['enrich_triples'] = parsed_enrichment['enhanced_triples']
                new_sample['hierarchical_enhancements'] = parsed_enrichment['hierarchical_enhancements']
                new_sample['enrichment_raw_response'] = parsed_enrichment['raw_response']
                
                enriched_data.append(new_sample)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {idx} in {subject}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # 添加原始数据，但增强字段为空
                new_sample = dict(dataset[idx])
                new_sample['enrich_triples'] = []
                new_sample['hierarchical_enhancements'] = []
                new_sample['enrichment_raw_response'] = ""
                enriched_data.append(new_sample)
        
        # 保存结果
        await self.save_subject_data(subject, enriched_data)

    async def save_subject_data(self, subject: str, enriched_data: List[Dict]):
        """保存处理后的数据"""
        try:
            # 创建输出目录
            output_subject_dir = self.output_dir / subject
            output_subject_dir.mkdir(parents=True, exist_ok=True)
            
            # 转换为Dataset并保存为parquet
            dataset = Dataset.from_list(enriched_data)
            output_file = output_subject_dir / "test-00000-of-00001.parquet"
            dataset.to_parquet(str(output_file))
            
            self.logger.info(f"Saved enriched data for {subject} to {output_file}")
            self.logger.info(f"Total samples saved: {len(enriched_data)}")
            
        except Exception as e:
            self.logger.error(f"Error saving data for {subject}: {e}")

    async def process_all_subjects(self, delay: float = 0.1, subjects: List[str] = None):
        """处理所有subject或指定的subjects"""
        if subjects is None:
            subjects = self.get_subject_folders()
        
        self.logger.info(f"Found {len(subjects)} subjects to process: {subjects}")
        
        for i, subject in enumerate(subjects, 1):
            try:
                self.logger.info(f"Processing subject {i}/{len(subjects)}: {subject}")
                await self.process_subject(subject, delay)
                self.logger.info(f"Completed processing {subject}")
            except Exception as e:
                self.logger.error(f"Error processing subject {subject}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                continue

    async def process_parallel(self, max_concurrent: int = 2, delay: float = 0.1, subjects: List[str] = None):
        """并行处理多个subject"""
        if subjects is None:
            subjects = self.get_subject_folders()
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(subject):
            async with semaphore:
                await self.process_subject(subject, delay)
        
        # 并行处理所有subject
        tasks = [process_with_semaphore(subject) for subject in subjects]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_single_sample(self, question: str = None, triples: List[Dict] = None):
        """测试单个样本的处理"""
        # 使用默认测试数据或提供的数据
        if question is None:
            question = "下列关于资本结构理论的说法中，不正确的是____。"
        
        if triples is None:
            triples = [
                {
                    "head": "资本结构理论",
                    "relation": "provides",
                    "similarity": 0.8103655576705933,
                    "tail": "理论依据",
                    "text": "资本结构理论 provides 理论依据"
                },
                {
                    "head": "菲舍尔等人",
                    "relation": "提出",
                    "similarity": 0.7829468250274658,
                    "tail": "资本结构理论",
                    "text": "菲舍尔等人 提出 资本结构理论"
                }
            ]
        
        result = await self.enrich_sample(question, triples)
        print("Enrichment result:")
        print(result)
        
        parsed = self.parse_enriched_response(result)
        print("\nParsed result:")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        
        return parsed

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Triple Enricher - Enhance knowledge graph triples using LLM")
    
    # 必需参数
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                       help="Input directory containing subject folders with parquet files")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                       help="Output directory to save enriched data")
    
    # LLM API 参数
    parser.add_argument("--base_url", type=str, default="https://api.key77qiqi.cn/v1",
                       help="Base URL for LLM API (default: https://api.key77qiqi.cn/v1)")
    parser.add_argument("--api_keys", type=str, nargs="+", default=None,
                       help="API keys for LLM (can provide multiple keys)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18",
                       help="LLM model name (default: gpt-4o-mini-2024-07-18)")
    
    # 可选参数
    parser.add_argument("--mode", "-m", type=str, choices=["serial", "parallel", "test"], default="serial",
                       help="Processing mode: serial, parallel, or test (default: serial)")
    parser.add_argument("--max_concurrent", "-c", type=int, default=2,
                       help="Maximum concurrent processes for parallel mode (default: 2)")
    parser.add_argument("--delay", "-d", type=float, default=0.1,
                       help="Delay between API calls in seconds (default: 0.1)")
    parser.add_argument("--subjects", "-s", type=str, nargs="+", default=None,
                       help="Specific subjects to process (default: all subjects)")
    parser.add_argument("--log_level", "-l", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    parser.add_argument("--prompt_file", "-p", type=str, default=None,
                       help="Path to custom prompt template file")
    
    # 测试模式参数
    parser.add_argument("--test_question", type=str, default=None,
                       help="Test question for test mode")
    parser.add_argument("--test_triples_file", type=str, default=None,
                       help="JSON file containing test triples for test mode")
    
    return parser.parse_args()

def load_prompt_template(prompt_file: str) -> str:
    """从文件加载prompt模板"""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading prompt template from {prompt_file}: {e}")
        return None

def load_test_triples(triples_file: str) -> List[Dict]:
    """从JSON文件加载测试三元组"""
    try:
        with open(triples_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading test triples from {triples_file}: {e}")
        return None

async def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    logger.info(f"Starting Triple Enricher with arguments: {vars(args)}")
    
    # 加载自定义prompt模板（如果提供）
    prompt_template = None
    if args.prompt_file:
        prompt_template = load_prompt_template(args.prompt_file)
        if prompt_template is None:
            logger.error("Failed to load prompt template, using default")
    
    # 创建TripleEnricher实例，传入API参数
    enricher = TripleEnricher(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        base_url=args.base_url,
        api_keys=args.api_keys,
        model=args.model,
        prompt_template=prompt_template
    )
    
    # 根据模式执行不同的操作
    if args.mode == "test":
        # 测试模式
        test_question = args.test_question
        test_triples = None
        
        if args.test_triples_file:
            test_triples = load_test_triples(args.test_triples_file)
        
        await enricher.test_single_sample(test_question, test_triples)
        
    elif args.mode == "parallel":
        # 并行处理模式
        logger.info(f"Running in parallel mode with max_concurrent={args.max_concurrent}")
        await enricher.process_parallel(
            max_concurrent=args.max_concurrent,
            delay=args.delay,
            subjects=args.subjects
        )
        
    else:
        # 串行处理模式
        logger.info("Running in serial mode")
        await enricher.process_all_subjects(
            delay=args.delay,
            subjects=args.subjects
        )
    
    logger.info("Triple enrichment process completed!")

if __name__ == "__main__":
    asyncio.run(main())

# 完整配置示例
# python script.py \
#   --input_dir "/disk0/lsz/datasets/ceval/ceval-exam-added-triples" \
#   --output_dir "/disk0/lsz/datasets/ceval/ceval-exam-enrich-triples" \
#   --base_url "https://api.key77qiqi.cn/v1" \
#   --api_keys "key1" "key2" \
#   --model "gpt-4o-mini-2024-07-18" \
#   --mode parallel \
#   --max_concurrent 5 \
#   --log_level DEBUG