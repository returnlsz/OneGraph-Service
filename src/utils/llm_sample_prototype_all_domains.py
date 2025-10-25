import pandas as pd
import random
import re
import os
import sys
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from llm.llm_client import llm_client  # 假设你的LLM客户端在这个模块中

# 原有的主领域模板定义
JUDGE_TEMPLATE = '''
<<INST>>
<<SYS>>
You are an experienced linguist.
<</SYS>>
<<instruction>>
Given a text, please determine which field(s) the text belongs to from the categories below. Please provide your reasoning process and final answer, with the answer enclosed in square brackets [].

Fields are as follows:
Natural Sciences: including mathematics, physics, chemistry, astronomy, earth sciences, biological sciences
Engineering & Technology: including mechanical engineering, electrical engineering, computer science & technology, materials science & engineering, civil engineering, environmental engineering, aerospace engineering
Medicine & Health: including basic medicine, clinical medicine, pharmacy, traditional Chinese medicine, public health & preventive medicine, nursing
Agriculture: including agronomy, horticulture, forestry, veterinary medicine, agricultural resources & environment
Social Sciences: including economics, law, education, sociology, political science, management, journalism & communication
Humanities: including linguistics, literature, philosophy, history, arts
Other

Your final answer must only include one or a combination of the following: [Natural Sciences][Engineering & Technology][Medicine & Health][Agriculture][Social Sciences][Humanities][Other]
<</instruction>>
### Your Turn
Input:
{text}
<</INST>>
'''

# 子领域模板定义
JUDGE_NATURAL_SCIENCES_TEMPLATE = '''
<<INST>>
<<SYS>>
You are an experienced linguist.
<</SYS>>
<<instruction>>
Given a text, please determine which field(s) the text belongs to from the categories below. Please provide your reasoning process and final answer, with the answer enclosed in square brackets [].

Fields are as follows:
mathematics, physics, chemistry, astronomy, earth sciences, biological sciences

Your final answer must only include one or a combination of the following: [mathematics][physics][chemistry][astronomy][earth sciences][biological sciences]
<</instruction>>
### Your Turn
Input:
{text}
<</INST>>
'''

JUDGE_ENGINEERING_TECHNOLOGY_TEMPLATE = '''
<<INST>>
<<SYS>>
You are an experienced linguist.
<</SYS>>
<<instruction>>
Given a text, please determine which field(s) the text belongs to from the categories below. Please provide your reasoning process and final answer, with the answer enclosed in square brackets [].

Fields are as follows:
mechanical engineering, electrical engineering, computer science & technology, materials science & engineering, civil engineering, environmental engineering, aerospace engineering

Your final answer must only include one or a combination of the following: [mechanical engineering][electrical engineering][computer science & technology][materials science & engineering][civil engineering][environmental engineering][aerospace engineering]
<</instruction>>
### Your Turn
Input:
{text}
<</INST>>
'''

JUDGE_MEDICINE_HEALTH_TEMPLATE = '''
<<INST>>
<<SYS>>
You are an experienced linguist.
<</SYS>>
<<instruction>>
Given a text, please determine which field(s) the text belongs to from the categories below. Please provide your reasoning process and final answer, with the answer enclosed in square brackets [].

Fields are as follows:
basic medicine, clinical medicine, pharmacy, traditional Chinese medicine, public health & preventive medicine, nursing

Your final answer must only include one or a combination of the following: [basic medicine][clinical medicine][pharmacy][traditional Chinese medicine][public health & preventive medicine][nursing]
<</instruction>>
### Your Turn
Input:
{text}
<</INST>>
'''

JUDGE_ARGRICULTURE_TEMPLATE = '''
<<INST>>
<<SYS>>
You are an experienced linguist.
<</SYS>>
<<instruction>>
Given a text, please determine which field(s) the text belongs to from the categories below. Please provide your reasoning process and final answer, with the answer enclosed in square brackets [].

Fields are as follows:
agronomy, horticulture, forestry, veterinary medicine, agricultural resources & environment

Your final answer must only include one or a combination of the following: [agronomy][horticulture][forestry][veterinary medicine][agricultural resources & environment]
<</instruction>>
### Your Turn
Input:
{text}
<</INST>>
'''

JUDGE_SOCIAL_SCIENCES_TEMPLATE = '''
<<INST>>
<<SYS>>
You are an experienced linguist.
<</SYS>>
<<instruction>>
Given a text, please determine which field(s) the text belongs to from the categories below. Please provide your reasoning process and final answer, with the answer enclosed in square brackets [].

Fields are as follows:
economics, law, education, sociology, political science, management, journalism & communication

Your final answer must only include one or a combination of the following: [economics][law][education][sociology][political science][management][journalism & communication]
<</instruction>>
### Your Turn
Input:
{text}
<</INST>>
'''

JUDGE_HUMANITIES_TEMPLATE = '''
<<INST>>
<<SYS>>
You are an experienced linguist.
<</SYS>>
<<instruction>>
Given a text, please determine which field(s) the text belongs to from the categories below. Please provide your reasoning process and final answer, with the answer enclosed in square brackets [].

Fields are as follows:
linguistics, literature, philosophy, history, arts

Your final answer must only include one or a combination of the following: [linguistics][literature][philosophy][history][arts]
<</instruction>>
### Your Turn
Input:
{text}
<</INST>>
'''

# 主领域映射到文件名
FIELD_TO_FILENAME = {
    'Natural Sciences': 'natural_sciences.csv',
    'Engineering & Technology': 'engineering_technology.csv',
    'Medicine & Health': 'medicine_health.csv',
    'Agriculture': 'agriculture.csv',
    'Social Sciences': 'social_sciences.csv',
    'Humanities': 'humanities.csv',
    'Other': 'other.csv'
}

# 子领域映射到文件名
NATURAL_SCIENCES_TO_FILENAME = {
    'mathematics': 'mathematics.csv',
    'physics': 'physics.csv',
    'chemistry': 'chemistry.csv',
    'astronomy': 'astronomy.csv',
    'earth sciences': 'earth_sciences.csv',
    'biological sciences': 'biological_sciences.csv'
}

ENGINEERING_TECHNOLOGY_TO_FILENAME = {
    'mechanical engineering': 'mechanical_engineering.csv',
    'electrical engineering': 'electrical_engineering.csv',
    'computer science & technology': 'computer_science_technology.csv',
    'materials science & engineering': 'materials_science_engineering.csv',
    'civil engineering': 'civil_engineering.csv',
    'environmental engineering': 'environmental_engineering.csv',
    'aerospace engineering': 'aerospace_engineering.csv'
}

MEDICINE_HEALTH_TO_FILENAME = {
    'basic medicine': 'basic_medicine.csv',
    'clinical medicine': 'clinical_medicine.csv',
    'pharmacy': 'pharmacy.csv',
    'traditional Chinese medicine': 'traditional_chinese_medicine.csv',
    'public health & preventive medicine': 'public_health_preventive_medicine.csv',
    'nursing': 'nursing.csv'
}

AGRICULTURE_TO_FILENAME = {
    'agronomy': 'agronomy.csv',
    'horticulture': 'horticulture.csv',
    'forestry': 'forestry.csv',
    'veterinary medicine': 'veterinary_medicine.csv',
    'agricultural resources & environment': 'agricultural_resources_environment.csv'
}

SOCIAL_SCIENCES_TO_FILENAME = {
    'economics': 'economics.csv',
    'law': 'law.csv',
    'education': 'education.csv',
    'sociology': 'sociology.csv',
    'political science': 'political_science.csv',
    'management': 'management.csv',
    'journalism & communication': 'journalism_communication.csv'
}

HUMANITIES_TO_FILENAME = {
    'linguistics': 'linguistics.csv',
    'literature': 'literature.csv',
    'philosophy': 'philosophy.csv',
    'history': 'history.csv',
    'arts': 'arts.csv'
}

# 子领域映射配置
SUBDOMAIN_CONFIG = {
    'Natural Sciences': {
        'template': JUDGE_NATURAL_SCIENCES_TEMPLATE,
        'field_mapping': NATURAL_SCIENCES_TO_FILENAME
    },
    'Engineering & Technology': {
        'template': JUDGE_ENGINEERING_TECHNOLOGY_TEMPLATE,
        'field_mapping': ENGINEERING_TECHNOLOGY_TO_FILENAME
    },
    'Medicine & Health': {
        'template': JUDGE_MEDICINE_HEALTH_TEMPLATE,
        'field_mapping': MEDICINE_HEALTH_TO_FILENAME
    },
    'Agriculture': {
        'template': JUDGE_ARGRICULTURE_TEMPLATE,
        'field_mapping': AGRICULTURE_TO_FILENAME
    },
    'Social Sciences': {
        'template': JUDGE_SOCIAL_SCIENCES_TEMPLATE,
        'field_mapping': SOCIAL_SCIENCES_TO_FILENAME
    },
    'Humanities': {
        'template': JUDGE_HUMANITIES_TEMPLATE,
        'field_mapping': HUMANITIES_TO_FILENAME
    }
}

# 添加文件锁来避免并发写入冲突
import threading
# 为主领域和所有子领域创建文件锁
all_filenames = list(FIELD_TO_FILENAME.values())
for subdomain_info in SUBDOMAIN_CONFIG.values():
    all_filenames.extend(subdomain_info['field_mapping'].values())
file_locks = {filename: threading.Lock() for filename in all_filenames}

def load_data(file_path):
    """加载CSV文件"""
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载数据，共 {len(df)} 条三元组")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def sample_triplets(df, num_samples):
    """一次性随机采样多个三元组"""
    if len(df) < num_samples:
        print(f"数据量不足，只能采样 {len(df)} 个三元组")
        return df.sample(n=len(df)).reset_index(drop=True)
    else:
        return df.sample(n=num_samples).reset_index(drop=True)

def extract_answer(response, field_mapping):
    """从LLM回答中提取领域答案"""
    # 查找所有方括号对并提取内容
    bracket_contents = []
    
    # 分割字符串来查找方括号
    parts = response.split('[')
    
    for part in parts[1:]:  # 跳过第一个部分（在第一个'['之前的内容）
        if ']' in part:
            # 找到第一个']'的位置
            end_pos = part.find(']')
            content = part[:end_pos].strip()
            if content:  # 内容不为空
                bracket_contents.append(content)
    
    # 检查方括号内容是否匹配已知领域
    found_fields = []
    for content in bracket_contents:
        if content in field_mapping:
            found_fields.append(content)
            print(f"匹配成功: '{content}'")
        else:
            print(f"未匹配: '{content}' 不在已知领域中")
    
    if found_fields:
        print("最终匹配的领域为:", found_fields)
        return found_fields
    else:
        print("未找到有效领域")
        return []

def save_to_field_file(triplet, fields, field_mapping, base_path):
    """将三元组保存到对应领域的CSV文件中（线程安全版本）"""
    for field in fields:
        filename = field_mapping[field]
        file_path = os.path.join(base_path, filename)
        
        # 使用文件锁确保并发安全
        with file_locks[filename]:
            # 检查文件是否存在，如果不存在则创建
            if not os.path.exists(file_path):
                # 创建新文件并写入表头
                df_new = pd.DataFrame(columns=['Head', 'Relation', 'Tail'])
                df_new.to_csv(file_path, index=False)
                print(f"创建新文件: {file_path}")
            
            # 读取现有数据
            try:
                df_existing = pd.read_csv(file_path)
            except:
                df_existing = pd.DataFrame(columns=['Head', 'Relation', 'Tail'])
            
            # 添加新的三元组
            new_row = pd.DataFrame([triplet])
            df_updated = pd.concat([df_existing, new_row], ignore_index=True)
            
            # 保存文件
            df_updated.to_csv(file_path, index=False)
            print(f"三元组已保存到 {filename}")

async def process_single_triplet_main_domain(triplet, output_dir, triplet_index, semaphore):
    """处理单个三元组 - 主领域分类"""
    async with semaphore:
        print(f"处理三元组 {triplet_index}: {triplet['Head']} -> {triplet['Relation']} -> {triplet['Tail']}")
        
        # 构造输入文本
        text = f"{triplet['Head']} -> {triplet['Relation']} -> {triplet['Tail']}"
        question = JUDGE_TEMPLATE.format(text=text)
        
        # 调用LLM
        try:
            llm = llm_client()
            response = await llm.response(question)
            print(f"LLM回答: {response}")
            
            # 提取答案
            fields = extract_answer(response, FIELD_TO_FILENAME)
            if not fields:
                fields = ['Other']
            print(f"提取的领域: {fields}")
            
            # 保存到对应文件
            triplet_dict = {
                'Head': triplet['Head'],
                'Relation': triplet['Relation'], 
                'Tail': triplet['Tail']
            }
            save_to_field_file(triplet_dict, fields, FIELD_TO_FILENAME, output_dir)
            
            print(f"三元组 {triplet_index} 处理完成！")
            
        except Exception as e:
            print(f"处理三元组 {triplet_index} 时出错: {e}")

async def process_single_triplet_subdomain(triplet, output_dir, triplet_index, semaphore, domain_name):
    """处理单个三元组 - 子领域分类"""
    async with semaphore:
        print(f"处理三元组 {triplet_index} ({domain_name}子领域): {triplet['Head']} -> {triplet['Relation']} -> {triplet['Tail']}")
        
        # 获取子领域配置
        subdomain_config = SUBDOMAIN_CONFIG[domain_name]
        template = subdomain_config['template']
        field_mapping = subdomain_config['field_mapping']
        
        # 构造输入文本
        text = f"{triplet['Head']} -> {triplet['Relation']} -> {triplet['Tail']}"
        question = template.format(text=text)
        
        # 调用LLM
        try:
            llm = llm_client()
            response = await llm.response(question)
            print(f"LLM回答: {response}")
            
            # 提取答案
            fields = extract_answer(response, field_mapping)
            if not fields:
                print(f"未找到有效子领域，跳过该三元组")
                return
            print(f"提取的子领域: {fields}")
            
            # 保存到对应文件
            triplet_dict = {
                'Head': triplet['Head'],
                'Relation': triplet['Relation'], 
                'Tail': triplet['Tail']
            }
            save_to_field_file(triplet_dict, fields, field_mapping, output_dir)
            
            print(f"三元组 {triplet_index} 处理完成！")
            
        except Exception as e:
            print(f"处理三元组 {triplet_index} 时出错: {e}")

async def batch_process_main_domain(num_samples=10, max_concurrent=10):
    """批量处理 - 主领域分类"""
    print(f"开始批量处理 {num_samples} 个三元组（主领域分类），最大并发数: {max_concurrent}...")
    
    # 文件路径
    input_file = "/home/lsz/OneGraph/data/onegraphv2-.csv"
    output_dir = "/home/lsz/OneGraph/data"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 一次性加载数据
    df = load_data(input_file)
    if df is None or len(df) == 0:
        print("数据加载失败或为空")
        return
    
    # 一次性随机采样多个三元组
    sampled_triplets = sample_triplets(df, num_samples)
    print(f"成功采样 {len(sampled_triplets)} 个三元组")
    
    # 创建信号量来控制并发数量
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 创建所有任务
    tasks = []
    for i, (_, triplet) in enumerate(sampled_triplets.iterrows(), 1):
        task = process_single_triplet_main_domain(triplet, output_dir, i, semaphore)
        tasks.append(task)
    
    # 记录开始时间
    start_time = time.time()
    
    # 并发执行所有任务
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n主领域分类处理完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个三元组耗时: {total_time/len(sampled_triplets):.2f} 秒")

async def batch_process_subdomain(domain_name, num_samples=10, max_concurrent=10):
    """批量处理 - 子领域分类"""
    if domain_name not in SUBDOMAIN_CONFIG:
        print(f"错误：不支持的领域 '{domain_name}'")
        print(f"支持的领域: {list(SUBDOMAIN_CONFIG.keys())}")
        return
    
    print(f"开始批量处理 {num_samples} 个三元组（{domain_name}子领域分类），最大并发数: {max_concurrent}...")
    
    # 文件路径
    main_domain_file = f"/home/lsz/OneGraph/data/{FIELD_TO_FILENAME[domain_name]}"
    output_dir = "/home/lsz/OneGraph/data/humanities"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载主领域数据
    df = load_data(main_domain_file)
    if df is None or len(df) == 0:
        print(f"主领域数据加载失败或为空: {main_domain_file}")
        return
    
    # 一次性随机采样多个三元组
    sampled_triplets = sample_triplets(df, num_samples)
    print(f"成功采样 {len(sampled_triplets)} 个三元组")
    
    # 创建信号量来控制并发数量
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 创建所有任务
    tasks = []
    for i, (_, triplet) in enumerate(sampled_triplets.iterrows(), 1):
        task = process_single_triplet_subdomain(triplet, output_dir, i, semaphore, domain_name)
        tasks.append(task)
    
    # 记录开始时间
    start_time = time.time()
    
    # 并发执行所有任务
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{domain_name}子领域分类处理完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个三元组耗时: {total_time/len(sampled_triplets):.2f} 秒")

def main():
    parser = argparse.ArgumentParser(description='知识图谱三元组领域分类工具')
    parser.add_argument('--mode', choices=['main', 'subdomain'], required=True,
                       help='分类模式: main=主领域分类, subdomain=子领域分类')
    parser.add_argument('--domain', type=str,
                       help='子领域分类时指定的主领域名称')
    parser.add_argument('--samples', type=int, default=10,
                       help='处理的三元组数量 (默认: 10)')
    parser.add_argument('--concurrent', type=int, default=10,
                       help='最大并发数 (默认: 10)')
    
    args = parser.parse_args()
    
    if args.mode == 'main':
        # 主领域分类
        asyncio.run(batch_process_main_domain(
            num_samples=args.samples, 
            max_concurrent=args.concurrent
        ))
    elif args.mode == 'subdomain':
        # 子领域分类
        if not args.domain:
            print("错误：子领域分类模式需要指定 --domain 参数")
            print(f"支持的领域: {list(SUBDOMAIN_CONFIG.keys())}")
            return
        
        asyncio.run(batch_process_subdomain(
            domain_name=args.domain,
            num_samples=args.samples,
            max_concurrent=args.concurrent
        ))

if __name__ == "__main__":
    main()