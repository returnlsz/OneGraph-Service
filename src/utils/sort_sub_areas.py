import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
warnings.filterwarnings('ignore')
class SubdomainClassifier:
    def __init__(self, 
                 sorted_data_dir="/home/lsz/OneGraph/sorted_data_v1",
                 prototype_dir="/home/lsz/OneGraph/data", 
                 output_dir="/home/lsz/OneGraph/sorted_data_v2",
                 model_name='/disk0/lsz/PLMs/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 max_workers=4):
        """
        初始化子领域分类器
        
        Args:
            sorted_data_dir: 已分类数据目录
            prototype_dir: 子领域原型文件目录
            output_dir: 输出目录
            model_name: 用于embedding的模型名称
            max_workers: 最大线程数
        """
        self.sorted_data_dir = sorted_data_dir
        self.prototype_dir = prototype_dir
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.model_name = model_name  # 保存模型名称
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载sentence transformer模型
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(model_name)
        
        # 为多线程创建模型副本的锁
        self.model_lock = threading.Lock()
        self.models = {}  # 为每个线程创建模型副本
        
        # 发现可用的领域和子领域
        self.discover_domains_and_subdomains()
        
        # 加载子领域原型
        self.load_subdomain_prototypes()
    
    def get_model_for_thread(self):
        """为当前线程获取模型实例"""
        thread_id = threading.current_thread().ident
        
        if thread_id not in self.models:
            with self.model_lock:
                if thread_id not in self.models:
                    # 为每个线程创建独立的模型实例
                    print(f"Creating model instance for thread {thread_id}")
                    self.models[thread_id] = SentenceTransformer(self.model_name)
        
        return self.models[thread_id]
    
    def discover_domains_and_subdomains(self):
        """发现可用的领域和对应的子领域"""
        self.domain_subdomain_mapping = {}
        
        # 扫描sorted_data_v1目录下的领域
        if not os.path.exists(self.sorted_data_dir):
            print(f"Error: Sorted data directory not found: {self.sorted_data_dir}")
            return
        
        domains = [d for d in os.listdir(self.sorted_data_dir) 
                  if os.path.isdir(os.path.join(self.sorted_data_dir, d))]
        
        print(f"Found domains: {domains}")
        
        # 对每个领域，检查是否有对应的子领域原型
        for domain in domains:
            prototype_domain_dir = os.path.join(self.prototype_dir, domain)
            if os.path.exists(prototype_domain_dir):
                # 获取该领域的所有子领域原型文件
                subdomain_files = [f for f in os.listdir(prototype_domain_dir) 
                                 if f.endswith('.csv')]
                
                if subdomain_files:
                    subdomains = [f.replace('.csv', '') for f in subdomain_files]
                    self.domain_subdomain_mapping[domain] = subdomains
                    print(f"Domain '{domain}' has subdomains: {subdomains}")
                    
                    # 创建输出子目录
                    for subdomain in subdomains:
                        subdomain_output_dir = os.path.join(self.output_dir, domain, subdomain)
                        os.makedirs(subdomain_output_dir, exist_ok=True)
                    
                    # 创建other目录
                    other_output_dir = os.path.join(self.output_dir, domain, 'other')
                    os.makedirs(other_output_dir, exist_ok=True)
                else:
                    print(f"No subdomain prototypes found for domain: {domain}")
            else:
                print(f"No prototype directory found for domain: {domain}")
    
    def load_subdomain_prototypes(self):
        """加载各子领域的原型三元组"""
        print("Loading subdomain prototypes...")
        
        self.subdomain_prototypes = {}
        self.subdomain_embeddings = {}
        
        for domain, subdomains in self.domain_subdomain_mapping.items():
            self.subdomain_prototypes[domain] = {}
            self.subdomain_embeddings[domain] = {}
            
            for subdomain in subdomains:
                prototype_file = os.path.join(self.prototype_dir, domain, f"{subdomain}.csv")
                
                if os.path.exists(prototype_file):
                    try:
                        df = pd.read_csv(prototype_file)
                        triples = []
                        
                        for _, row in df.iterrows():
                            # 确保有必要的列
                            if 'Head' in row and 'Relation' in row and 'Tail' in row:
                                triple_text = self.textualize_triple(row['Head'], row['Relation'], row['Tail'])
                                triples.append(triple_text)
                        
                        if triples:
                            self.subdomain_prototypes[domain][subdomain] = triples
                            
                            # 计算原型embeddings
                            embeddings = self.model.encode(triples)
                            # 使用平均embedding作为该子领域的prototype
                            self.subdomain_embeddings[domain][subdomain] = np.mean(embeddings, axis=0)
                            print(f"Loaded {len(triples)} prototypes for {domain}/{subdomain}")
                        else:
                            print(f"Warning: No valid triples found in {prototype_file}")
                            
                    except Exception as e:
                        print(f"Error loading prototype file {prototype_file}: {e}")
                else:
                    print(f"Warning: Prototype file not found: {prototype_file}")
    
    def textualize_triple(self, head, relation, tail):
        """将三元组文本化为 Head -> Relation -> Tail 格式"""
        return f"{head} -> {relation} -> {tail}"
    
    def classify_triples_to_subdomains(self, domain, triple_texts, threshold=0.0):
        """
        将三元组分类到子领域
        
        Args:
            domain: 当前处理的领域
            triple_texts: 三元组文本列表
            threshold: 相似度阈值
            
        Returns:
            list: 每个三元组对应的子领域
        """
        if domain not in self.subdomain_embeddings or not triple_texts:
            return ['other'] * len(triple_texts)
        
        subdomain_embeddings = self.subdomain_embeddings[domain]
        if not subdomain_embeddings:
            return ['other'] * len(triple_texts)
        
        # 获取当前线程的模型实例
        thread_model = self.get_model_for_thread()
        
        # 批量计算三元组的embeddings
        triple_embeddings = thread_model.encode(triple_texts)
        
        results = []
        for triple_embedding in triple_embeddings:
            # 计算与各子领域原型的相似度
            similarities = {}
            for subdomain, prototype_embedding in subdomain_embeddings.items():
                similarity = cosine_similarity(
                    triple_embedding.reshape(1, -1), 
                    prototype_embedding.reshape(1, -1)
                )[0][0]
                similarities[subdomain] = similarity
            
            # 找到相似度最高的子领域
            if similarities:
                max_similarity = max(similarities.values())
                best_subdomain = max(similarities, key=similarities.get)
                
                # 如果最高相似度超过阈值，返回该子领域；否则返回other
                if max_similarity >= threshold:
                    results.append(best_subdomain)
                else:
                    results.append('other')
            else:
                results.append('other')
        
        return results
    
    def process_single_file(self, domain, csv_file, file_index, total_files):
        """处理单个CSV文件"""
        file_path = os.path.join(self.sorted_data_dir, domain, csv_file)
        thread_id = threading.current_thread().ident
        
        try:
            print(f"[Thread {thread_id}] Processing file {file_index+1}/{total_files}: {csv_file}")
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            if df.empty:
                return defaultdict(int)
            
            # 确保有必要的列
            if not all(col in df.columns for col in ['Head', 'Relation', 'Tail']):
                print(f"[Thread {thread_id}] Warning: Missing required columns in {csv_file}")
                return defaultdict(int)
            
            print(f"[Thread {thread_id}] Processing {len(df)} triples from {csv_file}")
            
            # 统计信息
            file_subdomain_stats = defaultdict(int)
            batch_size = 1000  # 进一步减小批处理大小
            
            # 分批处理
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size].copy()
                
                # 准备三元组文本
                triple_texts = []
                for _, row in batch_df.iterrows():
                    triple_text = self.textualize_triple(row['Head'], row['Relation'], row['Tail'])
                    triple_texts.append(triple_text)
                
                # 分类到子领域
                assigned_subdomains = self.classify_triples_to_subdomains(domain, triple_texts)
                
                # 添加子领域标签
                batch_df['Subdomain'] = assigned_subdomains
                
                # 按子领域分组保存
                subdomain_groups = batch_df.groupby('Subdomain')
                
                for subdomain, group_df in subdomain_groups:
                    # 保存到对应的子领域目录
                    if subdomain == 'other':
                        output_dir = os.path.join(self.output_dir, domain, 'other')
                    else:
                        output_dir = os.path.join(self.output_dir, domain, subdomain)
                    
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 生成输出文件名
                    base_filename = csv_file.replace('.csv', '')
                    output_filename = f"{base_filename}_batch_{i//batch_size}_thread_{thread_id}.csv"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 保存
                    group_df.to_csv(output_path, index=False)
                    
                    # 更新统计
                    file_subdomain_stats[subdomain] += len(group_df)
                
                # 打印进度
                if (i // batch_size + 1) % 10 == 0:
                    print(f"[Thread {thread_id}] Processed {i + len(batch_df)}/{len(df)} triples in {csv_file}")
            
            print(f"[Thread {thread_id}] Completed {csv_file}: {dict(file_subdomain_stats)}")
            return file_subdomain_stats
            
        except Exception as e:
            print(f"[Thread {thread_id}] Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            return defaultdict(int)
    
    def process_domain_files(self, domain):
        """使用多线程处理某个领域的所有文件"""
        domain_dir = os.path.join(self.sorted_data_dir, domain)
        
        if not os.path.exists(domain_dir):
            print(f"Domain directory not found: {domain_dir}")
            return {}
        
        # 获取该领域的所有CSV文件
        csv_files = [f for f in os.listdir(domain_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found in {domain_dir}")
            return {}
        
        print(f"Processing {len(csv_files)} files for domain: {domain} using {self.max_workers} threads")
        
        # 统计信息
        domain_subdomain_stats = defaultdict(int)
        
        # 使用线程池处理文件
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.process_single_file, domain, csv_file, i, len(csv_files)): csv_file
                for i, csv_file in enumerate(csv_files)
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_file):
                csv_file = future_to_file[future]
                completed += 1
                try:
                    file_stats = future.result()
                    # 合并统计信息
                    for subdomain, count in file_stats.items():
                        domain_subdomain_stats[subdomain] += count
                    
                    print(f"Completed {completed}/{len(csv_files)} files for domain {domain}")
                        
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
        
        # 输出该领域的统计信息
        print(f"\nSubdomain classification results for {domain}:")
        for subdomain, count in domain_subdomain_stats.items():
            print(f"  {subdomain}: {count} triples")
        
        return dict(domain_subdomain_stats)
    
    def merge_subdomain_files(self, domain):
        """合并某个领域下各子领域的文件"""
        print(f"Merging files for domain: {domain}")
        
        domain_output_dir = os.path.join(self.output_dir, domain)
        if not os.path.exists(domain_output_dir):
            return
        
        # 获取所有子领域目录
        subdomain_dirs = [d for d in os.listdir(domain_output_dir) 
                         if os.path.isdir(os.path.join(domain_output_dir, d))]
        
        for subdomain in subdomain_dirs:
            subdomain_dir = os.path.join(domain_output_dir, subdomain)
            csv_files = [f for f in os.listdir(subdomain_dir) if f.endswith('.csv') and not f.endswith('_all.csv')]
            
            if not csv_files:
                continue
            
            print(f"  Merging {len(csv_files)} files for {subdomain}...")
            
            # 合并所有文件
            merged_data = []
            for csv_file in tqdm(csv_files, desc=f"Merging {subdomain}", leave=False):
                file_path = os.path.join(subdomain_dir, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    merged_data.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            
            if merged_data:
                merged_df = pd.concat(merged_data, ignore_index=True)
                
                # 保存合并后的文件
                merged_path = os.path.join(subdomain_dir, f"{subdomain}_all.csv")
                merged_df.to_csv(merged_path, index=False)
                
                print(f"  {subdomain}: merged {len(merged_df)} triples")
    
    def process_single_domain(self, domain):
        """处理单个领域"""
        print(f"\n=== Processing Domain: {domain} ===")
        start_time = time.time()
        
        # 处理该领域的文件
        domain_stats = self.process_domain_files(domain)
        
        # 合并该领域的子领域文件
        self.merge_subdomain_files(domain)
        
        end_time = time.time()
        print(f"Domain {domain} completed in {end_time - start_time:.2f} seconds")
        
        return domain, domain_stats
    
    def process_all_domains(self):
        """串行处理所有领域，但每个领域内部文件并行处理"""
        print("Starting subdomain classification for all domains...")
        
        domains = list(self.domain_subdomain_mapping.keys())
        total_stats = {}
        
        # 串行处理领域，避免资源竞争
        for domain in domains:
            try:
                domain_name, domain_stats = self.process_single_domain(domain)
                total_stats[domain_name] = domain_stats
            except Exception as e:
                print(f"Error processing domain {domain}: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存总体统计信息
        stats_path = os.path.join(self.output_dir, "subdomain_classification_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(total_stats, f, ensure_ascii=False, indent=2)
        
        # 输出总体统计
        print("\n=== Final Subdomain Classification Statistics ===")
        for domain, domain_stats in total_stats.items():
            print(f"\n{domain}:")
            for subdomain, count in domain_stats.items():
                print(f"  {subdomain}: {count} triples")
        
        return total_stats
    
    def create_summary_report(self):
        """创建分类总结报告"""
        print("Creating summary report...")
        
        report = {}
        
        for domain in self.domain_subdomain_mapping.keys():
            domain_output_dir = os.path.join(self.output_dir, domain)
            if not os.path.exists(domain_output_dir):
                continue
            
            domain_report = {}
            subdomain_dirs = [d for d in os.listdir(domain_output_dir) 
                             if os.path.isdir(os.path.join(domain_output_dir, d))]
            
            for subdomain in subdomain_dirs:
                subdomain_dir = os.path.join(domain_output_dir, subdomain)
                merged_file = os.path.join(subdomain_dir, f"{subdomain}_all.csv")
                
                if os.path.exists(merged_file):
                    try:
                        df = pd.read_csv(merged_file)
                        domain_report[subdomain] = len(df)
                    except:
                        domain_report[subdomain] = 0
                else:
                    domain_report[subdomain] = 0
            
            report[domain] = domain_report
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "classification_summary.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("Summary report saved to:", report_path)
        return report
def main():
    """主函数"""
    print("Starting subdomain classification...")
    
    # 创建子领域分类器
    classifier = SubdomainClassifier(
        sorted_data_dir="/home/lsz/OneGraph/sorted_data_v1",
        prototype_dir="/home/lsz/OneGraph/data",
        output_dir="/home/lsz/OneGraph/sorted_data_v2",
        model_name="/disk0/lsz/PLMs/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_workers=20  # 可以根据您的机器配置调整线程数
    )
    
    # 处理所有领域
    start_time = time.time()
    classifier.process_all_domains()
    end_time = time.time()
    
    print(f"\nAll domains processed in {end_time - start_time:.2f} seconds")
    
    # 创建总结报告
    classifier.create_summary_report()
    
    print("Subdomain classification completed!")
if __name__ == "__main__":
    main()
