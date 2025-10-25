import pandas as pd
import networkx as nx
import numpy as np
import pickle
import json
import os
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class TripleSorter:
    def __init__(self, prototype_dir="/home/lsz/OneGraph/data", graph_dir="/home/lsz/OneGraph/isa_graphs", 
                 output_dir="/home/lsz/OneGraph/sorted_data", model_name='/disk0/lsz/PLMs/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化三元组分类器
        
        Args:
            prototype_dir: 原型文件目录
            graph_dir: 连通图文件目录
            output_dir: 输出目录
            model_name: 用于embedding的模型名称
        """
        self.prototype_dir = prototype_dir
        self.graph_dir = graph_dir
        self.output_dir = output_dir
        
        # 领域映射到文件名
        self.field_to_filename = {
            'Natural Sciences': 'natural_sciences.csv',
            'Engineering & Technology': 'engineering_technology.csv',
            'Medicine & Health': 'medicine_health.csv',
            'Agriculture': 'agriculture.csv',
            'Social Sciences': 'social_sciences.csv',
            'Humanities': 'humanities.csv',
            'Other': 'other.csv'
        }
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        for field in self.field_to_filename.keys():
            os.makedirs(os.path.join(output_dir, field.replace(' & ', '_').replace(' ', '_').lower()), exist_ok=True)
        
        # 加载sentence transformer模型
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(model_name)
        
        # 加载原型和计算原型embeddings
        self.prototypes = {}
        self.prototype_embeddings = {}
        self.load_prototypes()
        
        # 加载图索引
        self.graph_loader = GraphLoader(graph_dir)
        
    def load_prototypes(self):
        """加载各领域的原型三元组"""
        print("Loading prototypes...")
        
        for field, filename in self.field_to_filename.items():
            filepath = os.path.join(self.prototype_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                triples = []
                for _, row in df.iterrows():
                    triple_text = self.textualize_triple(row['Head'], row['Relation'], row['Tail'])
                    triples.append(triple_text)
                
                self.prototypes[field] = triples
                
                # 计算原型embeddings
                if triples:
                    embeddings = self.model.encode(triples)
                    # 使用平均embedding作为该领域的prototype
                    self.prototype_embeddings[field] = np.mean(embeddings, axis=0)
                    print(f"Loaded {len(triples)} prototypes for {field}")
                else:
                    print(f"Warning: No prototypes found for {field}")
            else:
                print(f"Warning: Prototype file not found for {field}: {filepath}")
    
    def textualize_triple(self, head, relation, tail):
        """将三元组文本化为 Head -> Relation -> Tail 格式"""
        return f"{head} -> {relation} -> {tail}"
    
    def classify_batch_triples(self, triple_texts, threshold=0.0):
        """
        批量对三元组进行分类
        
        Args:
            triple_texts: 三元组文本列表
            threshold: 相似度阈值，超过此值才认为属于该领域
            
        Returns:
            list: 每个三元组对应的领域
        """
        if not self.prototype_embeddings or not triple_texts:
            return ['Other'] * len(triple_texts)
        
        # 批量计算三元组的embeddings
        triple_embeddings = self.model.encode(triple_texts)
        
        results = []
        for triple_embedding in triple_embeddings:
            # 计算与各领域原型的相似度
            similarities = {}
            for field, prototype_embedding in self.prototype_embeddings.items():
                similarity = cosine_similarity(
                    triple_embedding.reshape(1, -1), 
                    prototype_embedding.reshape(1, -1)
                )[0][0]
                similarities[field] = similarity
            
            # 找到相似度最高的领域
            max_similarity = max(similarities.values())
            best_field = max(similarities, key=similarities.get)
            
            # 如果最高相似度超过阈值，返回该领域；否则返回Other
            if max_similarity >= threshold:
                results.append(best_field)
            else:
                results.append('Other')
        
        return results
    
    def process_component_group(self, components_data, group_name):
        """批量处理一组连通分量"""
        print(f"Processing {group_name} components (batch size: {len(components_data)})...")
        
        # 收集所有代表性三元组
        representative_triples = []
        component_info = []
        
        for comp_id, component_data in components_data.items():
            # 获取边列表
            if isinstance(component_data, dict) and 'edges' in component_data:
                edges = component_data['edges']
            else:
                # 如果是NetworkX图对象
                edges = [(u, v, d.get('relation', '')) for u, v, d in component_data.edges(data=True)]
            
            if not edges:
                continue
            
            # 取第一个三元组作为代表
            first_edge = edges[0]
            head, tail, relation = first_edge
            representative_triple = self.textualize_triple(head, relation, tail)
            
            representative_triples.append(representative_triple)
            component_info.append({
                'comp_id': comp_id,
                'edges': edges,
                'representative': representative_triple
            })
        
        if not representative_triples:
            return {}
        
        # 批量分类代表性三元组
        print(f"Classifying {len(representative_triples)} representative triples...")
        assigned_fields = self.classify_batch_triples(representative_triples)
        
        # 组织结果
        batch_results = defaultdict(list)
        
        for i, comp_info in enumerate(component_info):
            assigned_field = assigned_fields[i]
            comp_id = comp_info['comp_id']
            edges = comp_info['edges']
            
            print(f"Component {comp_id}: {comp_info['representative']} -> {assigned_field}")
            
            # 将该连通分量的所有三元组分配到对应领域
            for head, tail, relation in edges:
                batch_results[assigned_field].append({
                    'Head': head,
                    'Relation': relation,
                    'Tail': tail,
                    'Component_ID': comp_id,
                    'Type': 'ISA_Component'
                })
        
        return batch_results
    
    def save_batch_results(self, batch_results, batch_name):
        """保存批次结果"""
        for field, triples in batch_results.items():
            if not triples:
                continue
                
            # 创建DataFrame
            df = pd.DataFrame(triples)
            
            # 确定输出文件路径
            field_dir = field.replace(' & ', '_').replace(' ', '_').lower()
            output_path = os.path.join(self.output_dir, field_dir, f"isa_{batch_name}.csv")
            
            # 保存到CSV
            df.to_csv(output_path, index=False)
            
            print(f"Saved {len(triples)} triples to {field} ({batch_name})")
    
    def process_all_components_batch(self):
        """批量处理所有ISA连通分量"""
        if not self.graph_loader.index:
            print("Error: Graph index not found!")
            return {}
        
        print("Processing ISA components in batches...")
        
        # 统计信息
        total_isa_stats = defaultdict(int)
        
        # 处理大型连通分量（单独存储的）
        large_dir = os.path.join(self.graph_dir, "large_components")
        if os.path.exists(large_dir):
            print("Processing large components...")
            large_files = [f for f in os.listdir(large_dir) if f.endswith('.pkl')]
            
            for filename in tqdm(large_files, desc="Large components"):
                filepath = os.path.join(large_dir, filename)
                comp_id = filename.replace('component_', '').replace('.pkl', '')
                
                try:
                    with open(filepath, 'rb') as f:
                        component_data = pickle.load(f)
                    
                    # 单独处理大型组件
                    batch_results = self.process_component_group({comp_id: component_data}, f"large_{comp_id}")
                    self.save_batch_results(batch_results, f"large_{comp_id}")
                    
                    # 更新统计
                    for field, triples in batch_results.items():
                        total_isa_stats[field] += len(triples)
                        
                except Exception as e:
                    print(f"Error processing large component {comp_id}: {e}")
                    continue
        
        # 处理批量存储的连通分量
        batch_files = ['tiny_components.pkl', 'small_components.pkl', 'medium_components.pkl']
        
        for batch_file in batch_files:
            filepath = os.path.join(self.graph_dir, batch_file)
            if not os.path.exists(filepath):
                continue
            
            print(f"Processing {batch_file}...")
            
            try:
                with open(filepath, 'rb') as f:
                    components_data = pickle.load(f)
                
                print(f"Loaded {len(components_data)} components from {batch_file}")
                
                # 批量处理这些组件
                batch_results = self.process_component_group(components_data, batch_file.replace('.pkl', ''))
                self.save_batch_results(batch_results, batch_file.replace('.pkl', ''))
                
                # 更新统计
                for field, triples in batch_results.items():
                    total_isa_stats[field] += len(triples)
                    
            except Exception as e:
                print(f"Error processing {batch_file}: {e}")
                continue
        
        return dict(total_isa_stats)
    
    def process_non_isa_triples(self):
        """处理非ISA关系的三元组"""
        print("Processing non-ISA triples...")
        
        # 查找所有非ISA关系文件
        non_isa_files = []
        for filename in os.listdir(self.graph_dir):
            if filename.startswith('non_isa_relations_batch_') and filename.endswith('.pkl'):
                non_isa_files.append(filename)
        
        if not non_isa_files:
            print("No non-ISA relation files found")
            return {}
        
        print(f"Found {len(non_isa_files)} non-ISA relation files")
        
        # 统计信息
        non_isa_stats = defaultdict(int)
        batch_size = 1000000  # 批量处理大小
        
        for filename in tqdm(non_isa_files, desc="Processing non-ISA files"):
            filepath = os.path.join(self.graph_dir, filename)
            
            try:
                with open(filepath, 'rb') as f:
                    triples = pickle.load(f)
                
                print(f"Processing {len(triples)} triples from {filename}")
                
                # 分批处理三元组
                for i in range(0, len(triples), batch_size):
                    batch = triples[i:i+batch_size]
                    
                    print(f"Processing batch {i//batch_size + 1}/{(len(triples)-1)//batch_size + 1}")
                    
                    # 准备批量数据
                    triple_texts = []
                    triple_data = []
                    
                    for head, relation, tail in batch:
                        triple_text = self.textualize_triple(head, relation, tail)
                        triple_texts.append(triple_text)
                        triple_data.append((head, relation, tail))
                    
                    # 批量分类
                    assigned_fields = self.classify_batch_triples(triple_texts)
                    
                    # 组织结果
                    batch_results = defaultdict(list)
                    for j, (head, relation, tail) in enumerate(triple_data):
                        assigned_field = assigned_fields[j]
                        
                        batch_results[assigned_field].append({
                            'Head': head,
                            'Relation': relation,
                            'Tail': tail,
                            'Component_ID': f"non_isa_{filename}_{i//batch_size}",
                            'Type': 'Non_ISA'
                        })
                        
                        non_isa_stats[assigned_field] += 1
                    
                    # 保存批次结果
                    self.save_non_isa_batch(batch_results, filename, i//batch_size)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        return dict(non_isa_stats)
    
    def save_non_isa_batch(self, batch_results, source_filename, batch_id):
        """保存非ISA三元组的批次结果"""
        for field, triples in batch_results.items():
            if not triples:
                continue
            
            # 创建DataFrame
            df = pd.DataFrame(triples)
            
            # 确定输出文件路径
            field_dir = field.replace(' & ', '_').replace(' ', '_').lower()
            output_path = os.path.join(self.output_dir, field_dir, f"non_isa_{source_filename}_{batch_id}.csv")
            
            # 保存到CSV
            df.to_csv(output_path, index=False)
    
    def process_all_data(self):
        """处理所有数据：ISA连通分量 + 非ISA三元组"""
        print("Starting comprehensive data processing...")
        
        # 处理ISA连通分量（批量）
        print("\n=== Processing ISA Components (Batch Mode) ===")
        isa_stats = self.process_all_components_batch()
        
        # 处理非ISA三元组
        print("\n=== Processing Non-ISA Triples ===")
        non_isa_stats = self.process_non_isa_triples()
        
        # 合并统计信息
        total_stats = defaultdict(int)
        for field, count in isa_stats.items():
            total_stats[field] += count
        for field, count in non_isa_stats.items():
            total_stats[field] += count
        
        # 输出统计信息
        print("\n=== Final Classification Statistics ===")
        print("ISA Components:")
        for field, count in isa_stats.items():
            print(f"  {field}: {count} triples")
        
        print("\nNon-ISA Triples:")
        for field, count in non_isa_stats.items():
            print(f"  {field}: {count} triples")
        
        print("\nTotal:")
        for field, count in total_stats.items():
            print(f"  {field}: {count} triples")
        
        # 保存统计信息
        stats_path = os.path.join(self.output_dir, "classification_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'isa_stats': isa_stats,
                'non_isa_stats': non_isa_stats,
                'total_stats': dict(total_stats)
            }, f, ensure_ascii=False, indent=2)
        
        return dict(total_stats)
    
    def merge_field_files(self):
        """合并每个领域的所有文件"""
        print("Merging field files...")
        
        for field in self.field_to_filename.keys():
            field_dir = field.replace(' & ', '_').replace(' ', '_').lower()
            field_path = os.path.join(self.output_dir, field_dir)
            
            if not os.path.exists(field_path):
                continue
            
            # 获取该领域的所有CSV文件
            csv_files = [f for f in os.listdir(field_path) if f.endswith('.csv')]
            
            if not csv_files:
                continue
            
            # 合并所有文件
            merged_data = []
            for csv_file in tqdm(csv_files, desc=f"Merging {field}"):
                file_path = os.path.join(field_path, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    merged_data.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            
            if merged_data:
                merged_df = pd.concat(merged_data, ignore_index=True)
                
                # 保存合并后的文件
                merged_path = os.path.join(field_path, f"{field_dir}_all.csv")
                merged_df.to_csv(merged_path, index=False)
                
                print(f"Merged {len(merged_df)} triples for {field}")
                
                # 统计ISA和非ISA的数量
                if 'Type' in merged_df.columns:
                    isa_count = len(merged_df[merged_df['Type'] == 'ISA_Component'])
                    non_isa_count = len(merged_df[merged_df['Type'] == 'Non_ISA'])
                    print(f"  - ISA Components: {isa_count}")
                    print(f"  - Non-ISA Triples: {non_isa_count}")


class GraphLoader:
    """从之前的脚本中复制的GraphLoader类"""
    def __init__(self, base_dir="isa_graphs"):
        self.base_dir = base_dir
        index_path = os.path.join(base_dir, 'index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = None
            print("Warning: Index file not found")


def main():
    """主函数"""
    print("Starting comprehensive triple classification...")
    
    # 创建分类器
    sorter = TripleSorter(
        prototype_dir="/home/lsz/OneGraph/data",
        graph_dir="/home/lsz/OneGraph/isa_graphs",
        output_dir="/home/lsz/OneGraph/sorted_data",
        model_name="/disk0/lsz/PLMs/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 处理所有数据（ISA连通分量 + 非ISA三元组）
    sorter.process_all_data()
    
    # 合并每个领域的文件
    sorter.merge_field_files()
    
    print("Classification completed!")


if __name__ == "__main__":
    main()