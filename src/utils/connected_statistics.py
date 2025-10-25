import pandas as pd
import networkx as nx
import os
from collections import defaultdict
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

class DomainGraphAnalyzer:
    def __init__(self, base_dir, max_workers=4):
        self.base_dir = base_dir
        self.max_workers = max_workers
        self.results = {}
        self.lock = threading.Lock()  # 用于线程安全
        
    def get_domain_folders(self):
        """获取所有领域文件夹"""
        domain_folders = []
        if not os.path.exists(self.base_dir):
            print(f"Base directory {self.base_dir} does not exist!")
            return domain_folders
            
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                domain_folders.append((item, item_path))
        
        return domain_folders
    
    def load_single_csv_file(self, csv_file):
        """加载单个CSV文件的三元组 - 用于多线程"""
        triples = []
        try:
            # 尝试读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查是否有必要的列
            required_columns = ['Head', 'Relation', 'Tail']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: CSV file {csv_file} missing required columns")
                return triples
            
            # 提取前三列的三元组
            for _, row in df.iterrows():
                if pd.notna(row['Head']) and pd.notna(row['Tail']) and pd.notna(row['Relation']):
                    triples.append((str(row['Head']), str(row['Relation']), str(row['Tail'])))
                    
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
        
        return triples
    
    def load_csv_files_from_domain_parallel(self, domain_path):
        """使用多线程从领域文件夹中加载所有CSV文件的三元组"""
        csv_files = []
        
        # 找到所有CSV文件
        for file in os.listdir(domain_path):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(domain_path, file))
        
        print(f"Found {len(csv_files)} CSV files in domain")
        
        if not csv_files:
            return []
        
        all_triples = []
        
        # 使用线程池并行处理CSV文件
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(self.load_single_csv_file, csv_file): csv_file 
                             for csv_file in csv_files}
            
            # 使用tqdm显示进度
            with tqdm(total=len(csv_files), desc="Loading CSV files") as pbar:
                for future in as_completed(future_to_file):
                    csv_file = future_to_file[future]
                    try:
                        triples = future.result()
                        all_triples.extend(triples)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing {csv_file}: {e}")
                        pbar.update(1)
        
        return all_triples
    
    def load_csv_files_from_domain_chunked(self, domain_path, chunk_size=50000):
        """使用分块读取优化大文件处理"""
        csv_files = []
        
        # 找到所有CSV文件
        for file in os.listdir(domain_path):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(domain_path, file))
        
        print(f"Found {len(csv_files)} CSV files in domain")
        
        if not csv_files:
            return []
        
        def load_csv_chunked(csv_file):
            """分块读取单个CSV文件"""
            triples = []
            try:
                # 分块读取大文件
                for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                    # 检查是否有必要的列
                    required_columns = ['Head', 'Relation', 'Tail']
                    if not all(col in chunk.columns for col in required_columns):
                        continue
                    
                    # 过滤有效行并提取三元组
                    valid_rows = chunk.dropna(subset=required_columns)
                    chunk_triples = [(str(row['Head']), str(row['Relation']), str(row['Tail'])) 
                                   for _, row in valid_rows.iterrows()]
                    triples.extend(chunk_triples)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
            
            return triples
        
        all_triples = []
        
        # 使用线程池并行处理CSV文件（分块读取）
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(load_csv_chunked, csv_file): csv_file 
                             for csv_file in csv_files}
            
            with tqdm(total=len(csv_files), desc="Loading CSV files (chunked)") as pbar:
                for future in as_completed(future_to_file):
                    csv_file = future_to_file[future]
                    try:
                        triples = future.result()
                        all_triples.extend(triples)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing {csv_file}: {e}")
                        pbar.update(1)
        
        return all_triples
    
    def build_graph_from_triples(self, triples):
        """从三元组构建图"""
        graph = nx.Graph()  # 使用无向图来计算连通分量
        
        # 批量添加边以提高性能
        edges = [(head, tail) for head, relation, tail in triples]
        graph.add_edges_from(edges)
        
        return graph
    
    def calculate_connectivity_stats(self, graph, total_triples):
        """计算连通性统计信息"""
        if total_triples == 0:
            return 0, 0.0
        
        # 计算连通分量数量
        connected_components = list(nx.connected_components(graph))
        num_components = len(connected_components)
        
        # 计算连通率：(三元组总数 - 连通分量个数) / 三元组总数
        if total_triples > 0:
            connectivity_rate = (total_triples - num_components) / total_triples
        else:
            connectivity_rate = 0.0
        
        return num_components, connectivity_rate
    
    def analyze_domain(self, domain_name, domain_path, use_chunked=True):
        """分析单个领域"""
        print(f"\nAnalyzing domain: {domain_name}")
        
        # 选择加载方法
        if use_chunked:
            triples = self.load_csv_files_from_domain_chunked(domain_path)
        else:
            triples = self.load_csv_files_from_domain_parallel(domain_path)
        
        total_triples = len(triples)
        
        if total_triples == 0:
            print(f"No valid triples found in domain {domain_name}")
            return {
                'total_triples': 0,
                'connected_components': 0,
                'connectivity_rate': 0.0,
                'largest_component_size': 0,
                'component_size_distribution': {},
                'total_nodes': 0,
                'total_edges': 0
            }
        
        print(f"Total triples: {total_triples}")
        
        # 构建图
        print("Building graph...")
        graph = self.build_graph_from_triples(triples)
        
        # 计算连通性统计
        print("Calculating connectivity statistics...")
        num_components, connectivity_rate = self.calculate_connectivity_stats(graph, total_triples)
        
        # 获取连通分量的详细信息
        connected_components = list(nx.connected_components(graph))
        component_sizes = [len(comp) for comp in connected_components]
        
        # 统计连通分量大小分布
        size_distribution = defaultdict(int)
        for size in component_sizes:
            if size >= 1000:
                size_distribution['large (>=1000)'] += 1
            elif size >= 100:
                size_distribution['medium (100-999)'] += 1
            elif size >= 10:
                size_distribution['small (10-99)'] += 1
            else:
                size_distribution['tiny (<10)'] += 1
        
        largest_component_size = max(component_sizes) if component_sizes else 0
        
        result = {
            'total_triples': total_triples,
            'connected_components': num_components,
            'connectivity_rate': connectivity_rate,
            'largest_component_size': largest_component_size,
            'component_size_distribution': dict(size_distribution),
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges()
        }
        
        return result
    
    def analyze_single_domain_worker(self, domain_info):
        """单个领域分析的工作函数 - 用于多线程分析多个领域"""
        domain_name, domain_path = domain_info
        try:
            result = self.analyze_domain(domain_name, domain_path)
            
            # 线程安全地更新结果
            with self.lock:
                self.results[domain_name] = result
            
            # 打印当前领域的结果
            print(f"\nResults for {domain_name}:")
            print(f"  Total triples: {result['total_triples']:,}")
            print(f"  Connected components: {result['connected_components']:,}")
            print(f"  Connectivity rate: {result['connectivity_rate']:.4f} ({result['connectivity_rate']*100:.2f}%)")
            print(f"  Largest component size: {result['largest_component_size']:,}")
            print(f"  Total nodes: {result['total_nodes']:,}")
            print(f"  Total edges: {result['total_edges']:,}")
            print(f"  Component size distribution: {result['component_size_distribution']}")
            
            return True
        except Exception as e:
            print(f"Error analyzing domain {domain_name}: {e}")
            return False
    
    def analyze_all_domains(self, parallel_domains=False):
        """分析所有领域"""
        domain_folders = self.get_domain_folders()
        
        if not domain_folders:
            print("No domain folders found!")
            return
        
        print(f"Found {len(domain_folders)} domain folders")
        print(f"Using {self.max_workers} threads for CSV loading")
        
        if parallel_domains and len(domain_folders) > 1:
            # 并行分析多个领域（适用于领域数量多但每个领域文件不太大的情况）
            print("Analyzing domains in parallel...")
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(domain_folders))) as executor:
                futures = [executor.submit(self.analyze_single_domain_worker, domain_info) 
                          for domain_info in domain_folders]
                
                for future in as_completed(futures):
                    future.result()  # 等待完成
        else:
            # 串行分析领域，但每个领域内部并行处理CSV文件
            for domain_name, domain_path in domain_folders:
                try:
                    result = self.analyze_domain(domain_name, domain_path)
                    self.results[domain_name] = result
                    
                    # 打印当前领域的结果
                    print(f"\nResults for {domain_name}:")
                    print(f"  Total triples: {result['total_triples']:,}")
                    print(f"  Connected components: {result['connected_components']:,}")
                    print(f"  Connectivity rate: {result['connectivity_rate']:.4f} ({result['connectivity_rate']*100:.2f}%)")
                    print(f"  Largest component size: {result['largest_component_size']:,}")
                    print(f"  Total nodes: {result['total_nodes']:,}")
                    print(f"  Total edges: {result['total_edges']:,}")
                    print(f"  Component size distribution: {result['component_size_distribution']}")
                    
                except Exception as e:
                    print(f"Error analyzing domain {domain_name}: {e}")
                    continue
    
    def save_results(self, output_file="domain_analysis_results.json"):
        """保存结果到JSON文件"""
        # 添加总体统计
        summary = {
            'total_domains': len(self.results),
            'total_triples_all_domains': sum(r['total_triples'] for r in self.results.values()),
            'total_components_all_domains': sum(r['connected_components'] for r in self.results.values()),
            'average_connectivity_rate': sum(r['connectivity_rate'] for r in self.results.values()) / len(self.results) if self.results else 0
        }
        
        output_data = {
            'summary': summary,
            'domain_results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def print_summary(self):
        """打印总结"""
        if not self.results:
            print("No results to summarize!")
            return
        
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        # 按三元组数量排序
        sorted_domains = sorted(self.results.items(), key=lambda x: x[1]['total_triples'], reverse=True)
        
        print(f"\nDomains ranked by number of triples:")
        print("-" * 60)
        for i, (domain, result) in enumerate(sorted_domains, 1):
            print(f"{i:2d}. {domain:30s}: {result['total_triples']:8,} triples, "
                  f"{result['connected_components']:6,} components, "
                  f"connectivity: {result['connectivity_rate']:.3f}")
        
        # 总体统计
        total_triples = sum(r['total_triples'] for r in self.results.values())
        total_components = sum(r['connected_components'] for r in self.results.values())
        avg_connectivity = sum(r['connectivity_rate'] for r in self.results.values()) / len(self.results)
        
        print(f"\nOverall Statistics:")
        print("-" * 40)
        print(f"Total domains analyzed: {len(self.results)}")
        print(f"Total triples across all domains: {total_triples:,}")
        print(f"Total connected components: {total_components:,}")
        print(f"Average connectivity rate: {avg_connectivity:.4f} ({avg_connectivity*100:.2f}%)")

def main():
    # 设置基础目录和线程数
    base_directory = "/home/lsz/OneGraph/sorted_data"
    max_workers = 32  # 可以根据你的CPU核心数调整
    
    # 创建分析器
    analyzer = DomainGraphAnalyzer(base_directory, max_workers=max_workers)
    
    # 分析所有领域
    print(f"Starting domain analysis with {max_workers} threads...")
    
    # parallel_domains=False: 串行分析领域，但每个领域内部并行处理CSV
    # parallel_domains=True: 并行分析领域（适用于领域很多的情况）
    analyzer.analyze_all_domains(parallel_domains=False)
    
    # 打印总结
    analyzer.print_summary()
    
    # 保存结果
    analyzer.save_results("domain_connectivity_analysis.json")
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()