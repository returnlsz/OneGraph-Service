import pandas as pd
import random
import argparse
import os
from typing import List, Tuple
import sys

class TriplesSampler:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.total_rows = 0
        
    def count_total_rows(self) -> int:
        """统计CSV文件的总行数（不包括header）"""
        print("Counting total rows in CSV file...")
        try:
            # 使用pandas快速统计行数
            df = pd.read_csv(self.csv_file, usecols=[0])  # 只读第一列来统计行数
            self.total_rows = len(df)
            print(f"Total rows in CSV: {self.total_rows:,}")
            return self.total_rows
        except Exception as e:
            print(f"Error counting rows: {e}")
            return 0
    
    def sample_triples_method1(self, k: int) -> List[Tuple[str, str, str]]:
        """
        方法1: 先读取所有数据再随机采样
        适用于: 文件不太大的情况
        """
        print(f"Method 1: Loading all data and sampling {k} triples...")
        
        try:
            # 读取前三列
            df = pd.read_csv(self.csv_file, usecols=[0, 1, 2])
            
            # 重命名列为标准名称
            df.columns = ['Head', 'Relation', 'Tail']
            
            # 去除包含NaN的行
            df_clean = df.dropna()
            print(f"Valid rows after removing NaN: {len(df_clean):,}")
            
            # 如果k大于可用行数，返回所有行
            if k >= len(df_clean):
                print(f"Requested {k} samples, but only {len(df_clean)} valid rows available. Returning all.")
                k = len(df_clean)
            
            # 随机采样
            sampled_df = df_clean.sample(n=k, random_state=None)
            
            # 转换为三元组列表
            triples = [(str(row['Head']), str(row['Relation']), str(row['Tail'])) 
                      for _, row in sampled_df.iterrows()]
            
            return triples
            
        except Exception as e:
            print(f"Error in method 1: {e}")
            return []
    
    def sample_triples_method2(self, k: int, chunk_size: int = 100000) -> List[Tuple[str, str, str]]:
        """
        方法2: 分块读取并蓄水池采样
        适用于: 大文件的情况
        """
        print(f"Method 2: Reservoir sampling with chunk size {chunk_size:,}")
        
        reservoir = []  # 蓄水池
        total_seen = 0  # 已处理的行数
        
        try:
            # 分块读取CSV文件
            chunk_iter = pd.read_csv(self.csv_file, usecols=[0, 1, 2], chunksize=chunk_size)
            
            for chunk_num, chunk in enumerate(chunk_iter, 1):
                # 重命名列
                chunk.columns = ['Head', 'Relation', 'Tail']
                
                # 去除NaN
                chunk_clean = chunk.dropna()
                
                # 对当前chunk中的每一行进行蓄水池采样
                for _, row in chunk_clean.iterrows():
                    total_seen += 1
                    triple = (str(row['Head']), str(row['Relation']), str(row['Tail']))
                    
                    if len(reservoir) < k:
                        # 蓄水池未满，直接添加
                        reservoir.append(triple)
                    else:
                        # 蓄水池已满，以概率 k/total_seen 替换
                        replace_idx = random.randint(1, total_seen)
                        if replace_idx <= k:
                            reservoir[replace_idx - 1] = triple
                
                if chunk_num % 10 == 0:  # 每10个chunk打印一次进度
                    print(f"Processed {chunk_num} chunks, {total_seen:,} rows")
            
            print(f"Total rows processed: {total_seen:,}")
            print(f"Sampled {len(reservoir)} triples")
            
            return reservoir
            
        except Exception as e:
            print(f"Error in method 2: {e}")
            return []
    
    def sample_triples_method3(self, k: int) -> List[Tuple[str, str, str]]:
        """
        方法3: 随机行号采样
        适用于: 已知总行数的情况，内存效率最高
        """
        print(f"Method 3: Random line number sampling")
        
        if self.total_rows == 0:
            self.count_total_rows()
        
        if self.total_rows == 0:
            print("Cannot determine total rows")
            return []
        
        # 生成随机行号
        if k >= self.total_rows:
            print(f"Requested {k} samples, but only {self.total_rows} rows available.")
            sample_indices = list(range(self.total_rows))
        else:
            sample_indices = sorted(random.sample(range(self.total_rows), k))
        
        print(f"Sampling {len(sample_indices)} rows at random positions...")
        
        triples = []
        try:
            # 使用skiprows参数只读取指定行
            # 注意：pandas的skiprows是跳过的行号，所以需要转换
            rows_to_skip = set(range(self.total_rows)) - set(sample_indices)
            
            df = pd.read_csv(self.csv_file, 
                           usecols=[0, 1, 2], 
                           skiprows=list(rows_to_skip))
            
            df.columns = ['Head', 'Relation', 'Tail']
            df_clean = df.dropna()
            
            triples = [(str(row['Head']), str(row['Relation']), str(row['Tail'])) 
                      for _, row in df_clean.iterrows()]
            
            return triples
            
        except Exception as e:
            print(f"Error in method 3: {e}")
            return []
    
    def sample_triples_auto(self, k: int, file_size_threshold: int = 100*1024*1024) -> List[Tuple[str, str, str]]:
        """
        自动选择最适合的采样方法
        """
        # 检查文件大小
        file_size = os.path.getsize(self.csv_file)
        print(f"CSV file size: {file_size / (1024*1024):.2f} MB")
        
        if file_size < file_size_threshold:  # 小于100MB使用方法1
            print("Using method 1 (load all data)")
            return self.sample_triples_method1(k)
        else:  # 大文件使用方法2
            print("Using method 2 (reservoir sampling)")
            return self.sample_triples_method2(k)
    
    def save_sampled_triples(self, triples: List[Tuple[str, str, str]], output_file: str):
        """保存采样结果到CSV文件"""
        if not triples:
            print("No triples to save")
            return
        
        df = pd.DataFrame(triples, columns=['Head', 'Relation', 'Tail'])
        df.to_csv(output_file, index=False)
        print(f"Saved {len(triples)} triples to {output_file}")
    
    def print_sample_info(self, triples: List[Tuple[str, str, str]], show_samples: int = 5):
        """打印采样信息"""
        if not triples:
            print("No triples sampled")
            return
        
        print(f"\nSampling completed!")
        print(f"Total sampled triples: {len(triples)}")
        
        if show_samples > 0:
            print(f"\nFirst {min(show_samples, len(triples))} sampled triples:")
            for i, (head, relation, tail) in enumerate(triples[:show_samples], 1):
                print(f"{i:2d}. ({head}, {relation}, {tail})")
        
        # 统计信息
        heads = set(triple[0] for triple in triples)
        relations = set(triple[1] for triple in triples)
        tails = set(triple[2] for triple in triples)
        
        print(f"\nStatistics:")
        print(f"Unique heads: {len(heads)}")
        print(f"Unique relations: {len(relations)}")
        print(f"Unique tails: {len(tails)}")
        print(f"Unique entities (heads + tails): {len(heads | tails)}")

def main():
    parser = argparse.ArgumentParser(description='Randomly sample k triples from a CSV file')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    parser.add_argument('k', type=int, help='Number of triples to sample')
    parser.add_argument('-o', '--output', help='Output CSV file (optional)')
    parser.add_argument('-m', '--method', type=int, choices=[1, 2, 3], 
                       help='Sampling method: 1=load all, 2=reservoir, 3=random lines')
    parser.add_argument('-s', '--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--chunk-size', type=int, default=100000, 
                       help='Chunk size for method 2 (default: 100000)')
    parser.add_argument('--show-samples', type=int, default=5,
                       help='Number of sample triples to display (default: 5)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} does not exist")
        sys.exit(1)
    
    # 设置随机种子
    if args.seed:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # 创建采样器
    sampler = TriplesSampler(args.csv_file)
    
    # 执行采样
    if args.method == 1:
        triples = sampler.sample_triples_method1(args.k)
    elif args.method == 2:
        triples = sampler.sample_triples_method2(args.k, args.chunk_size)
    elif args.method == 3:
        triples = sampler.sample_triples_method3(args.k)
    else:
        # 自动选择方法
        triples = sampler.sample_triples_auto(args.k)
    
    # 打印结果信息
    sampler.print_sample_info(triples, args.show_samples)
    
    # 保存结果
    if args.output:
        sampler.save_sampled_triples(triples, args.output)
    else:
        # 生成默认输出文件名
        base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
        output_file = f"{base_name}_sampled_{args.k}.csv"
        sampler.save_sampled_triples(triples, output_file)

# 简单使用示例
def simple_example():
    """简单使用示例"""
    csv_file = "/home/lsz/OneGraph/sorted_data/agriculture_all.csv"  # 替换为你的CSV文件路径
    k = 10  # 采样数量
    
    sampler = TriplesSampler(csv_file)
    
    # 自动选择最佳方法
    triples = sampler.sample_triples_auto(k)
    
    # 打印信息
    sampler.print_sample_info(triples)
    
    # 保存结果
    sampler.save_sampled_triples(triples, f"sampled_{k}_triples.csv")

if __name__ == "__main__":
    main()