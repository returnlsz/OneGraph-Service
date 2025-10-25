import pandas as pd
import csv
import os
import argparse
import time
from typing import Optional, List
import glob

class CSVRowCounter:
    def __init__(self, csv_file: str = None, folder_path: str = None):
        self.csv_file = csv_file
        self.folder_path = folder_path
        
    def get_csv_files_in_folder(self) -> List[str]:
        """
        获取文件夹下所有CSV文件的路径
        """
        if not self.folder_path:
            return []
            
        csv_files = []
        # 使用glob查找所有.csv文件
        pattern = os.path.join(self.folder_path, "*.csv")
        csv_files.extend(glob.glob(pattern))
        
        # 递归查找子文件夹中的CSV文件（可选）
        pattern_recursive = os.path.join(self.folder_path, "**", "*.csv")
        csv_files.extend(glob.glob(pattern_recursive, recursive=True))
        
        # 去重并排序
        csv_files = sorted(list(set(csv_files)))
        return csv_files
    
    def count_folder_csv_rows(self, method: str = 'auto', show_details: bool = True) -> dict:
        """
        统计文件夹下所有CSV文件的总行数
        """
        if not self.folder_path:
            return {'error': 'No folder path specified'}
            
        if not os.path.exists(self.folder_path):
            return {'error': f'Folder does not exist: {self.folder_path}'}
        
        csv_files = self.get_csv_files_in_folder()
        
        if not csv_files:
            return {'error': 'No CSV files found in the folder'}
        
        print(f"Found {len(csv_files)} CSV files in folder: {self.folder_path}")
        print("=" * 80)
        
        total_rows = 0
        total_size = 0
        file_details = []
        failed_files = []
        
        start_time = time.time()
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\nProcessing [{i}/{len(csv_files)}]: {os.path.basename(csv_file)}")
            
            try:
                # 获取文件大小
                file_size = os.path.getsize(csv_file)
                total_size += file_size
                
                # 创建单个文件的计数器
                file_counter = CSVRowCounter(csv_file=csv_file)
                
                # 根据文件大小选择方法
                if method == 'auto':
                    if file_size < 100 * 1024 * 1024:  # < 100MB
                        count = file_counter.count_rows_pandas_optimized()
                    else:
                        count = file_counter.count_rows_buffered()
                elif method == 'pandas':
                    count = file_counter.count_rows_pandas_optimized()
                elif method == 'buffered':
                    count = file_counter.count_rows_buffered()
                elif method == 'csv':
                    count = file_counter.count_rows_csv_module()
                elif method == 'lines':
                    count = file_counter.count_rows_simple_lines()
                else:
                    count = file_counter.count_rows_pandas_optimized()
                
                if count > 0:
                    total_rows += count
                    file_details.append({
                        'file': csv_file,
                        'rows': count,
                        'size_mb': file_size / (1024 * 1024)
                    })
                    
                    if show_details:
                        print(f"  Rows: {count:,} | Size: {file_size/(1024*1024):.2f} MB")
                else:
                    failed_files.append(csv_file)
                    print(f"  FAILED to count rows")
                    
            except Exception as e:
                failed_files.append(csv_file)
                print(f"  ERROR: {e}")
        
        elapsed_time = time.time() - start_time
        
        # 汇总结果
        result = {
            'total_files': len(csv_files),
            'successful_files': len(file_details),
            'failed_files': len(failed_files),
            'total_rows': total_rows,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024),
            'processing_time_seconds': elapsed_time,
            'file_details': file_details,
            'failed_files': failed_files
        }
        
        return result
    
    def count_rows_pandas_basic(self) -> int:
        """
        方法1: 使用pandas基本读取
        适用于: 中小型文件
        """
        print("Method 1: Using pandas basic read...")
        start_time = time.time()
        
        try:
            df = pd.read_csv(self.csv_file)
            row_count = len(df)
            
            elapsed = time.time() - start_time
            print(f"Pandas basic: {row_count:,} rows in {elapsed:.2f}s")
            return row_count
            
        except Exception as e:
            print(f"Error with pandas basic method: {e}")
            return -1
    
    def count_rows_pandas_optimized(self) -> int:
        """
        方法2: 使用pandas优化读取（只读一列）
        适用于: 大型文件，更快
        """
        if not hasattr(self, '_show_method_info'):
            print("Method 2: Using pandas optimized (single column)...")
        start_time = time.time()
        
        try:
            # 只读取第一列来计算行数，更快
            df = pd.read_csv(self.csv_file, usecols=[0])
            row_count = len(df)
            
            elapsed = time.time() - start_time
            if not hasattr(self, '_show_method_info'):
                print(f"Pandas optimized: {row_count:,} rows in {elapsed:.2f}s")
            return row_count
            
        except Exception as e:
            print(f"Error with pandas optimized method: {e}")
            return -1
    
    def count_rows_pandas_chunked(self, chunk_size: int = 100000) -> int:
        """
        方法3: 使用pandas分块读取
        适用于: 超大文件，内存受限
        """
        print(f"Method 3: Using pandas chunked read (chunk_size={chunk_size:,})...")
        start_time = time.time()
        
        try:
            total_rows = 0
            chunk_count = 0
            
            # 分块读取，只需要第一列
            for chunk in pd.read_csv(self.csv_file, chunksize=chunk_size, usecols=[0]):
                total_rows += len(chunk)
                chunk_count += 1
                
                if chunk_count % 10 == 0:  # 每10个chunk显示进度
                    print(f"  Processed {chunk_count} chunks, {total_rows:,} rows so far...")
            
            elapsed = time.time() - start_time
            print(f"Pandas chunked: {total_rows:,} rows in {elapsed:.2f}s")
            return total_rows
            
        except Exception as e:
            print(f"Error with pandas chunked method: {e}")
            return -1
    
    def count_rows_csv_module(self) -> int:
        """
        方法4: 使用Python内置csv模块
        适用于: 内存效率最高，但可能较慢
        """
        if not hasattr(self, '_show_method_info'):
            print("Method 4: Using Python csv module...")
        start_time = time.time()
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                row_count = sum(1 for row in csv_reader)
            
            elapsed = time.time() - start_time
            if not hasattr(self, '_show_method_info'):
                print(f"CSV module: {row_count:,} rows in {elapsed:.2f}s")
            return row_count
            
        except Exception as e:
            print(f"Error with csv module method: {e}")
            return -1
    
    def count_rows_simple_lines(self) -> int:
        """
        方法5: 简单按行计数
        适用于: 最快的方法，但不处理CSV格式问题
        """
        if not hasattr(self, '_show_method_info'):
            print("Method 5: Using simple line counting...")
        start_time = time.time()
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                row_count = sum(1 for line in file)
            
            elapsed = time.time() - start_time
            if not hasattr(self, '_show_method_info'):
                print(f"Simple lines: {row_count:,} rows in {elapsed:.2f}s")
            return row_count
            
        except Exception as e:
            print(f"Error with simple lines method: {e}")
            return -1
    
    def count_rows_buffered(self, buffer_size: int = 8192*1024) -> int:
        """
        方法6: 使用缓冲读取计数换行符
        适用于: 超大文件的最快方法
        """
        if not hasattr(self, '_show_method_info'):
            print(f"Method 6: Using buffered line counting (buffer_size={buffer_size:,})...")
        start_time = time.time()
        
        try:
            row_count = 0
            with open(self.csv_file, 'rb') as file:
                while True:
                    buffer = file.read(buffer_size)
                    if not buffer:
                        break
                    row_count += buffer.count(b'\n')
            
            elapsed = time.time() - start_time
            if not hasattr(self, '_show_method_info'):
                print(f"Buffered counting: {row_count:,} rows in {elapsed:.2f}s")
            return row_count
            
        except Exception as e:
            print(f"Error with buffered method: {e}")
            return -1
    
    def count_with_header_info(self, method: str = 'auto') -> dict:
        """
        计算行数并提供额外信息（是否有header等）
        """
        print("Analyzing CSV structure...")
        
        result = {
            'total_rows': 0,
            'has_header': False,
            'header_row': None,
            'data_rows': 0
        }
        
        try:
            # 先读取前几行来判断是否有header
            sample_df = pd.read_csv(self.csv_file, nrows=5)
            
            # 尝试检测header
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                sniffer = csv.Sniffer()
                sample = file.read(1024)
                file.seek(0)
                has_header = sniffer.has_header(sample)
                
                if has_header:
                    header_row = next(csv.reader(file))
                    result['has_header'] = True
                    result['header_row'] = header_row
            
            # 计算总行数
            if method == 'auto':
                file_size = os.path.getsize(self.csv_file)
                if file_size < 100 * 1024 * 1024:  # < 100MB
                    total_rows = self.count_rows_pandas_optimized()
                else:
                    total_rows = self.count_rows_buffered()
            elif method == 'pandas':
                total_rows = self.count_rows_pandas_optimized()
            elif method == 'buffered':
                total_rows = self.count_rows_buffered()
            else:
                total_rows = self.count_rows_simple_lines()
            
            result['total_rows'] = total_rows
            result['data_rows'] = total_rows - (1 if result['has_header'] else 0)
            
            return result
            
        except Exception as e:
            print(f"Error analyzing CSV: {e}")
            return result
    
    def compare_all_methods(self, chunk_size: int = 100000) -> dict:
        """
        比较所有方法的性能
        """
        print("Comparing all counting methods...")
        print("=" * 60)
        
        methods = [
            ('pandas_optimized', self.count_rows_pandas_optimized),
            ('pandas_chunked', lambda: self.count_rows_pandas_chunked(chunk_size)),
            ('csv_module', self.count_rows_csv_module),
            ('simple_lines', self.count_rows_simple_lines),
            ('buffered', self.count_rows_buffered)
        ]
        
        results = {}
        
        for method_name, method_func in methods:
            print(f"\n--- {method_name.upper()} ---")
            try:
                count = method_func()
                results[method_name] = count
            except Exception as e:
                print(f"Failed: {e}")
                results[method_name] = -1
        
        return results
    
    def get_file_info(self) -> dict:
        """
        获取文件基本信息
        """
        try:
            file_size = os.path.getsize(self.csv_file)
            file_info = {
                'file_path': self.csv_file,
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'file_size_gb': file_size / (1024 * 1024 * 1024),
                'exists': True
            }
            return file_info
        except Exception as e:
            return {'exists': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Count rows in CSV file(s)')
    
    # 互斥组：要么指定文件，要么指定文件夹
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('csv_file', nargs='?', help='Path to a single CSV file')
    group.add_argument('-f', '--folder', help='Path to folder containing CSV files')
    
    parser.add_argument('-m', '--method', 
                       choices=['pandas', 'pandas_chunked', 'csv', 'lines', 'buffered', 'compare', 'auto'],
                       default='auto', help='Counting method to use')
    parser.add_argument('-c', '--chunk-size', type=int, default=100000,
                       help='Chunk size for chunked methods (default: 100000)')
    parser.add_argument('-i', '--info', action='store_true',
                       help='Show detailed file and structure information')
    parser.add_argument('--no-details', action='store_true',
                       help='Hide per-file details when processing folder')
    
    args = parser.parse_args()
    
    # 处理文件夹模式
    if args.folder:
        if not os.path.exists(args.folder):
            print(f"Error: Folder '{args.folder}' does not exist!")
            return
        
        counter = CSVRowCounter(folder_path=args.folder)
        result = counter.count_folder_csv_rows(
            method=args.method, 
            show_details=not args.no_details
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # 显示汇总结果
        print("\n" + "=" * 80)
        print("FOLDER SUMMARY:")
        print("=" * 80)
        print(f"Total CSV files found: {result['total_files']}")
        print(f"Successfully processed: {result['successful_files']}")
        print(f"Failed to process: {result['failed_files']}")
        print(f"Total rows across all files: {result['total_rows']:,}")
        print(f"Total size: {result['total_size_gb']:.2f} GB ({result['total_size_mb']:.2f} MB)")
        print(f"Processing time: {result['processing_time_seconds']:.2f} seconds")
        
        if result['failed_files']:
            print(f"\nFailed files:")
            for failed_file in result['failed_files']:
                print(f"  - {failed_file}")
        
        # 显示最大的几个文件
        if result['file_details']:
            print(f"\nTop 5 largest files by row count:")
            sorted_files = sorted(result['file_details'], key=lambda x: x['rows'], reverse=True)
            for i, file_info in enumerate(sorted_files[:5], 1):
                print(f"  {i}. {os.path.basename(file_info['file'])}: {file_info['rows']:,} rows")
        
        return
    
    # 处理单个文件模式
    if not args.csv_file:
        print("Error: Please specify either a CSV file or use --folder option!")
        return
        
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' does not exist!")
        return
    
    # 创建计数器
    counter = CSVRowCounter(csv_file=args.csv_file)
    
    # 显示文件信息
    file_info = counter.get_file_info()
    print(f"File: {file_info['file_path']}")
    print(f"Size: {file_info['file_size_mb']:.2f} MB ({file_info['file_size_bytes']:,} bytes)")
    print("=" * 60)
    
    # 执行计数
    if args.method == 'compare':
        results = counter.compare_all_methods(args.chunk_size)
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS:")
        print(f"{'='*60}")
        for method, count in results.items():
            if count > 0:
                print(f"{method:15s}: {count:,} rows")
            else:
                print(f"{method:15s}: FAILED")
                
    elif args.method == 'auto':
        if args.info:
            result = counter.count_with_header_info('auto')
            print(f"\nDETAILED ANALYSIS:")
            print(f"Total rows: {result['total_rows']:,}")
            print(f"Has header: {result['has_header']}")
            if result['has_header']:
                print(f"Header: {result['header_row']}")
                print(f"Data rows: {result['data_rows']:,}")
        else:
            file_size = file_info['file_size_mb']
            if file_size < 100:
                count = counter.count_rows_pandas_optimized()
            else:
                count = counter.count_rows_buffered()
            print(f"\nFinal result: {count:,} rows")
    
    elif args.method == 'pandas':
        count = counter.count_rows_pandas_optimized()
        print(f"\nFinal result: {count:,} rows")
        
    elif args.method == 'pandas_chunked':
        count = counter.count_rows_pandas_chunked(args.chunk_size)
        print(f"\nFinal result: {count:,} rows")
        
    elif args.method == 'csv':
        count = counter.count_rows_csv_module()
        print(f"\nFinal result: {count:,} rows")
        
    elif args.method == 'lines':
        count = counter.count_rows_simple_lines()
        print(f"\nFinal result: {count:,} rows")
        
    elif args.method == 'buffered':
        count = counter.count_rows_buffered()
        print(f"\nFinal result: {count:,} rows")

# 简单使用示例
def simple_count_example():
    """简单使用示例"""
    # 单个文件
    csv_file = "your_file.csv"  # 替换为你的文件路径
    counter = CSVRowCounter(csv_file=csv_file)
    
    # 快速计数（自动选择最佳方法）
    file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
    if file_size < 100:
        count = counter.count_rows_pandas_optimized()
    else:
        count = counter.count_rows_buffered()
    
    print(f"Total rows: {count:,}")
    
    # 文件夹中所有CSV文件
    folder_path = "your_folder_path"  # 替换为你的文件夹路径
    folder_counter = CSVRowCounter(folder_path=folder_path)
    result = folder_counter.count_folder_csv_rows()
    
    if 'error' not in result:
        print(f"Total rows across all CSV files: {result['total_rows']:,}")

if __name__ == "__main__":
    main()