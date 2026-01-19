import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

def calculate_accuracy_for_dataset(dataset_path):
    """
    计算单个数据集的各种预测格式正确率
    """
    # 预测格式列表
    prediction_formats = [
        'edgetable_prediction',
        'nodesequence_prediction', 
        'code_prediction',
        'syntaxtree_prediction',
        'naturallanguage_prediction'
    ]
    
    # 读取parquet文件
    parquet_files = list(Path(dataset_path).glob("*.parquet"))
    if not parquet_files:
        print(f"Warning: No parquet files found in {dataset_path}")
        return None
    
    # 合并所有parquet文件的数据
    all_data = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if not all_data:
        return None
    
    # 合并数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 计算各格式正确率
    results = {}
    total_samples = len(combined_df)
    
    for format_name in prediction_formats:
        if format_name in combined_df.columns:
            # 计算正确的样本数
            correct_count = 0
            valid_count = 0  # 有效预测的数量（非空）
            
            for idx, row in combined_df.iterrows():
                answer = str(row['answer']).strip()
                prediction = str(row[format_name]).strip()
                
                # 跳过空预测
                if prediction == '' or prediction == 'nan' or pd.isna(row[format_name]):
                    continue
                    
                valid_count += 1
                
                # 判断是否正确
                if answer in prediction:
                    correct_count += 1
            
            # 计算正确率
            if valid_count > 0:
                accuracy = correct_count / valid_count
                results[format_name] = {
                    'correct': correct_count,
                    'valid_total': valid_count,
                    'total_samples': total_samples,
                    'accuracy': accuracy
                }
            else:
                results[format_name] = {
                    'correct': 0,
                    'valid_total': 0,
                    'total_samples': total_samples,
                    'accuracy': 0.0
                }
        else:
            print(f"Warning: Column '{format_name}' not found in {dataset_path}")
    
    return results

def save_to_csv(all_results, overall_stats, format_accuracies):
    """
    将结果保存到CSV文件
    """
    # 1. 保存详细结果（每个数据集的每种格式）
    detailed_data = []
    for dataset_name, results in all_results.items():
        for format_name, stats in results.items():
            detailed_data.append({
                'dataset': dataset_name,
                'prediction_format': format_name,
                'correct_count': stats['correct'],
                'valid_total': stats['valid_total'],
                'total_samples': stats['total_samples'],
                'accuracy': stats['accuracy']
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv('ceval_detailed_results.csv', index=False, encoding='utf-8')
    
    # 2. 保存总体统计
    overall_data = []
    for format_name, stats in overall_stats.items():
        if stats['valid_total'] > 0:
            overall_accuracy = stats['correct'] / stats['valid_total']
        else:
            overall_accuracy = 0.0
            
        overall_data.append({
            'prediction_format': format_name,
            'total_correct': stats['correct'],
            'total_valid': stats['valid_total'],
            'total_samples': stats['total_samples'],
            'overall_accuracy': overall_accuracy
        })
    
    overall_df = pd.DataFrame(overall_data)
    overall_df = overall_df.sort_values('overall_accuracy', ascending=False)
    overall_df.to_csv('ceval_overall_statistics.csv', index=False, encoding='utf-8')
    
    # 3. 保存数据集汇总（每个数据集的所有格式）
    dataset_summary = []
    for dataset_name, results in all_results.items():
        row = {'dataset': dataset_name}
        
        # 添加每种格式的准确率
        for format_name in ['edgetable_prediction', 'nodesequence_prediction', 
                           'code_prediction', 'syntaxtree_prediction', 'naturallanguage_prediction']:
            if format_name in results:
                row[f'{format_name}_accuracy'] = results[format_name]['accuracy']
                row[f'{format_name}_correct'] = results[format_name]['correct']
                row[f'{format_name}_valid_total'] = results[format_name]['valid_total']
            else:
                row[f'{format_name}_accuracy'] = None
                row[f'{format_name}_correct'] = None
                row[f'{format_name}_valid_total'] = None
        
        # 计算该数据集的平均准确率
        accuracies = [results[fmt]['accuracy'] for fmt in results.keys()]
        row['average_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else 0
        row['total_samples'] = list(results.values())[0]['total_samples'] if results else 0
        
        dataset_summary.append(row)
    
    dataset_df = pd.DataFrame(dataset_summary)
    dataset_df = dataset_df.sort_values('average_accuracy', ascending=False)
    dataset_df.to_csv('ceval_dataset_summary.csv', index=False, encoding='utf-8')
    
    # 4. 保存格式对比矩阵（每个数据集作为行，每种格式作为列）
    pivot_data = detailed_df.pivot(index='dataset', columns='prediction_format', values='accuracy')
    pivot_data.to_csv('ceval_accuracy_matrix.csv', encoding='utf-8')
    
    return detailed_df, overall_df, dataset_df, pivot_data

def main():
    # 主目录路径
    main_dir = "/disk0/lsz/datasets/ceval/ceval-exam-prediction"
    
    # 存储所有结果
    all_results = {}
    overall_stats = defaultdict(lambda: {'correct': 0, 'valid_total': 0, 'total_samples': 0})
    
    # 遍历所有子文件夹
    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)
        
        # 确保是文件夹
        if not os.path.isdir(subdir_path):
            continue
            
        print(f"Processing {subdir}...")
        
        # 计算该数据集的正确率
        results = calculate_accuracy_for_dataset(subdir_path)
        
        if results:
            all_results[subdir] = results
            
            # 累计总体统计
            for format_name, stats in results.items():
                overall_stats[format_name]['correct'] += stats['correct']
                overall_stats[format_name]['valid_total'] += stats['valid_total']
                overall_stats[format_name]['total_samples'] += stats['total_samples']
    
    # 打印详细结果
    print("\n" + "="*80)
    print("DETAILED RESULTS BY DATASET")
    print("="*80)
    
    for dataset_name, results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 50)
        for format_name, stats in results.items():
            print(f"{format_name:25s}: {stats['correct']:4d}/{stats['valid_total']:4d} = {stats['accuracy']:.4f} "
                  f"(Total samples: {stats['total_samples']})")
    
    # 打印总体统计
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    format_accuracies = []
    for format_name, stats in overall_stats.items():
        if stats['valid_total'] > 0:
            overall_accuracy = stats['correct'] / stats['valid_total']
            format_accuracies.append((format_name, overall_accuracy))
            print(f"{format_name:25s}: {stats['correct']:5d}/{stats['valid_total']:5d} = {overall_accuracy:.4f}")
        else:
            print(f"{format_name:25s}: No valid predictions found")
    
    # 按准确率排序
    format_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("RANKING BY ACCURACY")
    print("="*80)
    
    for i, (format_name, accuracy) in enumerate(format_accuracies, 1):
        print(f"{i}. {format_name:25s}: {accuracy:.4f}")
    
    # 保存结果到JSON文件
    output_file = "ceval_prediction_accuracy_results.json"
    
    # 准备保存的数据
    save_data = {
        'detailed_results': all_results,
        'overall_statistics': dict(overall_stats),
        'ranking': format_accuracies
    }
    
    # 转换numpy类型为Python原生类型以便JSON序列化
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy类型
            return obj.item()
        else:
            return obj
    
    save_data = convert_for_json(save_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    
    # 保存到CSV文件
    print("\nSaving results to CSV files...")
    detailed_df, overall_df, dataset_df, pivot_data = save_to_csv(all_results, overall_stats, format_accuracies)
    
    print("CSV files created:")
    print("- ceval_detailed_results.csv: 详细结果（每个数据集每种格式一行）")
    print("- ceval_overall_statistics.csv: 总体统计（每种格式的汇总）")
    print("- ceval_dataset_summary.csv: 数据集汇总（每个数据集一行，包含所有格式）")
    print("- ceval_accuracy_matrix.csv: 准确率矩阵（数据集vs格式）")
    
    # 显示一些CSV文件的预览
    print("\n" + "="*80)
    print("OVERALL STATISTICS PREVIEW")
    print("="*80)
    print(overall_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("TOP 10 DATASETS BY AVERAGE ACCURACY")
    print("="*80)
    display_cols = ['dataset', 'average_accuracy', 'total_samples']
    print(dataset_df[display_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()