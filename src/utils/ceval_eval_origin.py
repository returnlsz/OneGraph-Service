import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

def calculate_accuracy_for_dataset(dataset_path):
    """
    计算单个数据集的问答正确率
    """
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
    
    # 检查必要的列是否存在
    if 'answer' not in combined_df.columns or 'origin_prediction' not in combined_df.columns:
        print(f"Warning: Required columns 'answer' or 'origin_prediction' not found in {dataset_path}")
        available_cols = list(combined_df.columns)
        print(f"Available columns: {available_cols}")
        return None
    
    # 计算正确率
    total_samples = len(combined_df)
    correct_count = 0
    valid_count = 0  # 有效预测的数量（非空）
    
    for idx, row in combined_df.iterrows():
        answer = str(row['answer']).strip()
        prediction = str(row['origin_prediction']).strip()
        
        # 跳过空预测
        if prediction == '' or prediction == 'nan' or pd.isna(row['origin_prediction']):
            continue
            
        valid_count += 1
        
        # 判断是否正确 - 这里使用多种匹配策略
        if is_answer_correct(answer, prediction):
            correct_count += 1
    
    # 计算正确率
    if valid_count > 0:
        accuracy = correct_count / valid_count
        results = {
            'correct': correct_count,
            'valid_total': valid_count,
            'total_samples': total_samples,
            'accuracy': accuracy
        }
    else:
        results = {
            'correct': 0,
            'valid_total': 0,
            'total_samples': total_samples,
            'accuracy': 0.0
        }
    
    return results

def is_answer_correct(answer, prediction):
    """
    判断预测答案是否正确，支持多种匹配策略
    """
    answer = answer.strip().upper()
    prediction = prediction.strip().upper()
    
    # 策略1: 完全匹配
    if answer == prediction:
        return True
    
    # 策略2: 答案包含在预测中
    if answer in prediction:
        return True
    
    # 策略3: 预测包含在答案中（适用于答案较长的情况）
    if prediction in answer:
        return True
    
    # 策略4: 对于选择题，提取选项字母（A、B、C、D等）
    import re
    
    # 提取答案中的选项字母
    answer_options = re.findall(r'\b[A-Z]\b', answer)
    prediction_options = re.findall(r'\b[A-Z]\b', prediction)
    
    if answer_options and prediction_options:
        # 如果都包含选项字母，比较第一个选项
        if answer_options[0] == prediction_options[0]:
            return True
    
    # 策略5: 去除标点符号后比较
    import string
    answer_clean = answer.translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
    prediction_clean = prediction.translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
    
    if answer_clean == prediction_clean:
        return True
    
    return False

def save_to_csv(all_results, overall_stats):
    """
    将结果保存到CSV文件
    """
    # 1. 保存详细结果（每个数据集一行）
    detailed_data = []
    for dataset_name, stats in all_results.items():
        detailed_data.append({
            'dataset': dataset_name,
            'correct_count': stats['correct'],
            'valid_total': stats['valid_total'],
            'total_samples': stats['total_samples'],
            'accuracy': stats['accuracy']
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df = detailed_df.sort_values('accuracy', ascending=False)
    detailed_df.to_csv('ceval_qa_detailed_results.csv', index=False, encoding='utf-8')
    
    # 2. 保存总体统计
    if overall_stats['valid_total'] > 0:
        overall_accuracy = overall_stats['correct'] / overall_stats['valid_total']
    else:
        overall_accuracy = 0.0
    
    overall_data = [{
        'metric': 'Overall Performance',
        'total_correct': overall_stats['correct'],
        'total_valid': overall_stats['valid_total'],
        'total_samples': overall_stats['total_samples'],
        'overall_accuracy': overall_accuracy
    }]
    
    overall_df = pd.DataFrame(overall_data)
    overall_df.to_csv('ceval_qa_overall_statistics.csv', index=False, encoding='utf-8')
    
    # 3. 保存按准确率分组的统计
    accuracy_ranges = [
        (0.9, 1.0, 'Excellent (90-100%)'),
        (0.8, 0.9, 'Good (80-90%)'),
        (0.7, 0.8, 'Fair (70-80%)'),
        (0.6, 0.7, 'Poor (60-70%)'),
        (0.0, 0.6, 'Very Poor (0-60%)')
    ]
    
    range_stats = []
    for min_acc, max_acc, label in accuracy_ranges:
        datasets_in_range = [name for name, stats in all_results.items() 
                           if min_acc <= stats['accuracy'] < max_acc]
        
        if min_acc == 0.9:  # 包含100%的情况
            datasets_in_range = [name for name, stats in all_results.items() 
                               if min_acc <= stats['accuracy'] <= max_acc]
        
        range_stats.append({
            'accuracy_range': label,
            'dataset_count': len(datasets_in_range),
            'datasets': ', '.join(datasets_in_range[:10])  # 最多显示10个
        })
    
    range_df = pd.DataFrame(range_stats)
    range_df.to_csv('ceval_qa_accuracy_distribution.csv', index=False, encoding='utf-8')
    
    return detailed_df, overall_df, range_df

def main():
    # 主目录路径
    main_dir = "/disk0/lsz/datasets/ceval/ceval-exam-origin-prediction"
    
    # 存储所有结果
    all_results = {}
    overall_stats = {'correct': 0, 'valid_total': 0, 'total_samples': 0}
    
    # 遍历所有子文件夹
    subdirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    subdirs.sort()  # 按字母顺序排序
    
    print(f"Found {len(subdirs)} datasets to process...")
    
    for i, subdir in enumerate(subdirs, 1):
        subdir_path = os.path.join(main_dir, subdir)
        
        print(f"Processing [{i}/{len(subdirs)}] {subdir}...")
        
        # 计算该数据集的正确率
        results = calculate_accuracy_for_dataset(subdir_path)
        
        if results:
            all_results[subdir] = results
            
            # 累计总体统计
            overall_stats['correct'] += results['correct']
            overall_stats['valid_total'] += results['valid_total']
            overall_stats['total_samples'] += results['total_samples']
            
            # 实时显示进度
            print(f"  -> {results['correct']}/{results['valid_total']} correct = {results['accuracy']:.4f}")
        else:
            print(f"  -> Failed to process")
    
    # 打印详细结果
    print("\n" + "="*80)
    print("DETAILED RESULTS BY DATASET")
    print("="*80)
    
    # 按准确率排序显示
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for dataset_name, stats in sorted_results:
        print(f"{dataset_name:30s}: {stats['correct']:4d}/{stats['valid_total']:4d} = {stats['accuracy']:.4f} "
              f"(Total samples: {stats['total_samples']})")
    
    # 打印总体统计
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    if overall_stats['valid_total'] > 0:
        overall_accuracy = overall_stats['correct'] / overall_stats['valid_total']
        print(f"Total Correct Predictions: {overall_stats['correct']:,}")
        print(f"Total Valid Predictions:   {overall_stats['valid_total']:,}")
        print(f"Total Samples:            {overall_stats['total_samples']:,}")
        print(f"Overall Accuracy:         {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # 计算覆盖率
        coverage = overall_stats['valid_total'] / overall_stats['total_samples'] if overall_stats['total_samples'] > 0 else 0
        print(f"Prediction Coverage:      {coverage:.4f} ({coverage*100:.2f}%)")
    else:
        print("No valid predictions found across all datasets!")
    
    # 打印准确率分布统计
    print("\n" + "="*80)
    print("ACCURACY DISTRIBUTION")
    print("="*80)
    
    accuracy_ranges = [
        (0.9, 1.0, 'Excellent (90-100%)'),
        (0.8, 0.9, 'Good (80-90%)'),
        (0.7, 0.8, 'Fair (70-80%)'),
        (0.6, 0.7, 'Poor (60-70%)'),
        (0.0, 0.6, 'Very Poor (0-60%)')
    ]
    
    for min_acc, max_acc, label in accuracy_ranges:
        if min_acc == 0.9:  # 包含100%的情况
            datasets_in_range = [name for name, stats in all_results.items() 
                               if min_acc <= stats['accuracy'] <= max_acc]
        else:
            datasets_in_range = [name for name, stats in all_results.items() 
                               if min_acc <= stats['accuracy'] < max_acc]
        
        print(f"{label:20s}: {len(datasets_in_range):3d} datasets")
        if len(datasets_in_range) <= 5:
            for dataset in datasets_in_range:
                print(f"  - {dataset} ({all_results[dataset]['accuracy']:.4f})")
    
    # 显示最佳和最差的数据集
    print("\n" + "="*80)
    print("TOP 10 BEST PERFORMING DATASETS")
    print("="*80)
    
    for i, (dataset_name, stats) in enumerate(sorted_results[:10], 1):
        print(f"{i:2d}. {dataset_name:25s}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['valid_total']})")
    
    print("\n" + "="*80)
    print("TOP 10 WORST PERFORMING DATASETS")
    print("="*80)
    
    for i, (dataset_name, stats) in enumerate(sorted_results[-10:], 1):
        print(f"{i:2d}. {dataset_name:25s}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['valid_total']})")
    
    # 保存结果到JSON文件
    output_file = "ceval_qa_accuracy_results.json"
    
    # 准备保存的数据
    save_data = {
        'detailed_results': all_results,
        'overall_statistics': overall_stats,
        'summary': {
            'total_datasets': len(all_results),
            'overall_accuracy': overall_stats['correct'] / overall_stats['valid_total'] if overall_stats['valid_total'] > 0 else 0,
            'coverage': overall_stats['valid_total'] / overall_stats['total_samples'] if overall_stats['total_samples'] > 0 else 0
        }
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
    detailed_df, overall_df, range_df = save_to_csv(all_results, overall_stats)
    
    print("CSV files created:")
    print("- ceval_qa_detailed_results.csv: 详细结果（每个数据集一行）")
    print("- ceval_qa_overall_statistics.csv: 总体统计")
    print("- ceval_qa_accuracy_distribution.csv: 准确率分布统计")
    
    # 显示CSV文件的预览
    print("\n" + "="*80)
    print("DETAILED RESULTS PREVIEW (TOP 15)")
    print("="*80)
    print(detailed_df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()