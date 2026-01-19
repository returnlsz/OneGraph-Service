import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json
import re
import string

def is_answer_correct(answer, prediction):
    """
    判断预测答案是否正确，支持多种匹配策略
    """
    if pd.isna(prediction) or pd.isna(answer):
        return False
        
    answer = str(answer).strip().upper()
    prediction = str(prediction).strip().upper()
    
    # 跳过空预测
    if prediction == '' or prediction == 'NAN':
        return False
    
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
    answer_options = re.findall(r'\b[A-Z]\b', answer)
    prediction_options = re.findall(r'\b[A-Z]\b', prediction)
    
    if answer_options and prediction_options:
        # 如果都包含选项字母，比较第一个选项
        if answer_options[0] == prediction_options[0]:
            return True
    
    # 策略5: 去除标点符号后比较
    answer_clean = answer.translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
    prediction_clean = prediction.translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
    
    if answer_clean == prediction_clean:
        return True
    
    return False

def calculate_accuracy_for_dataset(dataset_path):
    """
    计算单个数据集的各种预测格式正确率
    """
    # 所有可能的预测格式
    all_prediction_formats = [
        'origin_prediction',  # 直接问答
        'edgetable_prediction',  # 带知识的预测格式
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
    
    # 检查answer列是否存在
    if 'answer' not in combined_df.columns:
        print(f"Warning: 'answer' column not found in {dataset_path}")
        print(f"Available columns: {list(combined_df.columns)}")
        return None
    
    # 检测数据集中实际存在的预测格式
    available_formats = []
    for format_name in all_prediction_formats:
        if format_name in combined_df.columns:
            available_formats.append(format_name)
    
    if not available_formats:
        print(f"Warning: No prediction columns found in {dataset_path}")
        print(f"Available columns: {list(combined_df.columns)}")
        return None
    
    # 计算各格式正确率
    results = {}
    total_samples = len(combined_df)
    
    for format_name in available_formats:
        correct_count = 0
        valid_count = 0  # 有效预测的数量（非空）
        
        for idx, row in combined_df.iterrows():
            answer = row['answer']
            prediction = row[format_name]
            
            # 跳过空预测
            if pd.isna(prediction) or str(prediction).strip() == '' or str(prediction).strip().lower() == 'nan':
                continue
                
            valid_count += 1
            
            # 判断是否正确
            if is_answer_correct(answer, prediction):
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
    detailed_df.to_csv('ceval_unified_detailed_results.csv', index=False, encoding='utf-8')
    
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
    overall_df.to_csv('ceval_unified_overall_statistics.csv', index=False, encoding='utf-8')
    
    # 3. 保存数据集汇总（每个数据集的所有格式）
    dataset_summary = []
    for dataset_name, results in all_results.items():
        row = {'dataset': dataset_name}
        
        # 添加每种格式的准确率
        all_formats = ['origin_prediction', 'edgetable_prediction', 'nodesequence_prediction', 
                      'code_prediction', 'syntaxtree_prediction', 'naturallanguage_prediction']
        
        for format_name in all_formats:
            if format_name in results:
                row[f'{format_name}_accuracy'] = results[format_name]['accuracy']
                row[f'{format_name}_correct'] = results[format_name]['correct']
                row[f'{format_name}_valid_total'] = results[format_name]['valid_total']
            else:
                row[f'{format_name}_accuracy'] = None
                row[f'{format_name}_correct'] = None
                row[f'{format_name}_valid_total'] = None
        
        # 计算该数据集的平均准确率（只计算存在的格式）
        accuracies = [results[fmt]['accuracy'] for fmt in results.keys()]
        row['average_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else 0
        row['total_samples'] = list(results.values())[0]['total_samples'] if results else 0
        row['available_formats'] = ', '.join(results.keys())
        
        dataset_summary.append(row)
    
    dataset_df = pd.DataFrame(dataset_summary)
    dataset_df = dataset_df.sort_values('average_accuracy', ascending=False)
    dataset_df.to_csv('ceval_unified_dataset_summary.csv', index=False, encoding='utf-8')
    
    # 4. 保存格式对比矩阵（每个数据集作为行，每种格式作为列）
    if detailed_data:
        pivot_data = detailed_df.pivot(index='dataset', columns='prediction_format', values='accuracy')
        pivot_data.to_csv('ceval_unified_accuracy_matrix.csv', encoding='utf-8')
    else:
        pivot_data = pd.DataFrame()
    
    # 5. 保存格式覆盖率统计
    format_coverage = {}
    total_datasets = len(all_results)
    
    for format_name in ['origin_prediction', 'edgetable_prediction', 'nodesequence_prediction', 
                       'code_prediction', 'syntaxtree_prediction', 'naturallanguage_prediction']:
        datasets_with_format = sum(1 for results in all_results.values() if format_name in results)
        format_coverage[format_name] = {
            'datasets_count': datasets_with_format,
            'coverage_percentage': datasets_with_format / total_datasets * 100 if total_datasets > 0 else 0
        }
    
    coverage_data = []
    for format_name, stats in format_coverage.items():
        coverage_data.append({
            'prediction_format': format_name,
            'datasets_with_format': stats['datasets_count'],
            'total_datasets': total_datasets,
            'coverage_percentage': stats['coverage_percentage']
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df = coverage_df.sort_values('coverage_percentage', ascending=False)
    coverage_df.to_csv('ceval_unified_format_coverage.csv', index=False, encoding='utf-8')
    
    return detailed_df, overall_df, dataset_df, pivot_data, coverage_df

def main():
    # 主目录路径
    main_dir = "/disk0/lsz/datasets/ceval/ceval-exam-retrieve-enrich-prediction-gpt-4o"
    
    # 存储所有结果
    all_results = {}
    overall_stats = defaultdict(lambda: {'correct': 0, 'valid_total': 0, 'total_samples': 0})
    
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
            for format_name, stats in results.items():
                overall_stats[format_name]['correct'] += stats['correct']
                overall_stats[format_name]['valid_total'] += stats['valid_total']
                overall_stats[format_name]['total_samples'] += stats['total_samples']
            
            # 实时显示进度
            print(f"  -> Found formats: {list(results.keys())}")
            for format_name, stats in results.items():
                print(f"     {format_name}: {stats['correct']}/{stats['valid_total']} = {stats['accuracy']:.4f}")
        else:
            print(f"  -> Failed to process")
    
    # 打印详细结果
    print("\n" + "="*100)
    print("DETAILED RESULTS BY DATASET")
    print("="*100)
    
    for dataset_name, results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 60)
        for format_name, stats in results.items():
            print(f"{format_name:30s}: {stats['correct']:4d}/{stats['valid_total']:4d} = {stats['accuracy']:.4f} "
                  f"(Total samples: {stats['total_samples']})")
    
    # 打印总体统计
    print("\n" + "="*100)
    print("OVERALL STATISTICS BY FORMAT")
    print("="*100)
    
    format_accuracies = []
    for format_name, stats in overall_stats.items():
        if stats['valid_total'] > 0:
            overall_accuracy = stats['correct'] / stats['valid_total']
            format_accuracies.append((format_name, overall_accuracy))
            print(f"{format_name:30s}: {stats['correct']:5d}/{stats['valid_total']:5d} = {overall_accuracy:.4f}")
        else:
            print(f"{format_name:30s}: No valid predictions found")
    
    # 按准确率排序
    format_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*100)
    print("RANKING BY ACCURACY")
    print("="*100)
    
    for i, (format_name, accuracy) in enumerate(format_accuracies, 1):
        print(f"{i}. {format_name:30s}: {accuracy:.4f}")
    
    # 打印格式覆盖率统计
    print("\n" + "="*100)
    print("FORMAT COVERAGE STATISTICS")
    print("="*100)
    
    total_datasets = len(all_results)
    all_formats = ['origin_prediction', 'edgetable_prediction', 'nodesequence_prediction', 
                   'code_prediction', 'syntaxtree_prediction', 'naturallanguage_prediction']
    
    for format_name in all_formats:
        datasets_with_format = sum(1 for results in all_results.values() if format_name in results)
        coverage_percentage = datasets_with_format / total_datasets * 100 if total_datasets > 0 else 0
        print(f"{format_name:30s}: {datasets_with_format:3d}/{total_datasets:3d} datasets ({coverage_percentage:5.1f}%)")
    
    # 分析数据集类型分布
    print("\n" + "="*100)
    print("DATASET TYPE ANALYSIS")
    print("="*100)
    
    qa_only_datasets = []  # 只有origin_prediction的数据集
    knowledge_datasets = []  # 有知识增强格式的数据集
    mixed_datasets = []  # 既有QA又有知识增强的数据集
    
    for dataset_name, results in all_results.items():
        has_origin = 'origin_prediction' in results
        has_knowledge = any(fmt in results for fmt in ['edgetable_prediction', 'nodesequence_prediction', 
                                                      'code_prediction', 'syntaxtree_prediction', 
                                                      'naturallanguage_prediction'])
        
        if has_origin and has_knowledge:
            mixed_datasets.append(dataset_name)
        elif has_origin:
            qa_only_datasets.append(dataset_name)
        elif has_knowledge:
            knowledge_datasets.append(dataset_name)
    
    print(f"QA Only datasets:        {len(qa_only_datasets):3d}")
    print(f"Knowledge Only datasets: {len(knowledge_datasets):3d}")
    print(f"Mixed datasets:          {len(mixed_datasets):3d}")
    print(f"Total datasets:          {len(all_results):3d}")
    
    # 保存结果到JSON文件
    output_file = "ceval_unified_evaluation_results.json"
    
    # 准备保存的数据
    save_data = {
        'detailed_results': all_results,
        'overall_statistics': dict(overall_stats),
        'ranking': format_accuracies,
        'dataset_analysis': {
            'qa_only_datasets': qa_only_datasets,
            'knowledge_only_datasets': knowledge_datasets,
            'mixed_datasets': mixed_datasets,
            'total_datasets': len(all_results)
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
    detailed_df, overall_df, dataset_df, pivot_data, coverage_df = save_to_csv(all_results, overall_stats, format_accuracies)
    
    print("CSV files created:")
    print("- ceval_unified_detailed_results.csv: 详细结果（每个数据集每种格式一行）")
    print("- ceval_unified_overall_statistics.csv: 总体统计（每种格式的汇总）")
    print("- ceval_unified_dataset_summary.csv: 数据集汇总（每个数据集一行，包含所有格式）")
    print("- ceval_unified_accuracy_matrix.csv: 准确率矩阵（数据集vs格式）")
    print("- ceval_unified_format_coverage.csv: 格式覆盖率统计")
    
    # 显示一些CSV文件的预览
    print("\n" + "="*100)
    print("OVERALL STATISTICS PREVIEW")
    print("="*100)
    print(overall_df.to_string(index=False))
    
    print("\n" + "="*100)
    print("FORMAT COVERAGE PREVIEW")
    print("="*100)
    print(coverage_df.to_string(index=False))
    
    print("\n" + "="*100)
    print("TOP 10 DATASETS BY AVERAGE ACCURACY")
    print("="*100)
    display_cols = ['dataset', 'average_accuracy', 'total_samples', 'available_formats']
    print(dataset_df[display_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()