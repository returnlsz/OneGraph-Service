import csv
import random
import os
from pathlib import Path

def batch_triple_classifier(input_file, output_dir="/home/lsz/OneGraph/data", batch_size=20):
    """批量三元组分类器 - 简洁版"""
    
    # 领域映射
    domains = {
        '1': '自然科学', '2': '工程技术', '3': '医药卫生', '4': '农学',
        '5': '社会科学', '6': '人文学科', '7': '其他'
    }
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取所有数据到内存（如果文件太大可以改为索引方式）
    print("正在读取文件...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        all_data = [row for row in reader if len(row) >= 3]
    
    print(f"文件共有 {len(all_data):,} 行有效数据")
    
    total_processed = 0
    batch_num = 0
    
    try:
        while True:
            batch_num += 1
            print(f"\n{'='*20} 第 {batch_num} 批次 {'='*20}")
            
            # 随机采样一批
            batch_samples = random.sample(all_data, min(batch_size, len(all_data)))
            print(f"已采样 {len(batch_samples)} 个样本")
            
            # 逐个处理批次中的样本
            for i, sample in enumerate(batch_samples, 1):
                print(f"\n--- 样本 {i}/{len(batch_samples)} ---")
                print(f"主语: {sample[0]}")
                print(f"谓语: {sample[1]}")  
                print(f"宾语: {sample[2]}")
                
                print("\n领域: 1-自然科学 2-工程技术 3-医药卫生 4-农学 5-社会科学 6-人文学科 7-其他")
                
                while True:
                    choice = input("选择 (1-7分类, s跳过, n下批次, q退出): ").strip()
                    
                    if choice == 'q':
                        raise KeyboardInterrupt
                    elif choice == 'n':
                        break
                    elif choice == 's':
                        print("跳过")
                        break
                    elif choice in domains:
                        # 保存样本
                        output_file = Path(output_dir) / f"{domains[choice]}.csv"
                        
                        if not output_file.exists():
                            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                                csv.writer(f).writerow(['subject', 'predicate', 'object'])
                        
                        with open(output_file, 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow(sample)
                        
                        print(f"✓ 保存到 {domains[choice]}")
                        total_processed += 1
                        break
                    else:
                        print("无效输入!")
                
                if choice == 'n':  # 跳到下一批次
                    break
            
            # 显示统计
            print(f"\n批次完成! 累计处理: {total_processed} 个样本")
            
            # 询问是否继续
            if input("继续下批次? (回车继续, q退出): ").strip() == 'q':
                break
                
    except KeyboardInterrupt:
        print("\n程序结束")
    
    print(f"\n最终统计: 共处理 {total_processed} 个样本")

# 使用
if __name__ == "__main__":
    input_file = input("CSV文件路径: ")
    batch_size = input("批次大小 (默认20): ")
    batch_size = int(batch_size) if batch_size.isdigit() else 20
    
    batch_triple_classifier(input_file, batch_size=batch_size)