import pandas as pd
import os
import glob
from pathlib import Path

def merge_csv_files(folder_path, output_file='merged_data.csv'):
    """
    合并指定文件夹下的所有CSV文件，只保留Head、Relation、Tail字段
    
    参数:
    folder_path: 包含CSV文件的文件夹路径
    output_file: 输出文件名
    """
    
    # 获取文件夹下所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"在文件夹 '{folder_path}' 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 存储所有数据的列表
    all_data = []
    
    # 定义需要的列名（支持不同的大小写组合）
    target_columns = ['head', 'relation', 'tail']
    
    for file_path in csv_files:
        try:
            print(f"正在处理: {os.path.basename(file_path)}")
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 将列名转换为小写以便匹配
            df.columns = df.columns.str.lower()
            
            # 检查是否包含所需的列
            missing_columns = [col for col in target_columns if col not in df.columns]
            if missing_columns:
                print(f"警告: 文件 {os.path.basename(file_path)} 缺少列: {missing_columns}")
                continue
            
            # 只选择需要的列
            selected_data = df[target_columns].copy()
            
            # 添加源文件信息（可选）
            selected_data['source_file'] = os.path.basename(file_path)
            
            all_data.append(selected_data)
            print(f"  - 成功读取 {len(selected_data)} 行数据")
            
        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出现问题: {str(e)}")
            continue
    
    if not all_data:
        print("没有成功读取任何数据")
        return
    
    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 如果不需要源文件信息，可以删除这一列
    # merged_df = merged_df.drop('source_file', axis=1)
    
    # 保存合并后的数据
    output_path = os.path.join(folder_path, output_file)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n合并完成!")
    print(f"总共合并了 {len(merged_df)} 行数据")
    print(f"输出文件: {output_path}")
    
    # 显示合并后数据的基本信息
    print(f"\n合并后数据预览:")
    print(merged_df.head())
    print(f"\n数据统计:")
    print(f"- 总行数: {len(merged_df)}")
    print(f"- 唯一的head数量: {merged_df['head'].nunique()}")
    print(f"- 唯一的relation数量: {merged_df['relation'].nunique()}")
    print(f"- 唯一的tail数量: {merged_df['tail'].nunique()}")

def main():
    # 设置文件夹路径
    folder_path = input("请输入CSV文件所在的文件夹路径: ").strip()
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    # 设置输出文件名
    # output_file = input("请输入输出文件名 (默认: merged_data.csv): ").strip()
    # if not output_file:
    output_file = "_all.csv"
    
    # 确保输出文件有.csv扩展名
    if not output_file.endswith('.csv'):
        output_file += '.csv'
    
    # 执行合并
    merge_csv_files(folder_path, output_file)

if __name__ == "__main__":
    main()