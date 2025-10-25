import pandas as pd
import os
from pathlib import Path

def extract_entities_from_subdomain(csv_file_path, output_dir):
    """
    从单个子领域的CSV文件中提取实体并保存到txt文件
    
    参数:
    csv_file_path: CSV文件路径
    output_dir: 输出目录
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 将列名转换为小写以便匹配
        df.columns = df.columns.str.lower()
        
        # 检查是否包含head和tail列
        if 'head' not in df.columns or 'tail' not in df.columns:
            print(f"警告: 文件 {csv_file_path} 缺少head或tail列")
            return 0
        
        # 提取所有实体（head和tail）
        entities = set()
        
        # 添加head列的所有值
        head_entities = df['head'].dropna().astype(str).tolist()
        entities.update(head_entities)
        
        # 添加tail列的所有值
        tail_entities = df['tail'].dropna().astype(str).tolist()
        entities.update(tail_entities)
        
        # 移除空字符串和'nan'
        entities = {entity.strip() for entity in entities if entity.strip() and entity.lower() != 'nan'}
        
        if not entities:
            print(f"警告: 文件 {csv_file_path} 中没有找到有效实体")
            return 0
        
        # 生成输出文件名
        subdomain_name = Path(csv_file_path).stem.replace('_all', '')
        output_file = os.path.join(output_dir, f"{subdomain_name}_entities.txt")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 将实体写入txt文件，每行一个
        with open(output_file, 'w', encoding='utf-8') as f:
            for entity in sorted(entities):  # 排序以便于查看
                f.write(f"{entity}\n")
        
        print(f"成功处理: {subdomain_name} - 提取了 {len(entities)} 个唯一实体")
        return len(entities)
        
    except Exception as e:
        print(f"错误: 处理文件 {csv_file_path} 时出现问题: {str(e)}")
        return 0

def process_all_domains(base_path, output_base_path=None):
    """
    处理所有领域下的所有子领域
    
    参数:
    base_path: 基础路径 (/home/lsz/OneGraph/sorted_data_v2)
    output_base_path: 输出基础路径，如果为None则在原路径下创建entities文件夹
    """
    
    if output_base_path is None:
        output_base_path = os.path.join(base_path, "extracted_entities")
    
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"错误: 基础路径 {base_path} 不存在")
        return
    
    total_entities = 0
    processed_subdomains = 0
    
    # 遍历所有领域文件夹
    for domain_dir in base_path.iterdir():
        if not domain_dir.is_dir():
            continue
            
        domain_name = domain_dir.name
        print(f"\n正在处理领域: {domain_name}")
        
        # 创建该领域的输出目录
        domain_output_dir = os.path.join(output_base_path, domain_name)
        
        # 遍历该领域下的所有子领域
        for subdomain_dir in domain_dir.iterdir():
            if not subdomain_dir.is_dir():
                continue
                
            subdomain_name = subdomain_dir.name
            
            # 查找以_all.csv结尾的文件
            csv_file_path = subdomain_dir / f"{subdomain_name}_all.csv"
            
            if csv_file_path.exists():
                print(f"  处理子领域: {subdomain_name}")
                entity_count = extract_entities_from_subdomain(str(csv_file_path), domain_output_dir)
                total_entities += entity_count
                processed_subdomains += 1
            else:
                print(f"  跳过: {subdomain_name} (未找到 {subdomain_name}_all.csv)")
    
    print(f"\n处理完成!")
    print(f"总共处理了 {processed_subdomains} 个子领域")
    print(f"提取了 {total_entities} 个实体")
    print(f"输出目录: {output_base_path}")

def process_single_domain(base_path, domain_name, output_base_path=None):
    """
    处理单个领域下的所有子领域
    
    参数:
    base_path: 基础路径
    domain_name: 领域名称
    output_base_path: 输出基础路径
    """
    
    if output_base_path is None:
        output_base_path = os.path.join(base_path, "extracted_entities")
    
    domain_path = Path(base_path) / domain_name
    
    if not domain_path.exists():
        print(f"错误: 领域路径 {domain_path} 不存在")
        return
    
    print(f"正在处理领域: {domain_name}")
    
    domain_output_dir = os.path.join(output_base_path, domain_name)
    total_entities = 0
    processed_subdomains = 0
    
    # 遍历该领域下的所有子领域
    for subdomain_dir in domain_path.iterdir():
        if not subdomain_dir.is_dir():
            continue
            
        subdomain_name = subdomain_dir.name
        csv_file_path = subdomain_dir / f"{subdomain_name}_all.csv"
        
        if csv_file_path.exists():
            print(f"  处理子领域: {subdomain_name}")
            entity_count = extract_entities_from_subdomain(str(csv_file_path), domain_output_dir)
            total_entities += entity_count
            processed_subdomains += 1
        else:
            print(f"  跳过: {subdomain_name} (未找到 {subdomain_name}_all.csv)")
    
    print(f"\n处理完成!")
    print(f"处理了 {processed_subdomains} 个子领域")
    print(f"提取了 {total_entities} 个实体")

def main():
    # 基础路径
    base_path = "/home/lsz/OneGraph/sorted_data_v2"
    
    print("请选择处理模式:")
    print("1. 处理所有领域")
    print("2. 处理单个领域")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        # 处理所有领域
        output_path = input("请输入输出路径 (回车使用默认路径): ").strip()
        if not output_path:
            output_path = None
        
        process_all_domains(base_path, output_path)
        
    elif choice == "2":
        # 处理单个领域
        domain_name = input("请输入领域名称 (例如: agriculture): ").strip()
        output_path = input("请输入输出路径 (回车使用默认路径): ").strip()
        if not output_path:
            output_path = None
            
        process_single_domain(base_path, domain_name, output_path)
        
    else:
        print("无效选择")

if __name__ == "__main__":
    main()