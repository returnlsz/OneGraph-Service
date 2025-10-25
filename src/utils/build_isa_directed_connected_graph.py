import pandas as pd
import networkx as nx
from collections import defaultdict
import pickle
import json
import os
from tqdm import tqdm

class ISAGraphBuilder:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.isa_relations = {'isa', 'is a', 'is-a', '是', '属于'}  # 可根据需要扩展
        self.isa_graph = nx.DiGraph()
        self.non_isa_triples = []
        
    def load_and_filter_data(self):
        """加载数据并分离ISA关系"""
        print("Loading CSV file...")
        
        # 分批读取大文件
        chunk_size = 100000
        isa_triples = []
        
        for chunk in tqdm(pd.read_csv(self.csv_file, chunksize=chunk_size)):
            # 筛选ISA关系
            isa_mask = chunk['Relation'].str.lower().isin([r.lower() for r in self.isa_relations])
            
            # 收集ISA三元组
            isa_chunk = chunk[isa_mask]
            isa_triples.extend([(row['Head'], row['Relation'], row['Tail']) 
                               for _, row in isa_chunk.iterrows()])
            
            # 收集非ISA三元组
            non_isa_chunk = chunk[~isa_mask]
            self.non_isa_triples.extend([(row['Head'], row['Relation'], row['Tail']) 
                                        for _, row in non_isa_chunk.iterrows()])
        
        return isa_triples
    
    def build_isa_graph(self, isa_triples):
        """构建ISA有向图"""
        print(f"Building ISA graph with {len(isa_triples)} edges...")
        
        for head, relation, tail in tqdm(isa_triples):
            self.isa_graph.add_edge(head, tail, relation=relation)  # 注意这里使用小写的relation
    
    def find_connected_components(self):
        """找到所有连通分量"""
        print("Finding connected components...")
        
        # 将有向图转为无向图来找连通分量
        undirected_graph = self.isa_graph.to_undirected()
        components = list(nx.connected_components(undirected_graph))
        
        print(f"Found {len(components)} connected components")
        
        # 为每个连通分量创建子图
        component_graphs = []
        for i, component in enumerate(components):
            subgraph = self.isa_graph.subgraph(component).copy()
            component_graphs.append((i, subgraph))
        
        return component_graphs

class GraphStorage:
    def __init__(self, base_dir="isa_graphs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def save_components_efficiently(self, component_graphs, non_isa_triples):
        """高效存储连通分量"""
        
        # 方案1: 按大小分组存储
        self.save_by_size_groups(component_graphs)
        
        # 方案2: 创建索引文件
        self.create_index_file(component_graphs)
        
        # 方案3: 保存非ISA关系
        self.save_non_isa_relations(non_isa_triples)
    
    def save_by_size_groups(self, component_graphs):
        """按连通分量大小分组存储"""
        size_groups = defaultdict(list)
        
        for comp_id, graph in component_graphs:
            size = graph.number_of_nodes()
            if size >= 1000:
                group = "large"
            elif size >= 100:
                group = "medium"
            elif size >= 10:
                group = "small"
            else:
                group = "tiny"
            
            size_groups[group].append((comp_id, graph))
        
        # 每组保存到一个文件
        for group, components in size_groups.items():
            print(f"Saving {len(components)} {group} components...")
            
            if group == "large":
                # 大型连通分量单独保存
                for comp_id, graph in components:
                    self.save_single_component(comp_id, graph, f"{group}_components")
            else:
                # 中小型连通分量批量保存
                self.save_multiple_components(components, f"{group}_components")
    
    def save_single_component(self, comp_id, graph, subdir):
        """保存单个连通分量"""
        comp_dir = os.path.join(self.base_dir, subdir)
        os.makedirs(comp_dir, exist_ok=True)
        
        # 保存为pickle格式（高效）
        with open(os.path.join(comp_dir, f"component_{comp_id}.pkl"), 'wb') as f:
            pickle.dump(graph, f)
        
        # 修复：使用小写的'relation'而不是'Relation'
        try:
            edge_list = [(u, v, d.get('relation', '')) for u, v, d in graph.edges(data=True)]
        except Exception as e:
            print(f"Error processing edges for component {comp_id}: {e}")
            # 如果出错，创建简单的边列表
            edge_list = [(u, v, '') for u, v in graph.edges()]
        
        # 同时保存为可读的边列表格式
        with open(os.path.join(comp_dir, f"component_{comp_id}_edges.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'component_id': comp_id,
                'nodes': list(graph.nodes()),
                'edges': edge_list,
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges()
            }, f, ensure_ascii=False, indent=2)
    
    def save_multiple_components(self, components, filename):
        """批量保存多个连通分量到一个文件"""
        filepath = os.path.join(self.base_dir, f"{filename}.pkl")
        
        components_data = {}
        for comp_id, graph in components:
            try:
                edges = [(u, v, d.get('relation', '')) for u, v, d in graph.edges(data=True)]
            except Exception as e:
                print(f"Error processing edges for component {comp_id}: {e}")
                edges = [(u, v, '') for u, v in graph.edges()]
            
            components_data[comp_id] = {
                'nodes': list(graph.nodes()),
                'edges': edges,
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges()
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(components_data, f)
        
        # 创建可读的摘要文件
        summary_file = os.path.join(self.base_dir, f"{filename}_summary.json")
        summary = {
            comp_id: {
                'node_count': data['node_count'],
                'edge_count': data['edge_count']
            } for comp_id, data in components_data.items()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def create_index_file(self, component_graphs):
        """创建全局索引文件"""
        index = {
            'total_components': len(component_graphs),
            'components': {},
            'statistics': {
                'total_nodes': 0,
                'total_edges': 0,
                'size_distribution': defaultdict(int)
            }
        }
        
        for comp_id, graph in component_graphs:
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()
            
            index['components'][comp_id] = {
                'node_count': node_count,
                'edge_count': edge_count,
                'sample_nodes': list(graph.nodes())[:5],  # 保存前5个节点作为样例
            }
            
            index['statistics']['total_nodes'] += node_count
            index['statistics']['total_edges'] += edge_count
            
            # 统计大小分布
            if node_count >= 1000:
                index['statistics']['size_distribution']['large'] += 1
            elif node_count >= 100:
                index['statistics']['size_distribution']['medium'] += 1
            elif node_count >= 10:
                index['statistics']['size_distribution']['small'] += 1
            else:
                index['statistics']['size_distribution']['tiny'] += 1
        
        # 转换defaultdict为普通dict以便JSON序列化
        index['statistics']['size_distribution'] = dict(index['statistics']['size_distribution'])
        
        with open(os.path.join(self.base_dir, 'index.json'), 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    def save_non_isa_relations(self, non_isa_triples):
        """保存非ISA关系"""
        print(f"Saving {len(non_isa_triples)} non-ISA triples...")
        
        # 分批保存非ISA关系
        batch_size = 1000000
        for i in range(0, len(non_isa_triples), batch_size):
            batch = non_isa_triples[i:i+batch_size]
            filename = os.path.join(self.base_dir, f'non_isa_relations_batch_{i//batch_size}.pkl')
            
            with open(filename, 'wb') as f:
                pickle.dump(batch, f)

class GraphLoader:
    def __init__(self, base_dir="isa_graphs"):
        self.base_dir = base_dir
        index_path = os.path.join(base_dir, 'index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = None
            print("Warning: Index file not found")
    
    def load_component(self, comp_id):
        """加载指定的连通分量"""
        if not self.index:
            print("Index not available")
            return None
            
        # 根据索引信息确定文件位置
        comp_info = self.index['components'].get(str(comp_id))
        if not comp_info:
            print(f"Component {comp_id} not found in index")
            return None
            
        node_count = comp_info['node_count']
        
        if node_count >= 1000:
            filepath = os.path.join(self.base_dir, "large_components", f"component_{comp_id}.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"File not found: {filepath}")
                return None
        else:
            # 从批量文件中加载
            if node_count >= 100:
                group = "medium"
            elif node_count >= 10:
                group = "small"
            else:
                group = "tiny"
                
            filepath = os.path.join(self.base_dir, f"{group}_components.pkl")
            
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    components_data = pickle.load(f)
                    if comp_id in components_data:
                        return self.reconstruct_graph(components_data[comp_id])
                    else:
                        print(f"Component {comp_id} not found in {group} components")
                        return None
            else:
                print(f"File not found: {filepath}")
                return None
    
    def reconstruct_graph(self, component_data):
        """从数据重构图"""
        graph = nx.DiGraph()
        graph.add_nodes_from(component_data['nodes'])
        for u, v, relation in component_data['edges']:
            graph.add_edge(u, v, relation=relation)
        return graph
    
    def get_statistics(self):
        """获取统计信息"""
        if self.index:
            return self.index['statistics']
        else:
            return None

def main(csv_file):
    # 构建ISA图
    builder = ISAGraphBuilder(csv_file)
    isa_triples = builder.load_and_filter_data()
    builder.build_isa_graph(isa_triples)
    component_graphs = builder.find_connected_components()
    
    # 保存结果
    storage = GraphStorage()
    storage.save_components_efficiently(component_graphs, builder.non_isa_triples)
    
    print("Graph construction and storage completed!")
    print(f"Results saved in: {storage.base_dir}")

# 使用示例
if __name__ == "__main__":
    csv_file = "/home/lsz/OneGraph/data/onegraphv2-.csv"  # 替换为你的CSV文件路径
    # main(csv_file)
    
    # 查询示例
    loader = GraphLoader()
    if loader.index:
        stats = loader.get_statistics()
        print("Statistics:", stats)
        
        # 加载特定连通分量（如果有大型连通分量的话）
        graph = loader.load_component(0)
        if graph:
            print(f"Loaded graph with {graph.number_of_nodes()} nodes")