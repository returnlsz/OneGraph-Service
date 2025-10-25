# OneGraph Service Workflow

## Datasets
请提前下载好CEval数据集

## Workflow
Input: 
Every User Query in CEval Datasets

### Stage: Prepare for Retrieve
Step 1: select sub-domians to retrieve
Step 2: retrieve triples (using question embedding/ topic entity embedding and triple embeddings)
以下脚本完成了Retrieve的功能:

    python src/workflow/service-retrieve.py \
        --api_base_url "https://api.key77qiqi.cn/v1" \
        --api_keys "key1" "key2" "key3" \
        --api_model "gpt-4o-mini-2024-07-18" \
        --ceval_data_path "/disk0/lsz/datasets/ceval/ceval-exam" \
        --output_path "/disk0/lsz/datasets/ceval/ceval-exam-added-triples" \
        --triple_data_path "/disk0/lsz/OneGraph/sorted_data_v2" \
        --model_path "/disk0/lsz/PLMs/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
        --other_data_path "/home/lsz/OneGraph/sorted_data_v1/other_all.csv" \
        --gpu_ids "0,1,2,3" \
        --max_workers 4 \
        --batch_size 10 \
        --num_embedding_workers 4 \
        --top_k 100 \
        --log_level INFO

参数:
api_keys可以填一个或者多个api_keys
ceval_data_path为CEval数据集的路径
output_path为处理后数据集的输出路径
triple_data_path为onegraph分类数据
model_path为使用的embedding模型路径
other_data_path针对other类型数据的路径
gpu_ids使用的gpu_id
max_workers,num_embedding_workers和batch_size为并发数和批处理数
top_k为每个sample检索回来的三元组数量

### Stage: Enrich the Retrieved Triples
Step 3: enrich the retrieved triples
以下脚本完成Enrich功能:

    python src/workflow/service-enrich.py \
        --input_dir "/disk0/lsz/datasets/ceval/ceval-exam-added-triples" \
        --output_dir "/disk0/lsz/datasets/ceval/ceval-exam-enrich-triples" \
        --base_url "https://api.key77qiqi.cn/v1" \
        --api_keys "key1" "key2" \
        --model "gpt-4o-mini-2024-07-18" \
        --mode parallel \
        --max_concurrent 100 \
        --log_level DEBUG

参数:
-i为输入路径,例如:/disk0/lsz/datasets/ceval/ceval-exam-added-triples
-o为输出路径,例如:/disk0/lsz/datasets/ceval/ceval-exam-enrich-triples
-m模式:请选择并行处理parallel,-c为对应的并发数量

### Stage: Reasoning (Not Implemented)
Step 4: transfer triple format to specific format
Step 5: reasoning

## Project Structure
OneGraph-Service
|-sorted_data_v1
|--other
|-sorted_data_v2
|--agriculture
|--engineering_technology
|--humanities
|--medicine_health
|--natural_sciences
|--social_sciences
|-src
|--llm
|---llm_client.py
|--workflow
|---service-enrich.py
|---service-retrieve.py
|--utils