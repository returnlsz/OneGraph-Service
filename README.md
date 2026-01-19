# OneGraph Service Workflow

此仓库用于运行OneGraph-V2增强LLMs在CEval上的实验。

## Enviroments

    conda create -n onegraph python=3.10
    pip install -r requirements.txt
    
## Datasets

CEval: https://huggingface.co/datasets/ceval/ceval-exam

## Workflow

### Reason within CEval

    python src/workflow/service-reason.py \
    --input-path "/path/to/input" \
    --output-path "/path/to/output" \
    --max-triples 10 \
    --batch-size 100 \
    --max-concurrent 10 \
    --api-base-url "https://api.key77qiqi.com/v1" \
    --api-keys "key1" "key2" "key3" \
    --api-model "gpt-4o" \

变量说明:

- input-path为enrich后的数据地址
- output-path为LLM response保存地址
- 每个样本retrieve和enrich三元组数量,例如设置为10则包含10个retrieve和10个enrich三元组，共20个三元组
- batch-size批处理大小
- max-concurrent调用LLM的最大并发数
- api-base-url, api-keys, api-model为api信息

### Evaluation

运行前请自行修改ceval_eval_all.py中的main_dir变量，设置为``Reason within CEval``中的``output-path``.

    python src/utils/ceval_eval_all.py