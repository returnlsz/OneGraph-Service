import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
data_str = """dataset,code_prediction,edgetable_prediction,naturallanguage_prediction,nodesequence_prediction,origin_prediction,syntaxtree_prediction
accountant,0.8958837772397095,0.8910411622276029,0.8856447688564477,0.9044117647058824,0.8813559322033898,0.9004854368932039
advanced_mathematics,0.9,0.9107142857142857,0.9226190476190477,0.9047619047619048,0.9254658385093167,0.8875739644970414
art_studies,0.8993288590604027,0.8926174496644296,0.8993288590604027,0.9093959731543624,0.9228187919463087,0.9026845637583892
basic_medicine,0.9142857142857143,0.92,0.9257142857142857,0.9257142857142857,0.9085714285714286,0.9257142857142857
business_administration,0.8561872909698997,0.8433333333333334,0.8471760797342193,0.8471760797342193,0.8471760797342193,0.8433333333333334
chinese_language_and_literature,0.8708133971291866,0.861244019138756,0.8708133971291866,0.8899521531100478,0.8755980861244019,0.8660287081339713
civil_servant,0.8568075117370892,0.8578199052132701,0.8254716981132075,0.8601895734597157,0.8676122931442081,0.8568075117370892
clinical_medicine,0.925,0.93,0.925,0.94,0.935,0.935
college_chemistry,0.8513513513513513,0.8552036199095022,0.8526785714285714,0.8571428571428571,0.875,0.8558558558558559
college_economics,0.8873239436619719,0.8913480885311871,0.8873239436619719,0.8812877263581489,0.8951612903225806,0.8873239436619719
college_physics,0.88,0.9034090909090909,0.8863636363636364,0.8857142857142857,0.9085714285714286,0.8693181818181818
college_programming,0.9178885630498533,0.9117647058823529,0.9152046783625731,0.9120234604105572,0.887905604719764,0.9035087719298246
computer_architecture,0.875,0.8393782383419689,0.875,0.875,0.890625,0.8697916666666666
computer_network,0.9368421052631579,0.8555555555555555,0.8602150537634409,0.9534883720930233,0.9042553191489362,0.875
discrete_mathematics,0.9144736842105263,0.8881578947368421,0.9013157894736842,0.8758169934640523,0.8609271523178808,0.8666666666666667
education_science,0.8518518518518519,0.8407407407407408,0.825925925925926,0.8555555555555555,0.8555555555555555,0.8407407407407408
electrical_engineer,0.7581120943952803,0.742603550295858,0.7337278106508875,0.7315634218289085,0.6834319526627219,0.7315634218289085
environmental_impact_assessment_engineer,0.8362989323843416,0.8291814946619217,0.8327402135231317,0.8428571428571429,0.8398576512455516,0.8434163701067615
fire_engineer,0.8267148014440433,0.8447653429602888,0.8339350180505415,0.8158844765342961,0.7870036101083032,0.8514492753623188
high_school_biology,0.9540229885057471,0.9710982658959537,0.9653179190751445,0.9655172413793104,0.9653179190751445,0.953757225433526
high_school_chemistry,0.9302325581395349,0.9473684210526315,0.9529411764705882,0.9590643274853801,0.9764705882352941,0.9467455621301775
high_school_chinese,0.6629213483146067,0.702247191011236,0.6685393258426966,0.6741573033707865,0.6966292134831461,0.6179775280898876
high_school_geography,0.9213483146067416,0.8932584269662921,0.9157303370786517,0.9269662921348315,0.9157303370786517,0.9101123595505618
high_school_history,0.8461538461538461,0.8626373626373627,0.8351648351648352,0.8351648351648352,0.8626373626373627,0.8406593406593407
high_school_mathematics,0.86875,0.9012345679012346,0.9,0.8867924528301887,0.9,0.9
high_school_physics,0.9314285714285714,0.9257142857142857,0.9485714285714286,0.9314285714285714,0.9485714285714286,0.9371428571428572
high_school_politics,0.9090909090909091,0.9147727272727273,0.8857142857142857,0.9147727272727273,0.8920454545454546,0.9257142857142857
ideological_and_moral_cultivation,0.9069767441860465,0.8953488372093024,0.8953488372093024,0.9244186046511628,0.9127906976744186,0.9011627906976745
law,0.8727272727272727,0.7962962962962963,0.8504672897196262,0.8796296296296297,0.8703703703703703,0.8545454545454545
legal_professional,0.7753623188405797,0.8014184397163121,0.7956204379562044,0.7697841726618705,0.7956204379562044,0.8088235294117647
logic,0.8712871287128713,0.8382352941176471,0.8284313725490197,0.8529411764705882,0.8768472906403941,0.8507462686567164
mao_zedong_thought,0.9541284403669725,0.944954128440367,0.9770642201834863,0.9493087557603687,0.9678899082568807,0.9541284403669725
marxism,0.9217877094972067,0.9497206703910615,0.9553072625698324,0.9106145251396648,0.9329608938547486,0.9273743016759777
metrology_engineer,0.8359788359788359,0.8691099476439791,0.8429319371727748,0.8835978835978836,0.8808290155440415,0.8526315789473684
middle_school_biology,0.96875,0.9479166666666666,0.96875,0.9739583333333334,0.9791666666666666,0.9583333333333334
middle_school_chemistry,0.9675675675675676,0.9891891891891892,0.9783783783783784,0.9945945945945946,0.9837837837837838,0.9783783783783784
middle_school_geography,0.9814814814814815,0.9814814814814815,0.9722222222222222,0.9722222222222222,0.9444444444444444,0.9814814814814815
middle_school_history,0.9565217391304348,0.9371980676328503,0.9371980676328503,0.927536231884058,0.9371980676328503,0.9323671497584541
middle_school_mathematics,0.943502824858757,0.9485714285714286,0.9488636363636364,0.9367816091954023,0.9542857142857143,0.9491525423728814
middle_school_physics,0.9808917197452229,0.9745222929936306,0.9554140127388535,0.9615384615384616,0.9872611464968153,0.9746835443037974
middle_school_politics,0.9528795811518325,0.9270833333333334,0.9421052631578948,0.9479166666666666,0.93717277486911,0.9319371727748691
modern_chinese_history,0.9047619047619048,0.909952606635071,0.919431279620853,0.9004739336492891,0.919431279620853,0.9142857142857143
operating_system,0.8491620111731844,0.8435754189944135,0.8379888268156425,0.8324022346368715,0.8324022346368715,0.8435754189944135
physician,0.9119638826185101,0.9187358916478555,0.9142212189616253,0.9300225733634312,0.9164785553047404,0.8939051918735892
plant_protection,0.8793969849246231,0.8994974874371859,0.8844221105527639,0.8994974874371859,0.8894472361809045,0.864321608040201
probability_and_statistics,0.9156626506024096,0.896969696969697,0.8855421686746988,0.891566265060241,0.9197530864197531,0.8855421686746988
professional_tour_guide,0.9135338345864662,0.9135338345864662,0.9318181818181818,0.9097744360902256,0.9132075471698113,0.9283018867924528
sports_science,0.8066666666666666,0.7866666666666666,0.8243243243243243,0.803921568627451,0.8367346938775511,0.8431372549019608
tax_accountant,0.8893905191873589,0.9164785553047404,0.8871331828442438,0.8826185101580135,0.916289592760181,0.9072398190045249
teacher_qualification,0.9649122807017544,0.9598997493734336,0.9649122807017544,0.949874686716792,0.9573934837092731,0.9674185463659147
urban_and_rural_planner,0.8086124401913876,0.7942583732057417,0.8110047846889952,0.8086124401913876,0.7942583732057417,0.8110047846889952
veterinary_medicine,0.9333333333333333,0.9523809523809523,0.9476190476190476,0.9523809523809523,0.9476190476190476,0.9476190476190476"""

# 读取数据
df = pd.read_csv(StringIO(data_str))

# 预测方法列表
predictions = ['code_prediction', 'edgetable_prediction', 'naturallanguage_prediction', 
               'nodesequence_prediction', 'origin_prediction', 'syntaxtree_prediction']

# 计算每个数据集所有方法的平均准确率
df['mean_accuracy'] = df[predictions].mean(axis=1)

# 按平均准确率从高到低排序
df_sorted = df.sort_values('mean_accuracy', ascending=False).reset_index(drop=True)

# ============ 生成大折线图 ============
fig, ax = plt.subplots(figsize=(20, 8))

# 定义颜色和线型
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
markers = ['o', 's', '^', 'D', 'v', '*']
linestyles = ['-', '--', '-.', ':', '-', '--']

# 绘制每个预测方法的折线
for idx, pred in enumerate(predictions):
    ax.plot(range(len(df_sorted)), df_sorted[pred].values, 
            color=colors[idx], marker=markers[idx], linestyle=linestyles[idx],
            linewidth=2.5, markersize=6, label=pred, alpha=0.85)

# 设置x轴标签
ax.set_xticks(range(len(df_sorted)))
ax.set_xticklabels(df_sorted['dataset'].values, rotation=90, fontsize=9)

# 设置标签和标题（英文）
ax.set_xlabel('Dataset (Sorted by Average Accuracy - High to Low)', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Prediction Method Accuracy Comparison Across Datasets\n(Sorted from High to Low)', 
             fontsize=15, fontweight='bold', pad=20)

# 设置y轴范围
ax.set_ylim([0.6, 1.0])

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--')

# 添加图例
ax.legend(loc='lower left', fontsize=11, framealpha=0.95, ncol=2)

# 调整布局
plt.tight_layout()
plt.savefig('prediction_lineplot_sorted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: prediction_lineplot_sorted.png")
plt.close()

# ============ 生成分组折线图（每组10个数据集）============
num_per_group = 10
num_groups = (len(df_sorted) + num_per_group - 1) // num_per_group

for group_idx in range(num_groups):
    fig, ax = plt.subplots(figsize=(16, 7))
    
    start_idx = group_idx * num_per_group
    end_idx = min(start_idx + num_per_group, len(df_sorted))
    df_group = df_sorted.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # 绘制每个预测方法的折线
    for idx, pred in enumerate(predictions):
        ax.plot(range(len(df_group)), df_group[pred].values, 
                color=colors[idx], marker=markers[idx], linestyle=linestyles[idx],
                linewidth=2.5, markersize=8, label=pred, alpha=0.85)
    
    # 设置x轴标签
    ax.set_xticks(range(len(df_group)))
    ax.set_xticklabels(df_group['dataset'].values, rotation=45, ha='right', fontsize=11)
    
    # 设置标签和标题（英文）
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Prediction Method Accuracy Comparison - Group {group_idx+1} (Rank {start_idx+1}-{end_idx})', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # 设置y轴范围
    ax.set_ylim([0.6, 1.0])
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'prediction_lineplot_group_{group_idx+1}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: prediction_lineplot_group_{group_idx+1}.png")
    plt.close()

# ============ 生成排序统计表 ============
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('tight')
ax.axis('off')

# 准备表格数据（前20个）（英文）
table_data = []
table_data.append(['Rank', 'Dataset', 'Avg Accuracy', 'Code', 'EdgeTable', 'NatLanguage', 'NodeSeq', 'Origin', 'SyntaxTree'])

for i in range(min(20, len(df_sorted))):
    row = [str(i+1), df_sorted.iloc[i]['dataset'], f"{df_sorted.iloc[i]['mean_accuracy']:.4f}"]
    for pred in predictions:
        row.append(f"{df_sorted.iloc[i][pred]:.4f}")
    table_data.append(row)

# 创建表格
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.05, 0.25, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# 设置表头样式
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置交替行颜色
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#F2F2F2')

plt.title('Dataset Accuracy Ranking Table (Top 20)', fontsize=14, fontweight='bold', pad=20)
plt.savefig('ranking_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ranking_table.png")
plt.close()

# ============ 保存排序数据 ============
output_df = df_sorted[['dataset', 'mean_accuracy'] + predictions].copy()
output_df.insert(0, 'rank', range(1, len(output_df)+1))
output_df.to_csv('prediction_sorted.csv', index=False, encoding='utf-8-sig')
print("✓ Saved: prediction_sorted.csv")

print("\n" + "="*70)
print("✅ All visualizations completed! Generated files:")
print("  📊 prediction_lineplot_sorted.png - Complete line chart (sorted high to low)")
print(f"  📊 prediction_lineplot_group_*.png - Grouped line charts ({num_groups} groups)")
print("  📊 ranking_table.png - Ranking table (Top 20)")
print("  📄 prediction_sorted.csv - Sorted data")
print("="*70)
