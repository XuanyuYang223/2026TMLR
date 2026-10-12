"""CPU-only Chinese reading companion to the frozen validation report."""
import csv
import html
import json
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_learnability')


def main():
    models = list(csv.DictReader((D / 'model_learning_summary.csv').open()))
    pairs = list(csv.DictReader((D / 'pair_learning_gates.csv').open()))
    protocol = json.loads((D / 'protocol.json').read_text())
    provenance = json.loads((D / 'completion_audit.json').read_text())
    independent = json.loads((D / 'independent_learning_audit.json').read_text())
    assert len(models) == 16 and len(pairs) == 8
    assert all(int(r['actual_step']) == 8000 and r['validation_complete'] == 'True' for r in models)
    assert provenance['status'] == independent['status'] == 'passed'
    model_pass = sum(r['gate_pass'] == 'True' for r in models)
    pair_pass = sum(r['gate_pass'] == 'True' for r in pairs)
    lookup = {(r['architecture'], r['encoding'], r['task']): r for r in models}
    columns = [('transformer', 'one_line'), ('transformer', 'canonical_cycle'),
               ('gru', 'one_line'), ('gru', 'canonical_cycle')]
    tasks = dict(descents='下降数', recoils='逆下降数', peaks='峰数', valleys='谷数')
    title = 'PermWorld 可学习性开发：全部完成，共同学习资格未通过'
    paragraphs = [
        f'本轮在已有模型诊断后运行一个固定开发方案：长度10、11、12，四任务、两编码、两架构，共16模型；每模型实际完成8000更新，验证每长度1024个排列。GPU阶段实际墙钟为{provenance["execution_wall_seconds"]/60:.2f}分钟，包含工程恢复、验证和保存开销，低于120分钟上限；不是GPU核函数利用率计时。',
        f'学习资格为{model_pass}/16，编码任务对联合资格为{pair_pass}/8。完整训练与资格未通过是不同状态：本轮不是缺失产物或预算中断，而是未找到两编码都充分学习且成绩相近的设置。因此没有生成独立几何确认清单，没有启动新CKA、GroupWorld或更多训练。',
        '所有表内准确率都是恒等输入验证集上的全词表答案argmax；各长度分别计算后按1/3汇总。长度众数由训练标签固定，未使用验证众数选择。EOS为真实答案条件下的teacher-forced指标，本轮为99.97%—100%，不计为答案任务掌握。',
    ]
    header = ['任务', '长度基线', 'Transformer 一行', 'Transformer 轮换', 'GRU 一行', 'GRU 轮换']
    table = []
    for task in protocol['tasks']:
        base = lookup[('transformer', 'one_line', task)]
        table.append([tasks[task], f'{100*float(base["equal_length_baseline"]):.2f}%'] +
            [f'{100*float(lookup[(a,e,task)]["equal_length_accuracy"]):.2f}%' for a,e in columns])
    paragraphs_after = [
        '通过单模型资格的是：Transformer一行记号的逆下降数、峰数，以及GRU一行记号的四个任务。轮换记号没有模型通过。均值准确率要求≥90%、各长度≥85%、超过长度众数≥20pp、EOS≥99%；同架构同任务还要求两编码均值与最大逐长度差均≤5pp。该开发门槛不是TOST，也不证明统计等价。',
        '曲线显示两类失败线索：Transformer轮换记号的验证准确率和答案CE多数停在基线附近；GRU轮换记号的末批训练combined CE很低，但验证答案CE升至4.09—5.22。同口径训练末批combined CE与验证combined CE的分离与拟合而未泛化相容，不能把单批loss当成全训练准确率，也不能据此确认具体记忆机制。',
        '一行记号也有未充分学习的条件：Transformer下降数56.67%、谷数80.99%。因此不能把全部学习不足归因于轮换编码，不能将当前几何差异解释为仅由编码改变了数学关系表征。',
        '两编码在同架构内共用底层排列、数学标签、批次、真实保存的step-0和更新预算。新输入按完整八元轨道排除历史及上一轮开发输入；训练、验证、预留测试轨道分离。测试排列只用于数据隔离审计，训练/选择/报告均未读测试标签或计算测试成绩。',
        '本轮只改变一个开发方案，但同时缩小长度窗口、改变训练样本量、学习率、dropout、microbatch与更新数，不能把改善归因于单个因素。Transformer约324万参数，GRU约166万参数，架构间不等参数量。只有一个新数据世界和每架构一个初始化，不是16个独立世界，也不构成新的确认实验。',
        f'来源与工程审计通过：冻结{provenance["frozen_source_files"]}个源码文件；真实CUDA连续32更新与16后恢复至32更新在四个架构×编码条件下参数/优化器差均为0且RNG相同；{provenance["protected_legacy_files"]}个受保护旧产物哈希未变，F17及原多层报告保持原样。损失数值与报告schema的两处CPU工程修复均在新训练前记录并重新冻结；数据、模型、训练目标和真实初始权重未改。',
        '独立算术审查从保存的逐长度统计复算基线、学习门槛及编码差，属于结果聚合审查，不是重做模型前向。所有训练子进程已结束。当前结果支持“这套开发配置的共同可学习性尚未达到”，不能推出轮换编码普遍不可学习。',
    ]
    links = [
        ('完整冻结报告', 'learnability_report.html'),
        ('16模型终点及全部门槛', 'model_learning_summary.csv'),
        ('逐长度、逐阶段数据', 'validation_length.csv'),
        ('答案与损失曲线数据', 'validation_curves.csv'),
        ('8个配对资格', 'pair_learning_gates.csv'),
        ('全部答案准确率曲线', 'learnability_plots/equal_length_accuracy.png'),
        ('全部答案CE曲线', 'learnability_plots/equal_length_answer_ce.png'),
        ('已有模型学习诊断', 'old_learning_diagnosis.txt'),
        ('完整最终曲线审查', 'learning_curve_review.txt'),
        ('数据与初始化身份审查', 'data_identity_review.txt'),
        ('独立结果审查', 'independent_learning_audit.json'),
        ('保护、检查点、预算与停止审计', 'completion_audit.json'),
        ('冻结协议', 'protocol.json'),
        ('源码冻结与工程修复记录', 'freeze.json'),
    ]
    text = title + '\n\n' + '\n\n'.join(paragraphs) + '\n\n'
    text += '\t'.join(header) + '\n' + '\n'.join('\t'.join(r) for r in table)
    text += '\n\n' + '\n\n'.join(paragraphs_after)
    text += '\n\n产物入口：\n' + '\n'.join(f'{name}: {path}' for name,path in links)
    text += f'\n\n大产物（权重、数据、真实初始化、日志）：{ROOT}\n'
    (D / 'completion_reader_report.txt').write_text(text)
    body = ''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
    body += '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in header)+'</tr></thead><tbody>'
    body += ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>' for r in table)+'</tbody></table>'
    body += ''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs_after)
    body += '<p>产物入口：</p><ul>'+''.join('<li><a href="'+html.escape(path)+'">'+html.escape(name)+'</a></li>' for name,path in links)+'</ul>'
    body += '<p>大产物：'+html.escape(str(ROOT))+'</p>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+html.escape(title)+'</title>'
    document += '<style>body{max-width:1100px;margin:32px auto;font:16px/1.7 system-ui,sans-serif;padding:0 20px;color:#202632}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px 10px;border:1px solid #cdd3dd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef1f6}a{color:#1f5caa}</style>'
    document += '<h1>'+html.escape(title)+'</h1>'+body+'</html>'
    (D / 'completion_reader_report.html').write_text(document)
    print(json.dumps(dict(status='written',models=16,learning_passed=model_pass,pair_passed=pair_pass),ensure_ascii=False))


if __name__ == '__main__':
    main()
