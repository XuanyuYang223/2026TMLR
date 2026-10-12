PermWorld：验证集驱动的可学习性开发，2026-10-10

本轮已完成并停止。16/16模型均完成固定8k更新及最终验证；GPU阶段
实际墙钟1703.47秒（28.39分钟，含工程/验证/保存），低于2h上限。
6/16个模型通过预定学习门槛，0/8编码任务对通过联合门槛；因此没有
找到共同充分学习设置，没有新几何确认、独立确认训练或GroupWorld。
简短中文入口：completion_reader_report.html / .txt。
完整报告：learnability_report.html / .txt；逐长度、曲线、门槛CSV和
8张固定图均已保存。407个受保护旧产物哈希保持一致，F17未修改。
completion_audit.json含源码/真实初始化/检查点/预算/停止审计；
independent_learning_audit.json独立复算报告，不是重做模型前向。

本轮数据生成、CPU检查、CUDA恢复、开发训练和报告命令已全部执行，
不要为了读结果重启develop。只读重建报告可以显式使用--stage report。

本轮承接已完成的多层分析与16模型开发预跑。新增工作先核查已有
训练曲线、逐长度答案准确率和训练长度众数，再运行一个预先固定的
较短长度窗口开发方案。旧CKA/Delta不参与配置、长度或检查点选择。

旧诊断入口：old_learning_diagnosis.txt，以及同名CSV/plots/audit产物。
代码审查：code_readiness_review.txt、independent_code_review.txt。
独立成本建议：protocol_review.txt；其建议不是最终协议，protocol.json
才是冻结方案（例如本轮验证集为每长度1024，均值编码差界为5pp）。

唯一开发方案：n10..12、下降/逆下降/峰/谷、两编码、Transformer/GRU。
小模型仍256维4层（Transformer8头）。新数据、真实保存step0；同架构
任务及编码共享初始参数。更新8k，batch64、microbatch64、LR1e-3、
warmup400、dropout0，其余旧AdamW参数不变。多个训练因素同时改变，
不能把任何提升归因于某一个单独因素。

学习资格在训练前冻结：均值答案准确率≥90%、各长度≥85%、超过由
训练标签定义的长度众数≥20pp、EOS≥99%，两编码同架构同任务的
均值与最大逐长度精度差均≤5pp。部分2k/4k不代替固定8k最终资格。
这些是开发资格门槛，不是统计等价检验或独立数学机制的判据。

训练先2k覆盖全部16条件，然后固定4k、8k；不按效果删任务或挑指标。
GPU阶段总壁钟上限7200秒，其中最后30秒预留收尾。所有来源、初始化、
协议、源版本、实际step、样本曝光与耗时均保留。工程检查含CPU已知
logits/损失/causal独立验证，以及真实CUDA32连续对比16+32独立进程恢复。
成本截止时不再更新，不把缺失最终验证的状态冒充完整模型。

训练/验证仅使用identity任务样本；学习驱动不加载预留test数组、不
评价测试集或CKA。数据生成和划分审计检查所有split的轨道，这是
数据身份/隔离检查，不是测试集选方案。候选与上一轮所有训练、验证、
测试输入所在八元轨道均排除；两编码共用底层对象、标签与批次。

大产物：/mnt/yangx-projects-f/20261010/permworld_learnability/
代码：experiments/permworld_learnability_20261010/
本目录保存冻结协议、源码哈希、报告、CSV及图。F17与上一轮PermWorld
报告/数据/代码只读保护，见protected_legacy_before.json。

各阶段均是显式命令；不要为了读报告运行develop：
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage setup
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage prepare
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage cpu-check
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage develop
  .venv/bin/python -m experiments.permworld_learnability_20261010 --stage report

独立只读验证：
  .venv/bin/python diagnostics/permworld_learnability_20261010/completion_audit.py
  .venv/bin/python diagnostics/permworld_learnability_20261010/independent_learning_audit.py --run-real
  .venv/bin/python diagnostics/permworld_learnability_20261010/completion_reader.py

若共同学习门槛未达到，保留全部负结果并停止，不用新几何结果替代。
若达到，先冻结独立新世界/新初始化验证清单；开发不能当作确认。
本轮没有GroupWorld、F17修改、额外候选、正式几何训练或自动扩容。
