PermWorld：既有模型多层诊断与编码×架构工程开发单元
开始日期：2026-10-09；执行跨到2026-10-10（America/Los_Angeles）。

完成状态：已有模型多层分析及16模型开发预跑均已完成，所有开发模型
固定20k更新。工程、独立激活/CKA复核、完整性和停止审计通过；本轮
执行进程均已结束，未启动正式确认清单。

阅读入口
- completion_reader_report.html / completion_reader_report.txt：本轮总报告，
  含完整分解、任务学习程度及测量口径说明。
- reader_report.html：既有模型多层结果与30张图。
- encoding_development/report.html：16模型开发结果与40张图。
- final_audit.json、final_stop_audit.json：完整性、F17保护及停止证据。
- development_results_review.txt、encoding_development/reader_review.json：
  独立算术/基线复算、缺失值和科学解释复核。

本轮最直接的新发现：编码改变初始化配对偏好，从而大幅改变Delta。
以答案控制的GRU取逆CKA为例，一行/轮换的B0为-0.6169/+0.8010，
Bt为+0.0675/+0.0793，Delta为+0.6844/-0.7217；Delta翻转没有对应
训练后偏好翻转。轮换编码八个模型仅比训练集定义的长度众数高
0.27--1.78个百分点，编码间任务掌握程度不匹配，不能将结果解释为
同等学习程度下的编码机制。单开发世界及两架构参数量不同的限制保留。

本目录独立于既有F17论文、实验和发布文件。体积大的新缓存、开发数据
和权重放在F盘，路径见storage.json和下面的开发根目录。

范围与停止边界
1. 复用既有82检查点与固定输入，补充各层和一种固定pooling。
2. 完成四性质×两编码×两架构=16模型的工程开发单元，固定20k更新。
3. 报告后停止。开发结果不计正式确认；不自动运行旧正式清单、追加
   种子、GroupWorld、有限域/矩阵、零空间、GELU组合或更大模型。

已有模型分析
- run.py、input_geometry.py、summarize.py及所有输入/源依赖受freeze.json
  约束。兼容旧提取版本和必要数值修复分别保留在compatible_extraction_
  freezes.json、engineering_repairs.json及*_before_*快照。
- 输入、轨道、长度10..30、每长度1/21权重、错误变换、三种主指标
  （线性CKA、k10邻域重叠、留出Procrustes）均沿用旧协议。debiased
  CKA为固定敏感性。最终层、查询、输出概率、真实答案原值直接复用。
- 所有旧初始权重均为grade C：按原构造程序重建，原始step-0身份
  未验证。新检查不将它们升级为精确匹配。
- 原始block residual与final_norm分开；task前输入末端为主位置。
  数值token的contextual mean是唯一pooling敏感性。固定长度下
  embedding_end和embedding_value_mean是常量，结果明确undefined。
- position_numeric_onehot为静态等距输入参照，不构造初始化或Delta。
  position_numeric_embedding使用各检查点真实token embedding。
- 62同种子模型对，共187488逐长度单元，含静态/未定义基线，不能
  视为独立重复。训练种子变异含初始化及优化随机性；不构成新世界CI。
  review_notes.txt纠正冻结摘要模板中较窄的“初始化差异”表述。

只读复算已有层指标（不训练）：
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python diagnostics/permworld_layers_20261009/run.py validate
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python diagnostics/permworld_layers_20261009/run.py evaluate
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python diagnostics/permworld_layers_20261009/summarize.py
已有缓存验证后跳过提取/已完成配对。完整交付包括layer_metric_results.csv、
各层/长度/关系摘要、leave-one敏感性表、图、report.html和brief_report.txt。

编码×架构开发（单世界、单源单元）
代码：experiments/permworld_encoding_archdev_20261009/
大产物：/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev/
小产物与报告：本目录encoding_development/（已由冻结encoding_summary.py导出）。
性质：descents、recoils、peaks、valleys。编码：one_line、canonical_cycle。
Transformer=256维4层8头；GRU=256维4层，参数量不同，不是容量匹配实验。
真实step-0在各架构内共享，跨架构不声称参数相同。GRU参数保留逐值
相等的键迁移，显式dropout修复后台RNG恢复问题，审计和原始权重均保留。

每一步为显式命令；以下不是自动执行的正式训练计划：
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage cpu-check
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage overfit
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage resume
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage train
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage evaluate
  .venv/bin/python -m experiments.permworld_encoding_archdev_20261009 --stage report
  .venv/bin/python diagnostics/permworld_layers_20261009/encoding_summary.py
train命令只访问独立开发manifest，且要求前置工程门槛通过；不调用正式清单。
开发只按固定20k终点取值，不以几何效果方向通过，不延长弱模型预算。
本次执行另用parallel_tail_driver.py运行固定清单末尾最多8个独立模型，
缩短等待；串行主驱动验证并跳过已完成模型，各模型仍只接受一次20k预算。
调度只看清单位置及完成状态，不读取效果决定条件。tail和middle分配
互不重叠的任务，各自JSON/log记录运行时序；同一worker不能启动两次。
parallel_orchestration_manifest.json及v1脚本快照记录增加第二worker时序；
没有修改冻结训练函数、超参数、预算或16模型清单。
另由parallel_reserved_driver.py提前执行固定清单索引8（GRU一行下降数），
主驱动当时仍在索引3（索引从0起）。middle后续跳过完成的同一模型，避免串行收尾等待；
新增的是执行进程，不是模型或科学条件。reserved审计、旧调度快照和清单
修订时序均保留，所有完成模型仍只有一次固定20k更新。
middle因固定位置门槛结束后，parallel_recoils_driver.py接续原清单索引9
（GRU一行逆下降数）；开始前核实middle结束及该模型从未启动，使用其
空出的执行位置。仍只调整原清单执行顺序，审计和调度快照完整保留。

共同读数
B0=初始(正确-错误)，Bt=训练后(正确-错误)，dc=正确CKA变化，
dw=错误CKA变化，Delta=Bt-B0=dc-dw。编码交互定义为canonical_cycle
减one_line，分别报告Gamma0/Gammat/Gamma_correct/Gamma_wrong/GammaDelta。
正Delta不能单独解释为正确表征趋同；隐藏Delta减输出Delta不能分解机制。
源任务准确率单独报告，工程门槛不能代替任务内容学习证据。

审计
final_audit.py只核对已完成产物，不训练。final_audit.json包含唯一cell
覆盖、分解算术、继承结果逐项一致、冻结版本和F17保护文件哈希。
review_notes.txt和工程修复日志保留发现问题与修复次序，开发不包装为确认。

开发代码的只读审查见development_code_review.txt：协议中“all32”是沿用
文字，实际固定清单为16模型。output_prob仅在合法数字答案0..30内重新
归一化，不是完整词表分布。测量状态在task token之前并受因果约束；
实际完整前向仍带task/等号suffix，另有prefix-only独立数值检查。
答案分层分别使用fit/test真值组均值，是测量控制而非可部署的预测步骤。

完成产物的只读汇总（不训练、不改变原评价数据）：
  .venv/bin/python diagnostics/permworld_layers_20261009/overview.py
  .venv/bin/python diagnostics/permworld_layers_20261009/annotate_completion_report.py
overview.py、encoding_summary.py在完整结果出现前冻结，分别受
overview_freeze.json和encoding_exporter_freeze.json约束。读者报告仅增加
口径说明，原始报告及表格保持逐字节一致，见completion_annotation_audit.json。
汇总前发现overview.py的两处显示格式整理不匹配冻结哈希；已仅撤销
这些修改、恢复精确冻结字节并重新生成。原始冻结记录没有重定，错误
生成的报告与代码保存于before_freeze_restore/，issue.json记录完整次序。
训练数据、真实step-0、完整检查点和大激活缓存保留F盘；本地诊断目录
保留代码、协议、小产物、表格和图，不复制这些大文件占用固态空间。
