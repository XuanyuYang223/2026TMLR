从 /home/yangx/ICML 执行；PYTHONDONTWRITEBYTECODE=1 防止修改旧模块的缓存。

已完成的准备验证（不启动正式训练）：
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.validate --output results/permworld_encoding_plan_20261009/encoding_validation.json
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.checks --config results/permworld_encoding_plan_20261009/planning_candidate.json --output results/permworld_encoding_plan_20261009/integration_checks.json

未来正式执行（本轮没有执行以下命令；不能把微基准当作正式训练）：
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.data --config results/permworld_encoding_plan_20261009/protocol.json
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.train --config results/permworld_encoding_plan_20261009/protocol.json --run-id u001_one_line_descents
  按training_manifest.csv中的run_id串行执行全部模型；实际字母顺序见清单，不按成绩重排/删减。
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.evaluate --config results/permworld_encoding_plan_20261009/protocol.json --unit u001
  所有单元预定模型完成后，每单元显式评估一次，然后：
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.evaluate --config results/permworld_encoding_plan_20261009/protocol.json --summarize

正式数据/step0/训练检查点仅写results/permworld_encoding_confirmation_v1，不写F17或旧诊断目录。模型若中断，用完全相同命令续跑；协议/数据/代码哈希不匹配会拒绝续跑。正式协议不可覆盖；修改须建立显式新版本并重新冻结。
主要结果：unit_components.csv、encoding_interactions.csv、primary_statistics.csv；附带baseline_unit_components.csv、baseline_encoding_interactions.csv、task_accuracy_summary.csv，以及逐长度各表。
缺失/未定义单元不悄悄剔除，不自动补模型或增加样本。表征指标分量不是因果贡献比例；不以隐藏Delta减概率Delta定义机制。
