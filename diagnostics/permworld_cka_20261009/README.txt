PermWorld CKA：已有检查点探索性诊断（2026-10-09）

范围：单性质八关系、三既有种子及其九种有序配对；Property32五个任务池划分、k=1/2/4/8/16的50既有模型。禁止新训练。本目录不修改F17论文/结果/发布产物或原CKA分析。

先读 report.html。完整数据在CSV；图同时提供PNG和PDF。

环境与数据：
  /home/yangx/ICML @ 1073dcb3be4db90ec23476bab8f503c9e00f364b
  /home/yangx/neurips @ 74f0de2017115f06f11b7285366b66e7e8431d30
  external/neurips/src 为可审计上游代码，既有检查点和Property32缓存在/home/yangx/neurips。
  既有specialist数据/缓存位于results/specialist_cka_controls。
  既有逐长度普通/输入调整数值位于results/specialist_input_adjustment。
  模型、配置、缓存、数据与本目录代码的SHA见各审计JSON与deliverable_manifest.json。
  原始检查点/激活缓存必须存在才能完全重现；不允许重新训练补齐缺失。

从工作区根目录按顺序执行（只有构造初始化与前向，没有优化器或训练）：
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/inventory.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/audit_history.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/analyze_relations.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/analyze_k.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/verify.py
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostics/permworld_cka_20261009/report.py
  完成后停止。无需训练任何模型。

主要交付：
  protocol.json：在新指标计算前固定长度权重、输入配对与分析范围。
  model_inventory.csv / initialization_audit.json：98逻辑模型、94唯一检查点的初始身份、配置、权重哈希与缺失产物。
  initialization_history.json / vocabulary_audit.json：历史分析代码版本、词表与RNG初始化路径；不能替代历史初始权重证据。
  artifact_inventory.json / cache_validation.json：数据与版本来源、146项缓存前向取点检查。
  relation_length_cka.csv：八关系×九种子配对×21长度×两控制×三指标的9,072行原始CKA和Δ。
  relation_seed_summary.csv / relation_summary.csv：固定长度加权、均值、中位数、种子范围和正长度数。
  macro_summary.csv / transform_family_summary.csv：固定八关系的描述性汇总，不增加独立关系数。
  leave_one_relation.csv / leave_one_length.csv：全关系与长度敏感性。
  answer_output_length_baselines.csv / input_length_baselines.csv：答案、概率输出及预定输入核基线。
  permutation_length_calibration.csv / permutation_summary.csv / permutation_delta.npz：原32次条件置换规则的逐长度版本，只作描述性校准。
  input_bootstrap_delta.npz / resampling_diagnostics.csv：共同orbit-anchor重采样分布。有限样本偏移较大，百分位不能当作无偏总体置信区间。
  relation_dependencies.json：共享数据/轨道/变换类型的依赖结构。
  k_length_cka.csv / k_model_self_initial_length.csv / k_summary.csv：任务数系列两池间与各自初始化的逐长度表。
  k_pooled_control.csv / k_shared_initialization_audit.csv / k_length_input_audit.csv：合并长度、共享初始化与重复输入审计。
  k_initial_reference_seed*.npy / k_probe_inputs.npz：复用或补充前向的初始化参照与原固定探针输入。
  k_drift.png/pdf：用户要求的两条并列曲线。
  verification.json / protected_before.json：独立数值验证与受保护文件SHA对照。
  completion.json：任务完成且停止，无新训练。

统计解释：
  所有结果都是探索性，不宣称确认性检验。3种子或9种子配对不是新的独立关系。
  原始初始化权重身份全部不可直接验证；当前结果相对于按原程序重建的参照。
  R0/R1/R2是三个既有源种子；R3/R4是固定seed17任务子集敏感性，语料与探针仍共享。
  主分析每个长度单独算CKA后按固定权重平均；绝不以去长度均值后合并CKA替代。
  真值答案残差为常量时CKA未定义，保留空值，不填0。
  U-centered CKA允许负值；Δ是两个对比的差，允许超过1。
  置信度控制仅复用已有探索结果，不新增回归模型、核或任务筛选。
  不显著不代表没有学习；CKA不能单独认定答案独立信息、关系编码或数学计算机制。

本目录中的文件可重新生成；原有实验与F17产物禁止改动。没有自动发布或修改GitHub release。
