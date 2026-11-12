# EmoMM-MusicRec

当前范围：读取原始数据、提取特征、训练 S/A/P 三分支、TextVA、MusicVA。

## 运行

使用现有 Conda 环境 **msmmr**，解释器为 `D:\Anaconda3\envs\msmmr\python.exe`。
在 IDE 中选择这个解释器后，先运行 `scripts/extract_features.py`，完成后再运行 `scripts/train.py`，两者均支持无参数启动。
脚本以项目目录为基准解析相对路径，不要求 IDE 的工作目录与项目目录相同。

也可以在项目目录执行：

```powershell
conda activate msmmr
python scripts/extract_features.py --check-only
python scripts/extract_features.py
python scripts/train.py --check-only
python scripts/train.py
```

第一次训练前必须先完成特征提取。`--check-only` 不写入特征、权重或运行记录。
默认离线读取已经缓存的预训练底座。若换机器，需要配置原始数据、CLaMP3 权重及 MERT/XLM-R 缓存；需要下载时将 `config/data.yaml` 的 `local_files_only` 改为 `false`。

## 配置与文件位置

| 位置 | 内容 |
| --- | --- |
| `config/data.yaml` | 原始数据、划分来源、预训练底座、音频截取、特征提取参数 |
| `config/train.yaml` | 三项模型参数、损失权重、优化器、epoch、seed |
| `src/emomm/data.py` | 数据集读取、已有划分检查、训练数据加载 |
| `src/emomm/features/mert.py` | 原始音频转为 MERT tokens |
| `src/emomm/features/clamp3.py` | 文本和 MERT tokens 转为 768 维 CLaMP3 特征 |
| `src/emomm/models/` | 三项模型定义 |
| `src/emomm/train/` | 三项训练逻辑及 `losses.py`；包初始化文件提供共享指标和保存工具 |
| `assets/features/clamp3/<feature_id>/` | 七个基础 NPZ，以及同批次新生成的 `pairs_merge_views.npz` |
| `runs/extract/<feature_id>/` | `effective_config.yaml` 和 `summary.json` |
| `runs/train/<run_id>/` | 配置快照、各模型每个 seed 的权重/曲线/指标、`selection.yaml`、运行摘要 |

上述产物目录初始为空，执行时创建具体批次。MERT tokens 在内存中传给 CLaMP3，不另存一份完整中间特征。
长时间音频提取会产生本批次的 `.partial.npy` 断点文件；相应 NPZ 完成后自动移除这些临时文件。
各训练模型均随机初始化；冻结 CLaMP3 对照组不训练。TextVA 与 MusicVA 都直接使用冻结 CLaMP3 特征，分别训练自己的 VA 头，不经过 A 分支。
原始数据、公开预训练 CLaMP3/MERT/XLM-R 底座属于输入；旧项目生成的特征、训练权重、曲库和结果均不作为输入。
目前 CLaMP3 的公开预训练文件通过绝对路径引用旧目录中的 `model_weights/clamp3.pt`，其余项目源码不依赖旧项目。

## 数据与训练路线

| 数据集 | 新特征文件 | 用途与划分 |
| --- | --- | --- |
| EmoBank | `text_emobank.npz` | TextVA；CSV 官方 train/dev/test，dev 映射为 validation |
| DEAM | `music_deam.npz` | MusicVA；读取配置指定的本地 train/validation/test 文件 |
| PMEmo | `music_pmemo.npz` | MusicVA；读取配置指定的本地 train/validation/test 文件 |
| MERGE | `pairs_merge.npz`、`pairs_merge_views.npz` | S 分支；附带的 70/15/15 划分，片段只来自训练集 |
| MTG-Jamendo | `mtg_music.npz`、`mtg_tags.npz` | A/P 分支；官方 split-0，标签支持数仅在训练集计算 |
| Song Describer | `pairs_song_describer.npz` | 关联 MTG 官方划分；当前本地版本全部为 test，训练时不使用 |

DEAM/PMEmo 的本地划分文件来源按 `supplied_local_split_files` 记录，不将其未经确认地称为官方划分。
所有划分原样读取，不重新随机划分。完全相同的音频内容跨划分时，训练加载按 test、validation、train 的优先级保留原始记录，排除较低优先级划分中的重复记录，并在训练配置快照中记录这些排除。原始数据和提取文件不改写。
例如 PMEmo 的 90 在 train、92 在 test，但原始音频完全相同；加载训练集时排除 90，92 仍属于原始 test。若希望遇到冲突直接停止，可将 `duplicate_audio_policy` 改为 `error`。
音频默认取开头、中间、结尾各 30 秒，再按 5 秒提取 MERT tokens；短于 30 秒时使用原音频。
Song Describer 与 MTG 重叠的歌曲复用同一个新提取批次内的音频向量。

- **S/A/P 四组对比**：`clamp3` 直接对冻结 CLaMP3 特征做余弦检索，不训练适应头；`adapter` 训练六个文本/音乐残差 Adapter（768→64→768），S 使用配对多正样本对比损失，A/P 使用双向标签匹配损失；`residual_distill` 使用相同 Adapter、数据、歌词增强、种子、优化器和选模规则，在 S 上增加 0.5 倍冻结 CLaMP3 相似度蒸馏（温度 0.07），输出归一化混合 0.8 原始表示和 0.2 适配后表示；`ap_multiview` 在第三组的基础上，只更换 A/P 的训练提示。A/P 不增加蒸馏。第三组与第二组检验的是“原始表示混合＋蒸馏”的组合效果，第四组与第三组检验标签多表达增强。各分支按验证指标独立选择 epoch，再组装最佳权重。
- **A/P 标签多表达对照**：每个现有标签保留原始提示，再添加 `A piece of ...`、`I would like to listen to ...`、`Please recommend ...` 三种表达，只包装原有音乐短语，不增加同义标签、情绪、乐器、风格或组合条件。每个训练步骤为每个标签均匀随机选一种表达，仍对应原始多标签目标，提示数量和损失权重不扩大。专用随机数生成器与数据加载、Dropout 随机流分开。原始单提示的文本向量逐项原样复用。两组都从相同种子的随机初始化开始，不加载旧训练权重。
- **增强评估口径**：所有四组都使用原始单提示进行主评估和选模；最终最佳权重额外评估未参与训练的 `Find me a track described as ...` 模板，只报告模板改写鲁棒性，不参与选 epoch/seed，也不等同于真实用户查询或未见标签评估。主指标为 A/P 各自的 nDCG@10、mAP，按相同种子报告均值、样本标准差、配对差值和提升种子数。S 的训练方法和两项 VA 的代码/参数保持原方案；由于三分支联合梯度裁剪及早停，两个独立拟合组的 S 权重不保证逐项相同。
- **歌词片段**：仅从 MERGE 训练集原歌词生成，少于 64 词跳过；从最多五个连续窗口中，根据与同批次音频、全文的平均余弦相似度选出最多两个。窗口通常 96 词，较短歌词取 75%。全文与片段保留相同歌曲身份，按多正样本训练；跨划分重复音频的原样本若被排除，其片段一起排除。验证／测试不生成片段。三组 Adapter 训练使用完全相同的片段。
- **TextVA**：直接读取 `text_emobank.npz` 的冻结 CLaMP3 文本向量，训练 `Linear(768,2) + tanh`，损失为 SmoothL1 + CCC。EmoBank 1–5 标签映射为 [-1,1]。不读取、训练或依赖三分支 Adapter。
- **MusicVA**：冻结 CLaMP3 音频特征输入量化回归头，输出两轴的 q05/q50/q95，使用 pinball 损失。DEAM/PMEmo 本地 1–9 标签先统一到 [-1,1]，训练时转为 [0,1]。本阶段按源数据完整 validation 选模，尚不生成校准区间。

默认 seed 为 42、43、44。选模只用验证集：S/A/P 取验证检索指标最大值，TextVA 取平均 CCC 最大值，MusicVA 取各数据源等权 pinball 最小值。当前训练脚本不运行 test 评估。
冻结 CLaMP3 对照不涉及随机训练，只评估一次，不重复三次制造标准差；三组 Adapter 分别使用相同三个种子从头训练，汇总为均值与样本标准差（ddof=1）。四组的检索样本、候选曲库、查询标签和重复音频排除规则相同。四组对比仅针对 S/A/P 检索任务，VA 模型单独训练与报告，冻结编码器自身不能直接输出 VA。
新实现不加载旧数据增强文件。重训结果不承诺复现旧数值。
歌词片段增强和语义保持作为既有训练手段使用，不单独作为项目的方法创新。

## 指定批次、阶段与恢复

```powershell
# 指定特征批次；重复 ID 默认拒绝覆盖。
python scripts/extract_features.py --feature-id fresh_001

# 中断后继续同一批次：验证配置、原始文件状态及已完成文件。
python scripts/extract_features.py --feature-id fresh_001 --resume

# 训练明确指定的新特征批次。
python scripts/train.py --feature-id fresh_001 --run-id train_001

# 可单独执行四组检索对比，或训练任一 VA 模型。
python scripts/train.py --feature-id fresh_001 --stage three_branch --run-id sap_001
python scripts/train.py --feature-id fresh_001 --stage music_va --run-id music_001

# TextVA 已独立，不需要 --three-branch-run。
python scripts/train.py --feature-id fresh_001 --stage text_va --run-id text_001
```

`feature_id: latest` 只选择本项目成功完成且满足所需文件集合的特征批次。训练开始后在配置快照中固定这个 ID 及文件摘要。
每次训练写入新的 `<run_id>`，各模型下保存 `seed_<n>/best.pt`、`history.csv`、`metrics.json`，以及阶段的 `aggregate.json`。
四组对比保存位置：

```text
runs/train/<run_id>/
├─ three_branch/
│  ├─ seed_42/、seed_43/、seed_44/  # residual_distill 组，保留主模型原有路径
│  ├─ aggregate.json
│  ├─ comparison.json             # 四组相同口径的验证指标，均值/样本标准差
│  ├─ ap_prompt_comparison.json   # 原始单提示 vs 多表达，按种子配对的主/辅助指标
│  ├─ ap_prompt_comparison.csv    # 同一份对照表，可直接用 Excel 打开
│  └─ comparison/
│     ├─ clamp3/metrics.json       # 无训练、无 checkpoint、SD 为 null
│     ├─ adapter/
│     │  ├─ seed_42/、seed_43/、seed_44/
│     │  └─ aggregate.json
│     └─ ap_multiview/
│        ├─ seed_42/、seed_43/、seed_44/
│        └─ aggregate.json
├─ text_va/                       # CLaMP3 文本特征直接回归
├─ music_va/
├─ effective_config.yaml
├─ selection.yaml
└─ summary.json
```

`selection.yaml` 记录主模型验证集选出的权重路径、seed、摘要和验证分数。主三分支暂沿用 `residual_distill` 组，组间结果供后续选择方法，不自动拼接不同组的权重。TextVA 权重保存 `input_representation: clamp3_text` 和 CLaMP3 底座摘要，加载器会拒绝旧的 A→VA 权重，防止输入路线混用。
重新训练可复用当前完整特征批次。首次运行 `scripts/train.py` 会自动补提取 732 条标签文本（56 个 A 标签、127 个 P 标签，每个标签 3 条新训练表达＋1 条未参与训练的评估表达），不提取音频，也不启动 MERT。新批次按实际标签数计算。补充特征存于 `assets/features/clamp3/<feature_id>/mtg_tag_prompt_views_<signature>.npz`，对应文字、来源及哈希记录存于 `runs/extract/<feature_id>/mtg_tag_prompt_views_<signature>.json`；下次验证哈希后复用。原有特征及已完成提取摘要不改写。

直接运行 `scripts/train.py` 会训练三组三分支和两项 VA，每组均为三个种子，较上次多训练一组三分支。若这次只比较 A/P，执行 `python scripts/train.py --stage three_branch` 即可。主模型暂仍指向 `residual_distill`，不会根据本次验证结果自动切换方法。旧训练记录不会被覆盖，PPT 等新训练完成后再更新。
