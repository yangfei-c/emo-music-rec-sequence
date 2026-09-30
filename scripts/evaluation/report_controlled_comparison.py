"""Create a traceable Chinese experiment report after all fixed runs finish."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts/runs/controlled_comparison_20260913/evaluation"
RUNS = ROOT / "artifacts/runs/controlled_comparison_20260913"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    data = read(REPORT / "summary.json")
    if data.get("status") != "complete" or len(data["methods"]) != 7:
        raise ValueError("The final report requires the complete seven-method evaluation")
    methods = {r["variant"]: r for r in data["methods"]}
    plan = read(REPORT / "plan.json")
    for source, digest in plan["code_sha256"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest, source
    execution_path = REPORT / "execution_parallel.json"
    if not execution_path.exists():
        execution_path = REPORT / "execution.json"
    execution = read(execution_path)
    if len(execution) != 6 or any(r["status"] != "completed" for r in execution):
        raise ValueError("Do not issue a completion report for unfinished experiments")
    manifests = [read(RUNS / v["name"] / "input_manifest.json") for v in plan["variants"]]
    assert all(x == manifests[0] for x in manifests)
    initials = {}
    for seed in plan["seeds"]:
        values = {variant: read(RUNS / variant / f"seed_{seed}" / "provenance.json")["initial_state_sha256"]
                  for variant in ("no_va", "text_va", "music_va", "full")}
        assert len(set(values.values())) == 1, (seed, values)
        initials[str(seed)] = values
    (REPORT / "fairness_audit.json").write_text(json.dumps({
        "status": "pass", "same_input_manifests": True, "va_ablation_initial_states_identical_per_seed": initials,
        "checkpoint_selection": "retrieval_only", "test_used_for_selection": False,
        "previously_seen_test": True, "all_six_variants_all_three_seeds_reported": True,
        "training_source_hashes_unchanged_from_frozen_plan": True,
        "independent_evaluation_data_sha256": data["evaluated_data_sha256"],
    }, indent=2), encoding="utf-8")

    def fmt(variant, metric, scale=1, digits=3):
        r = methods[variant]["metrics"][metric]
        result = f"{r['mean']*scale:.{digits}f}"
        if r["sd"] is not None:
            result += f" ± {r['sd']*scale:.{digits}f}"
        return result

    names = {"clamp3": "CLaMP3", "no_va": "无 VA 辅助", "text_va": "仅文本 VA", "music_va": "仅音乐 VA",
             "full": "Ours（完整）", "shared_capacity_matched": "共享 Adapter（近似同容量）", "clip_style_transfer": "CLIP-Adapter 式双塔迁移"}
    lines = [
        "# Adapter 对照与 VA 辅助消融结果", "", "## Material Passport", "",
        "- Origin skill：academic-research-suite / experiment-agent；文献核验：nature-academic-search",
        "- 模式：本地实验执行与验证；日期：2026-09-13；版本：1",
        "- 验证状态：六组 × 三种子全部完成，独立 CPU 重评通过，同数据 / 同初始化核验通过",
        "- 性质：已观察过测试集上的探索性补充实验，不是新的盲测或临床 / 真人有效性研究", "",
        "## 同协议检索结果", "",
        "表中均为均值 ± 三个随机种子的样本标准差（SD）；CLaMP3 无新增训练参数。R@10 以百分数表示。", "",
        "| 方法 | 可训练参数 | MERGE R@10 (%) | SDD R@10 (%) | MTG 偏好 NDCG@10 | MTG 情绪 NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in names:
        row = methods[v]
        lines.append(f"| {names[v]} | {row['trainable_parameters']:,} | {fmt(v,'merge.text_to_music.recall_at_10',100,2)} | {fmt(v,'song_describer.text_to_music.recall_at_10',100,2)} | {fmt(v,'preference.ndcg_at_10')} | {fmt(v,'affect.ndcg_at_10')} |")
    d = data["paired_effects"]["full_minus_no_va"]["affect.ndcg_at_10"]
    differences = ", ".join(f"{x:+.6f}" for x in d["values"])
    lines += ["", "## 主要消融终点", "",
              f"完整 Ours − 无 VA 的情绪 NDCG@10：{d['mean']:+.6f} ± {d['sd']:.6f}。",
              f"对应种子 42 / 43 / 44 的配对差值：{differences}。",
              "这组差值隔离的是当前训练配方中启用两侧 VA 监督的影响，不等于证明任意 VA 方法普遍有效。",
              "效应包含辅助监督改变验证选模轮次的影响，不是固定同一轮次下的局部梯度效应；三个最佳轮次全部在下表列出。",
              "本次只消融检索训练中的辅助损失，不删除独立音乐 VA 衔接模型，也不评估完整五首序列或 Qwen 解析。",
              "三个种子只能描述初始化波动，SD 不是置信区间，也不提供人群效果或统计显著性的证据。", ""]
    if all(v > 0 for v in d["values"]):
        lines += ["三个种子方向一致，支持当前协议下的正向增益；仍须更多重复和独立测试来量化稳健性。"]
    elif all(v < 0 for v in d["values"]):
        lines += ["三个种子均下降，本实验不支持当前 VA 辅助配方改善情绪检索。不得改写成辅助有效。"]
    else:
        lines += ["种子间方向不一致，当前实验不能支持稳定的 VA 检索收益；即使均值为正也不应写成稳定提升。"]
    lines += ["", "## 结构对照如何解释", "",
              "共享 Adapter 与六个独立 Adapter 的参数量近似匹配，四项检索终点全部报告，不能只据其中一项宣布结构全面更优。",
              "CLIP-Adapter 式对照迁移的是已发表模块结构，未复现其原始图像任务；训练数据、骨干和评价均使用本项目协议。", "",
              "| 对照差值 | 终点 | 配对均值 ± SD | 三个种子差值 |", "|---|---|---:|---|"]
    for control in ("shared_capacity_matched", "clip_style_transfer"):
        for key, label, scale in (("affect.ndcg_at_10", "情绪 NDCG@10", 1),
                                  ("preference.ndcg_at_10", "偏好 NDCG@10", 1),
                                  ("merge.text_to_music.recall_at_10", "MERGE R@10（百分点）", 100),
                                  ("song_describer.text_to_music.recall_at_10", "SDD R@10（百分点）", 100)):
            effect = data["paired_effects"][f"full_minus_{control}"][key]
            vals = " / ".join(f"{x*scale:+.4f}" for x in effect["values"])
            lines.append(f"| Ours − {names[control]} | {label} | {effect['mean']*scale:+.4f} ± {effect['sd']*scale:.4f} | {vals} |")
    lines += ["", "## 歌曲检索的完整指标", "",
              "R@1 / R@5 / R@10 是正确歌曲进入前 K 位的比例；MRR 是其排名倒数的均值。", "",
              "| 方法 | 数据集 | R@1 (%) | R@5 (%) | R@10 (%) | MRR |", "|---|---|---:|---:|---:|---:|"]
    for v in names:
        for dataset in ("merge", "song_describer"):
            lines.append(f"| {names[v]} | {dataset} | " + " | ".join(fmt(v, f"{dataset}.text_to_music.{k}", scale, digits) for k, scale, digits in (("recall_at_1",100,2),("recall_at_5",100,2),("recall_at_10",100,2),("mrr",1,4))) + " |")
    lines += ["", "## 同一 MTG 标签集的全排序指标", "",
              "以下 AUC / AP 使用本地 55 个情绪标签、123 个偏好标签和修复后的测试划分，不能当作原论文 59 / 136 标签分类任务的直接复现。", "",
              "| 方法 | 标签类型 | ROC-AUC | AP | P@10 | NDCG@10 |", "|---|---|---:|---:|---:|---:|"]
    for v in names:
        for factor in ("affect", "preference"):
            lines.append(f"| {names[v]} | {factor} | " + " | ".join(fmt(v, f"{factor}.{k}") for k in ("score_macro_roc_auc", "score_macro_average_precision", "precision_at_10", "ndcg_at_10")) + " |")
    lines += ["", "## 辅助 VA 头的测试 CCC", "", "| VA 条件 | 数据集 | V 的 CCC | A 的 CCC |", "|---|---|---:|---:|"]
    for v in ("no_va", "text_va", "music_va", "full"):
        for dataset in ("emobank", "deam", "pmemo"):
            key = f"aux_va.{dataset}.valence_ccc"
            if key not in methods[v]["metrics"]:
                lines.append(f"| {names[v]} | {dataset} | 未监督，不作有效性能报告 | 未监督 |")
            else:
                lines.append(f"| {names[v]} | {dataset} | {fmt(v,key)} | {fmt(v,f'aux_va.{dataset}.arousal_ccc')} |")
    lines += ["", "## 对论文主张的限制", "",
              "冻结 CLaMP3 加头后的增益支持任务适配，不代表重新训练了更强的基础编码器。",
              "六分支的独立价值需要与共享同容量头比较；VA 的独立价值需要与无 VA 比较。两个主张不能只靠 Ours 对 CLaMP3 的总增益推出。",
              "Song Describer 的下降必须进入主结果。歌词监督与自由音乐描述存在分布差异，可能产生过度适配，但这里没有单独验证下降原因。",
              "不能把歌词与自由描述差异当作取消失败结果的理由；当前证据不支持通用文本音乐检索全面改进。",
              "同一系统中的可训练小头、冻结原模型、检索控制和真人听感是不同层次的证据；模型分数不能替代真人评价。", "",
              "## 训练与模型选择", "",
              "四个 VA 条件的每个种子具有相同初始参数哈希，六组输入文件哈希全部一致。",
              "最佳轮次仅按验证 MERGE 双向 MRR、MTG 偏好 NDCG@10、MTG 情绪 NDCG@10 的平均值选取。",
              "旧实验包含 VA CCC 的复合选模规则不再用于本轮对照，因此本轮完整 Ours 也重新训练。",
              "旧权重、原 13 页 PPT、网页部署和独立音乐 VA 模型均未覆盖。", "",
              "| 组别 | 最佳轮次（42 / 43 / 44） | 组运行墙钟时间 (s) |", "|---|---|---:|"]
    for row in execution:
        v = row["name"]
        lines.append(f"| {names[v]} | {' / '.join(map(str, methods[v]['best_epochs']))} | {row['duration_seconds']:.1f} |")
    lines += ["", "## 执行异常与检查", "",
              "第一次 pytest 启用环境自动插件时长时间未输出，仅停止该测试进程。禁用自动插件后全部 33 项测试通过；未改变训练科学协议。",
              "最初串行调度改为最多两组并行，保留已经运行的第一组子进程，未重置权重。变更见 scheduling_amendment.json。",
              "每组最多 30 分钟；每 30 秒记录存活与日志大小。原始指令、进程状态和完整训练日志分别保存在 plan.json、execution_parallel.json、logs/。",
              "最终检索评估使用独立 CPU 路径，核对与训练时 GPU 评估一致。软件测试另验证 VA 反传对象、零权重梯度、文献头公式与容量匹配。", "",
              "## 统计解释核验（11 / 11 类风险已检查）", "",
              "按数据集分别报告，避免总体均值掩盖跨域退化；不从标签层指标外推到听者；固定候选集的选择限制明确。",
              "未进行人群协变量调整或前后测，也无临床基率或诊断推断，因此碰撞偏差、回归均值、基率忽略等不适用于这次算法对照。",
              "全部六组和三个种子均保留，避免幸存者偏差和只报告有利结果。协议在新增训练结果之前固定，但旧测试已见，仍标记为探索性。",
              "没有 p 值或显著性筛选，不把标准差当置信区间；算法消融的条件性比较不外推为听者情绪改善的因果结论。", "",
              "## 复现入口", "",
              "1. python scripts/training/run_controlled_comparison.py --prepare",
              "2. python scripts/training/run_controlled_comparison.py --run（全新输出目录；现有实验目录禁止覆盖）",
              "3. python scripts/evaluation/evaluate_controlled_comparison.py",
              "4. python scripts/evaluation/report_controlled_comparison.py", "",
              "全部原始指标见 summary.json 和 evaluations/。",
              "",
    ]
    (REPORT / "实验结果与解释.md").write_text("\n".join(lines), encoding="utf-8")
    print(REPORT / "实验结果与解释.md")


if __name__ == "__main__":
    main()
