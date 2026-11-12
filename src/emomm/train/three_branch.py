"""Matched backbone, Adapter, semantic preservation and A/P prompt comparisons."""
from __future__ import annotations
import copy
import csv
from pathlib import Path
import numpy as np
import torch
from . import load_yaml, set_seed, check_identity_splits, training_provenance, write_json, sha256, state_sha256
from . import retrieval_selection_score, tag_balanced_selection_score
from .. import require_device
from ..models.three_branch import FACTORS, FrozenCLaMP3Retriever, RetrieverModel, load_retriever
from ..data import load_pair_data, make_pair_loader, make_mtg_loader, read_features
from ..features.tag_prompts import TRAIN_VIEWS
from .losses import multi_positive_cross_entropy, symmetric_multi_positive_info_nce, symmetric_similarity_distillation_loss
from . import grouped_retrieval_metrics
from . import label_to_track_metrics

ROOT = Path(__file__).resolve().parents[3]


def comparison_configs(config):
    """Keep data, architecture, initialization, and optimizer matched across fits.

    All fits use the same residual Adapter. The third method additionally
    uses the existing semantic output blend and frozen-teacher distillation.
    This is a comparison of the preservation package, not a distillation-only
    ablation. The fourth differs from the third only in A/P training prompts.
    A/P do not acquire additional distillation objectives.
    """
    adapter = copy.deepcopy(config)
    adapter["method"] = "adapter"
    adapter["ap_prompt_mode"] = "single"
    adapter["model"]["semantic_output_blend"] = 1.0
    adapter["loss"]["semantic_distillation_weight"] = 0.0
    preserved = copy.deepcopy(config)
    preserved["method"] = "residual_distill"
    preserved["ap_prompt_mode"] = "single"
    augmented = copy.deepcopy(preserved)
    augmented.update(method="ap_multiview", ap_prompt_mode="multiview")
    return {"adapter": adapter, "residual_distill": preserved, "ap_multiview": augmented}


def _summarize_metrics(values):
    first = values[0]
    if isinstance(first, dict):
        if any(set(value) != set(first) for value in values):
            raise ValueError("Comparison metrics have inconsistent keys")
        return {key: _summarize_metrics([value[key] for value in values]) for key in first}
    numbers = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numbers).all():
        raise ValueError("Nonfinite comparison metric")
    return dict(mean=float(numbers.mean()),
                sample_sd=float(numbers.std(ddof=1)) if len(numbers) > 1 else None,
                n=len(numbers))


def evaluate_validation(model, pair_data, mtg, music_index, config, device):
    cfg = config["training"]
    result = {"semantic_pairs": pair_evaluation(model, pair_data, "validation", device,
              cfg["batch_size_pairs"], tuple(config["evaluation"]["retrieval_ks"]))}
    result.update({f"{factor}_tags": tag_evaluation(model, factor, mtg, music_index, "validation", device,
                  cfg["batch_size_mtg"], tuple(config["evaluation"]["tag_ks"]))
                  for factor in ("affect", "preference")})
    return result


def evaluate_prompt_robustness(model, mtg, music_index, config, device):
    """A supplementary unseen template; never used to select epochs or seeds."""
    return {f"{factor}_tags": tag_evaluation(model, factor, mtg, music_index, "validation", device,
                config["training"]["batch_size_mtg"], tuple(config["evaluation"]["tag_ks"]),
                query_embeddings=mtg[f"{factor}_held_out_embeddings"])
            for factor in ("affect", "preference")}


def write_ap_prompt_comparison(output, single_results, multiview_results):
    """Report paired seed differences without automatically choosing a winner."""
    single = {result["seed"]: result for result in single_results}
    multiview = {result["seed"]: result for result in multiview_results}
    if not single or set(single) != set(multiview):
        raise ValueError("A/P prompt comparison requires identical seeds")
    seeds = sorted(single)
    for seed in seeds:
        if single[seed]["training"]["initial_state_sha256"] != multiview[seed]["training"]["initial_state_sha256"]:
            raise ValueError("A/P comparison did not start from identical model initialization")
    rows = []
    for query_set in ("canonical", "held_out_template"):
        for factor in ("affect", "preference"):
            for metric in ("ndcg_at_10", "average_precision"):
                def collect(result):
                    values = (result["results"]["validation"] if query_set == "canonical"
                              else result["prompt_robustness_validation"])
                    return values[f"{factor}_tags"]["macro"][metric]
                original = np.asarray([collect(single[seed]) for seed in seeds])
                augmented = np.asarray([collect(multiview[seed]) for seed in seeds])
                delta = augmented - original
                row = dict(query_set=query_set, factor=factor, metric=metric,
                    single_mean=float(original.mean()),
                    single_sample_sd=float(original.std(ddof=1)) if len(seeds) > 1 else None,
                    multiview_mean=float(augmented.mean()),
                    multiview_sample_sd=float(augmented.std(ddof=1)) if len(seeds) > 1 else None,
                    absolute_delta=float(delta.mean()),
                    relative_change_percent=float(100 * delta.mean() / original.mean()) if original.mean() else None,
                    improved_seeds=int((delta > 0).sum()), total_seeds=len(seeds),
                    paired_deltas=[dict(seed=seed, delta=float(value)) for seed, value in zip(seeds, delta)])
                rows.append(row)
                if query_set == "canonical":
                    print(f"A/P 提示对比 {factor} {metric}: 单提示={row['single_mean']:.6f}；"
                          f"多表达={row['multiview_mean']:.6f}；差值={row['absolute_delta']:+.6f}；"
                          f"提升种子={row['improved_seeds']}/{len(seeds)}", flush=True)
    report = dict(split="validation", baseline="residual_distill", treatment="ap_multiview",
        seeds=seeds, primary_query_set="canonical", secondary_query_set="held_out_template",
        training_views_per_label=TRAIN_VIEWS, sampling="one_uniform_view_per_label_per_step",
        target_labels_unchanged=True, additional_ap_distillation=False,
        checkpoint_selection="canonical_validation_only",
        interpretation="Positive absolute_delta favors multiview. These are paired descriptive differences, "
                       "not a significance test. The secondary template is unseen during fitting; it is "
                       "template robustness, not unseen-label or human-query evaluation.", rows=rows)
    write_json(output / "ap_prompt_comparison.json", report)
    fields = [key for key in rows[0] if key != "paired_deltas"]
    with (output / "ap_prompt_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def train_comparison(config, pair_data, mtg, music_index, seeds, output):
    """Evaluate once without fitting, then independently fit three methods per seed.

    The production residual_distill group's existing seed_N paths are kept.
    Comparison-only artifacts live under three_branch/comparison/.
    """
    output = Path(output)
    device = require_device(config["device"])
    baseline = FrozenCLaMP3Retriever().to(device).eval()
    baseline_validation = evaluate_validation(baseline, pair_data, mtg, music_index, config, device)
    baseline_result = dict(method="clamp3", trainable_parameters=0, training_performed=False,
                           seed=None, results={"validation": baseline_validation}, test_used=False,
                           prompt_robustness_validation=evaluate_prompt_robustness(baseline, mtg, music_index, config, device))
    baseline_path = output / "comparison/clamp3/metrics.json"
    write_json(baseline_path, baseline_result)
    methods = {"clamp3": dict(trained=False, repetitions=1, seeds=[],
               metrics_path=baseline_path.relative_to(output).as_posix(),
               validation=_summarize_metrics([baseline_validation]),
               prompt_robustness_validation=_summarize_metrics([baseline_result["prompt_robustness_validation"]]))}
    configs = comparison_configs(config)
    final_results = None
    paired_results = {}
    for method, variant in configs.items():
        target = output if method == "residual_distill" else output / "comparison" / method
        print(f"三分支对比组: {method}；种子: {seeds}", flush=True)
        results = [train_seed(variant, pair_data, mtg, music_index, seed, target / f"seed_{seed}")
                   for seed in seeds]
        selected = max(results, key=lambda result: (result["best_validation_score"], -result["seed"]))
        aggregate_path = target / "aggregate.json"
        write_json(aggregate_path, dict(method=method, runs=results, selected_seed=selected["seed"],
                                        test_used_for_selection=False))
        methods[method] = dict(trained=True, repetitions=len(results), seeds=list(seeds),
                               aggregate_path=aggregate_path.relative_to(output).as_posix(),
                               validation=_summarize_metrics([result["results"]["validation"] for result in results]),
                               prompt_robustness_validation=_summarize_metrics([result["prompt_robustness_validation"] for result in results]))
        if method == "residual_distill":
            final_results = results
        if method in {"residual_distill", "ap_multiview"}:
            paired_results[method] = results
    write_ap_prompt_comparison(output, paired_results["residual_distill"], paired_results["ap_multiview"])
    report = dict(feature_id=config["feature_id"], split="validation", sd_ddof=1,
                  production_method="residual_distill", methods=methods,
                  matched_protocol=dict(shared_data=True, shared_lyric_views=True, seeds=list(seeds),
                                        shared_adapter_architecture=True, shared_optimizer=True,
                                        shared_selection_criteria=True, canonical_evaluation_prompts=True,
                                        prompt_sampling_rng="private_numpy_generator_per_seed",
                                        held_out_template_used_for_selection=False, test_used=False),
                  method_definitions={
                      "clamp3": "Frozen CLaMP3 text/audio cosine retrieval; no Adapter and no fitting",
                      "adapter": "Residual Adapters with supervised S/A/P losses; no output blend or distillation",
                      "residual_distill": "Same Adapters plus semantic raw-feature blend and similarity distillation",
                      "ap_multiview": "Same as residual_distill; sample one of four equivalent A/P prompts per label per step",
                  },
                  preservation=dict(factor="semantic",
                                    adapted_output_weight=config["model"]["semantic_output_blend"],
                                    distillation_weight=config["loss"]["semantic_distillation_weight"],
                                    distillation_temperature=config["loss"]["semantic_distillation_temperature"]),
                  interpretation="The third-vs-second comparison tests blend plus distillation jointly. "
                                 "The fourth-vs-third comparison changes only A/P training text views; S/VA recipes stay fixed. "
                                 "The deterministic frozen reference is evaluated once; its SD is null, not a repeated-seed estimate.")
    write_json(output / "comparison.json", report)
    return final_results

def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path

def cycle(loader):
    if not len(loader):
        raise ValueError("Training batch size exceeds the available data")
    while True:
        yield from loader

def evenly_spaced_steps(active_steps, total_steps):
    if not 0 <= active_steps <= total_steps or total_steps <= 0:
        raise ValueError("active_steps must be in [0, total_steps]")
    return {(i * total_steps)//active_steps for i in range(active_steps)}

def capture_factor_state(model, factor):
    if factor not in FACTORS:
        raise ValueError(f"Unknown factor: {factor}")
    prefixes = (f"text_adapters.{factor}.", f"music_adapters.{factor}.", f"logit_scales.{factor}")
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(prefixes)}

def assembled_factor_state(model, factor_states):
    if set(factor_states) != set(FACTORS):
        raise ValueError("Missing validation-selected factor state")
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for factor in FACTORS:
        state.update(factor_states[factor])
    return state

@torch.inference_mode()
def project(model, values, modality, factor, device, batch_size):
    return np.concatenate([model.encode_factor(modality,factor,torch.from_numpy(values[i:i+batch_size].astype(np.float32)).to(device)).cpu().numpy()
                           for i in range(0,len(values),batch_size)])

@torch.inference_mode()
def pair_evaluation(model, data, split, device, batch_size, ks):
    model.eval()
    mask = data['splits'].astype(str) == split
    if not mask.any():
        raise ValueError(f"No semantic pairs in {split}")
    text = project(model,data['text_embeddings'][mask],'text','semantic',device,batch_size)
    music = project(model,data['audio_embeddings'][mask],'music','semantic',device,batch_size)
    ids, domains = data['global_identities'][mask].astype(str), data['datasets'][mask].astype(str)
    per_dataset = {name:grouped_retrieval_metrics(text[domains==name],music[domains==name],ids[domains==name],ks)
                   for name in sorted(set(domains))}
    score = np.mean([.5*(v['text_to_music']['mrr']+v['music_to_text']['mrr']) for v in per_dataset.values()])
    return {'overall':grouped_retrieval_metrics(text,music,ids,ks), 'per_dataset':per_dataset, 'selection_score':float(score)}

@torch.inference_mode()
def tag_evaluation(model, factor, mtg, music_index, split, device, batch_size, ks, *, query_embeddings=None):
    model.eval()
    mask = mtg['splits'].astype(str) == split
    text = project(model,mtg[f'{factor}_label_embeddings'] if query_embeddings is None else query_embeddings,
                   'text',factor,device,batch_size)
    music = project(model,music_index[mtg['index_positions'][mask]],'music',factor,device,batch_size)
    result = label_to_track_metrics(text@music.T,mtg[f'{factor}_targets'][mask].astype(bool).T,ks)
    result.pop('per_query')
    return result

def sample_label_views(views, rng):
    """One query per label keeps loss scale, target columns and dropout shapes fixed."""
    count, choices, _ = views.shape
    selected = torch.as_tensor(rng.integers(choices, size=count), device=views.device)
    return views[torch.arange(count, device=views.device), selected]


def train_seed(config, pair_data, mtg, music_index, seed, output, evaluate_test=False):
    set_seed(seed)
    output.mkdir(parents=True,exist_ok=False)
    device = require_device(config['device'])
    model_config = config['model']
    model = RetrieverModel(**model_config).to(device)
    cfg, weights = config['training'], config['loss']
    provenance = training_provenance(model,seed,config)
    pairs = make_pair_loader(pair_data,'train',cfg['batch_size_pairs'],seed)
    tags = make_mtg_loader(mtg,music_index,'train',cfg['batch_size_mtg'],seed)
    if not len(pairs) or not len(tags):
        raise ValueError('Both semantic and tag training require a complete batch')
    labels = {f:torch.from_numpy(mtg[f'{f}_label_embeddings'].astype(np.float32)).to(device) for f in ('affect','preference')}
    prompt_mode = config.get('ap_prompt_mode', 'single')
    if prompt_mode not in {'single', 'multiview'}:
        raise ValueError('Unknown A/P prompt training mode')
    # A dedicated RNG never advances the torch/NumPy streams used by data or dropout.
    prompt_rng = np.random.default_rng(np.random.SeedSequence([seed, 1701]))
    label_views = ({f: torch.from_numpy(mtg[f'{f}_view_embeddings'].astype(np.float32))
                        .reshape(len(labels[f]), TRAIN_VIEWS, 768).to(device)
                    for f in ('affect','preference')} if prompt_mode == 'multiview' else {})
    provenance['ap_prompt_training'] = dict(mode=prompt_mode, views_per_label=TRAIN_VIEWS if label_views else 1,
        sampling='one_uniform_view_per_label_per_step' if label_views else 'canonical_only',
        targets='unchanged_original_multilabel_matrix', evaluation='canonical_only_for_selection')
    semantic_parameters = [*model.text_adapters['semantic'].parameters(),*model.music_adapters['semantic'].parameters(),model.logit_scales['semantic']]
    semantic_ids = {id(p) for p in semantic_parameters}
    optimizer = torch.optim.AdamW([
        {'params':semantic_parameters,'lr':cfg['semantic_learning_rate']},
        {'params':[p for p in model.parameters() if id(p) not in semantic_ids],'lr':cfg['learning_rate']}],weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['max_epochs'])
    amp = bool(cfg['amp']) and device.type=='cuda'
    scaler = torch.amp.GradScaler('cuda',enabled=amp)
    best_scores = {f:-float('inf') for f in FACTORS}
    stop_reference = -float('inf')
    best_epochs, best_states = {f:0 for f in FACTORS}, {}
    history, bad_epochs = [], 0
    retrieval_ks, tag_ks = tuple(config['evaluation']['retrieval_ks']),tuple(config['evaluation']['tag_ks'])
    for epoch in range(1,cfg['max_epochs']+1):
        model.train()
        steps = max(len(pairs),len(tags))
        pair_steps = evenly_spaced_steps(len(pairs),steps)
        pair_iter, tag_iter = iter(pairs),cycle(tags)
        totals = {'total':0.,'semantic':0.,'semantic_distillation':0.,'affect':0.,'preference':0.}
        for step in range(steps):
            raw_music, pref_target, affect_target = [v.to(device) for v in next(tag_iter)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=amp):
                losses = {key:raw_music.sum()*0 for key in ('semantic','semantic_distillation')}
                if step in pair_steps:
                    raw_text, raw_audio, ids, _ = [v.to(device) for v in next(pair_iter)]
                    text = model.encode_factor('text','semantic',raw_text)
                    music = model.encode_factor('music','semantic',raw_audio)
                    losses['semantic'] = symmetric_multi_positive_info_nce(text,music,ids[:,None]==ids[None,:],logit_scale=model.logit_scales['semantic'].exp())
                    if weights['semantic_distillation_weight'] > 0:
                        losses['semantic_distillation'] = symmetric_similarity_distillation_loss(text,music,raw_text,raw_audio,temperature=weights['semantic_distillation_temperature'])
                for factor, target in (('preference',pref_target),('affect',affect_target)):
                    queries = sample_label_views(label_views[factor], prompt_rng) if label_views else labels[factor]
                    logits = model.factor_similarity(factor,queries,raw_music,scaled=True)
                    direction = weights[f'{factor}_label_to_track_weight']
                    losses[factor] = direction*multi_positive_cross_entropy(logits,target.T,dim=1)+(1-direction)*multi_positive_cross_entropy(logits,target.T,dim=0)
                loss = (weights['semantic_weight']*losses['semantic']+weights['semantic_distillation_weight']*losses['semantic_distillation']
                        +weights['preference_weight']*losses['preference']+weights['affect_tag_weight']*losses['affect'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite retrieval training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            totals['total'] += loss.item()
            for name,value in losses.items():
                totals[name] += value.item()
        scheduler.step()
        validation = {'semantic_pairs':pair_evaluation(model,pair_data,'validation',device,cfg['batch_size_pairs'],retrieval_ks)}
        scores = {'semantic':validation['semantic_pairs']['selection_score']}
        for factor in ('affect','preference'):
            metrics = tag_evaluation(model,factor,mtg,music_index,'validation',device,cfg['batch_size_mtg'],tag_ks)
            validation[f'{factor}_tags'] = metrics
            scores[factor] = tag_balanced_selection_score(metrics['macro']['ndcg_at_10'],metrics['macro']['average_precision'])
        improved = False
        for factor in FACTORS:
            if scores[factor] > best_scores[factor]:
                best_scores[factor],best_epochs[factor] = scores[factor],epoch
                best_states[factor] = capture_factor_state(model,factor)
                improved = True
        best_score = retrieval_selection_score(best_scores['semantic'],best_scores['preference'],best_scores['affect'])
        if improved:
            torch.save({'model':assembled_factor_state(model,best_states),'model_config':model_config,'seed':seed,
                        'epoch':max(best_epochs.values()),'factor_best_epochs':best_epochs.copy(),'factor_best_validation_scores':best_scores.copy(),
                        'selection_score':best_score,'run_id':config['run_id'],'feature_id':config['feature_id'],
                        'method':config.get('method','residual_distill'),'ap_prompt_mode':prompt_mode},output/'best.pt')
        if best_score > stop_reference + cfg['early_stopping_min_delta']:
            stop_reference, bad_epochs = best_score, 0
        else:
            bad_epochs += 1
        row = {'epoch':epoch,**{f'loss_{k}':v/(len(pairs) if k.startswith('semantic') else steps) for k,v in totals.items()},
               **{f'validation_{k}':v for k,v in scores.items()},'best_validation_score':best_score}
        history.append(row)
        with (output/'history.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader();writer.writerows(history)
        print(f'Retriever method={config.get("method","residual_distill")} seed={seed} epoch={epoch} loss={row["loss_total"]:.5f} validation={best_score:.5f}',flush=True)
        if bad_epochs >= cfg['early_stopping_patience']:
            break
    loaded = load_retriever(output/'best.pt').to(device)
    provenance.update(best_epoch=max(best_epochs.values()),factor_best_epochs=best_epochs,
                      final_state_sha256=state_sha256(loaded),checkpoint_sha256=sha256(output/'best.pt'),save_reload_verified=True)
    if provenance['initial_state_sha256']==provenance['final_state_sha256']:
        raise RuntimeError('Training did not change adapter parameters')
    results = {}
    for split in (('validation','test') if evaluate_test else ('validation',)):
        results[split] = {'semantic_pairs':pair_evaluation(loaded,pair_data,split,device,cfg['batch_size_pairs'],retrieval_ks)}
        results[split].update({f'{f}_tags':tag_evaluation(loaded,f,mtg,music_index,split,device,cfg['batch_size_mtg'],tag_ks) for f in ('affect','preference')})
    robustness = evaluate_prompt_robustness(loaded, mtg, music_index, config, device)
    result = {'method':config.get('method','residual_distill'),'seed':seed,'best_epoch':max(best_epochs.values()),'factor_best_epochs':best_epochs,'best_validation_score':best_score,'results':results,'training':provenance,
              'prompt_robustness_validation':robustness}
    write_json(output/'metrics.json',result)
    return result

def load_training_data(config, excluded=None):
    cfg = config['supervision']
    excluded = excluded or {}
    blocks = []
    for value in cfg['semantic_pairs']:
        path = resolve(value)
        block = load_pair_data([path])
        with np.load(path, allow_pickle=False) as archive:
            source_name = 'pairs_merge' if path.stem == 'pairs_merge_views' else path.stem
            source_ids = archive['source_pair_ids'] if source_name != path.stem else archive['ids']
            keep = ~np.isin(source_ids.astype(str), excluded.get(source_name, []))
            if source_name != path.stem and not np.all(block['splits'].astype(str) == 'train'):
                raise ValueError('Lyric views must contain training rows only')
        blocks.append({k: v[keep] for k, v in block.items()})
    pairs = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    check_identity_splits(pairs['global_identities'],pairs['splits'],'semantic pairs')
    with np.load(resolve(cfg['mtg_factors']),allow_pickle=False) as data:
        tags = {k:data[k] for k in data.files}
    prompt_path = resolve(cfg['tag_prompt_views'])
    if sha256(prompt_path) != cfg['tag_prompt_views_sha256']:
        raise ValueError('A/P prompt feature hash changed before training')
    prompts = read_features(prompt_path)
    for factor in ('affect', 'preference'):
        if not np.array_equal(prompts[f'{factor}_labels'], tags[f'{factor}_labels']):
            raise ValueError('A/P prompt labels do not align with the original targets')
        if not np.array_equal(prompts[f'{factor}_view_embeddings'].reshape(-1, TRAIN_VIEWS, 768)[:,0], tags[f'{factor}_label_embeddings']):
            raise ValueError('A/P prompt canonical vectors do not match original label features')
        for suffix in ('view_embeddings', 'held_out_embeddings'):
            tags[f'{factor}_{suffix}'] = prompts[f'{factor}_{suffix}']
    keep = ~np.isin(tags['track_ids'].astype(str), excluded.get('mtg_music', []))
    for key in ('track_ids', 'index_positions', 'artist_ids', 'splits', 'affect_targets', 'preference_targets'):
        if key in tags:
            tags[key] = tags[key][keep]
    with np.load(resolve(cfg['music_index']),allow_pickle=False) as data:
        music, ids = data['embeddings'].astype(np.float32),data['track_ids'].astype(str)
    check_identity_splits(tags['track_ids'],tags['splits'],'MTG')
    if not np.array_equal(ids[tags['index_positions']],tags['track_ids'].astype(str)):
        raise ValueError('Tag supervision does not align with music index track IDs')
    mtg_splits = {str(int(i.removeprefix('track_'))):str(s) for i,s in zip(tags['track_ids'],tags['splits'],strict=True)}
    for identity,split in zip(pairs['global_identities'],pairs['splits'],strict=True):
        if str(identity).startswith('mtg:'):
            catalog_split = mtg_splits.get(str(int(str(identity)[4:])))
            if (catalog_split is None and split!='test') or (catalog_split is not None and catalog_split!=split):
                raise ValueError(f'Song Describer / MTG split disagreement: {identity}')
    return pairs,tags,music
