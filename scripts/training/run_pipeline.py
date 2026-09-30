"""Train and evaluate a fresh Route2 experiment under one run directory."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    print('RUN ' + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--prepare-features', action='store_true', help='Prepare missing feature outputs; existing artifacts are not overwritten.')
    parser.add_argument('--download-public-sources', action='store_true', help='Download and checksum MERGE and Song Describer.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,80}', args.run_id):
        parser.error('Use a simple run ID with letters, numbers, underscores or hyphens.')
    run_root = ROOT / 'artifacts/runs' / args.run_id
    result_root = run_root / 'evaluation'
    if run_root.exists():
        raise FileExistsError('Run ID already exists. Fresh training requires a new ID.')
    if args.download_public_sources:
        run('scripts/data/download_sources.py')
    if args.prepare_features:
        manifest_names = ('text_emobank', 'music_deam', 'music_pmemo', 'pairs_merge', 'pairs_song_describer')
        manifests = [ROOT / 'data/manifests' / f'{name}.csv' for name in manifest_names]
        if not all(path.is_file() for path in manifests):
            if any(path.exists() for path in manifests):
                raise FileNotFoundError('Manifest set is incomplete; inspect existing manifests before regenerating it.')
            run('scripts/data/prepare_manifests.py')
        # Existing DEAM/PMEmo/MTG tokens are declared external frozen feature inputs.
        run('scripts/features/extract_mert_tokens.py', '--manifest', 'data/manifests/pairs_merge.csv')
        for name, modality in [('text_emobank','text'),('music_deam','audio'),('music_pmemo','audio'),('pairs_merge','pair'),('pairs_song_describer','pair')]:
            output=ROOT/'artifacts/features/clamp3'/f'{name}.npz'
            if not output.exists():
                run('scripts/features/extract_clamp3_features.py', f'data/manifests/{name}.csv', '--modality', modality, '--output', output)
        if not (ROOT/'artifacts/features/retrieval/mtg_factor_supervision.npz').exists():
            if not (ROOT/'artifacts/catalog/music.npz').exists():
                run('scripts/features/build_music_catalog.py', '--output', 'artifacts/catalog/music.npz')
            run('scripts/features/build_tag_supervision.py')
        if not (ROOT/'artifacts/features/clamp3/clean/content_split_report.json').exists():
            arguments=[]
            for name in ('music_deam','music_pmemo','pairs_merge','pairs_song_describer'):
                arguments += ['--input', f'artifacts/features/clamp3/{name}.npz']
            run('scripts/data/group_content_splits.py', *arguments, '--output-dir', 'artifacts/features/clamp3/clean', '--catalog', 'artifacts/catalog/music.npz', '--tag-supervision', 'artifacts/features/retrieval/mtg_factor_supervision.npz', '--deduplicate-music')
        if not (ROOT/'artifacts/features/clamp3/clean/mtg_factor_supervision_mapped.npz').exists():
            run('scripts/features/build_mapped_tag_supervision.py')
        if not (ROOT/'artifacts/features/clamp3/augmentation/merge_train_views/metadata.json').exists():
            run('scripts/features/build_merge_train_views.py')
    retriever=yaml.safe_load((ROOT/'configs/training/current_retriever.yaml').read_text(encoding='utf-8'))
    va=yaml.safe_load((ROOT/'configs/training/va_quantiles.yaml').read_text(encoding='utf-8'))
    text_va=yaml.safe_load((ROOT/'configs/training/text_va.yaml').read_text(encoding='utf-8'))
    for config in (retriever,va,text_va):config['run_id']=args.run_id
    text_va['data']['retriever_runs']=f'artifacts/runs/{args.run_id}/retriever'
    paths=[*retriever['supervision']['semantic_pairs'], retriever['supervision']['text_va'], *retriever['supervision']['music_va'], retriever['supervision']['mtg_factors'], retriever['supervision']['music_index']]
    missing=[p for p in paths if not (ROOT/p).is_file()]
    if missing:raise FileNotFoundError('Missing inputs: '+json.dumps(missing))
    config_root=run_root/'configs'
    config_root.mkdir(parents=True)
    for name,config in [('retriever',retriever),('va_quantiles',va),('text_va',text_va)]:
        (config_root/f'{name}.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    run('scripts/training/train_retriever.py', '--config',config_root/'retriever.yaml','--output',run_root/'retriever','--validation-only')
    run('scripts/training/train_text_va.py','--config',config_root/'text_va.yaml','--output',run_root/'text_va')
    run('scripts/training/train_va_quantiles.py','--config',config_root/'va_quantiles.yaml','--output',run_root/'music_va_quantiles')
    run('scripts/evaluation/evaluate_retrieval.py','--config',config_root/'retriever.yaml','--runs',run_root/'retriever','--output',result_root/'retrieval_comparison.json')
    run('scripts/evaluation/evaluate_route2.py','--run',run_root/'music_va_quantiles','--output',result_root/'route2_evaluation','--independent-point-policy')
    run('scripts/training/select_run.py','--run-id',args.run_id,'--evaluation',result_root/'route2_evaluation'/'report.json')
    run('scripts/verify.py','--config',f'artifacts/runs/{args.run_id}/selected/system.yaml','--output',result_root/'verification.json')
    print(f'Fresh experiment complete: {run_root}. Active configuration is unchanged.')


if __name__=='__main__':main()
