"""Build a bare frozen MTG catalog from declared per-track CLaMP3 embeddings.

No old predicted VA or Adapter index is copied. Deployment generates those from
freshly trained weights. This tool never overwrites an existing catalog.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from emomm_sequence.data.tags import read_mtg_autotagging_tsv
from emomm_sequence.training.records import sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/data.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,help='Only for a small construction smoke test; not for training.')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    checkpoint=ROOT/config['clamp3']['checkpoint']
    if sha256(checkpoint)!=config['clamp3']['checkpoint_sha256']:raise ValueError('Backbone checkpoint hash mismatch')
    emb_root=Path(config['paths']['clamp3_precomputed_audio_embeddings'])
    audio_root=Path(config['paths']['mtg_audio'])
    split_root=Path(config['paths']['mtg_split_dir'])
    rows=[]
    for split in ('train','validation','test'):rows+=read_mtg_autotagging_tsv(split_root/f'autotagging-{split}.tsv',split)
    rows=sorted(rows,key=lambda r:r.track_id)
    if len({r.track_id for r in rows})!=len(rows):raise ValueError('Repeated official track IDs')
    ids=[];audio=[];vectors=[];sources=[];missing=[]
    for row in rows:
        source=emb_root/f'{row.track_id}.npy';audio_path=audio_root/row.relative_path
        if not source.is_file() or not audio_path.is_file():
            missing.append({'track_id':row.track_id,'embedding_exists':source.is_file(),'audio_exists':audio_path.is_file()});continue
        raw=np.load(source,allow_pickle=False).astype(np.float32)
        if raw.shape not in {(768,),(1,768)}:raise ValueError(f'Unexpected frozen feature shape: {source}: {raw.shape}')
        vector=raw.reshape(768);norm=float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm<1e-8:raise ValueError(f'Invalid vector: {source}')
        ids.append(row.track_id);audio.append(str(audio_path));vectors.append(vector/norm)
        sources.append({'track_id':row.track_id,'sha256':sha256(source)})
        if args.limit and len(ids)>=args.limit:break
    if not ids:raise FileNotFoundError('No available audio/embedding pairs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,track_ids=np.asarray(ids),audio_paths=np.asarray(audio),embeddings=np.stack(vectors),source_embedding_dir=np.asarray(str(emb_root)),checkpoint_sha256=np.asarray(config['clamp3']['checkpoint_sha256']))
    write_json(args.output.with_suffix('.json'),{'tracks':len(ids),'source_features':sources,'missing':missing,'test_only_subset':bool(args.limit),'checkpoint_sha256':config['clamp3']['checkpoint_sha256'],'output_sha256':sha256(args.output),'source_provenance_note':'Declared external frozen feature input; individual file hashes preserved, not re-inferred from values.'})
    print(json.dumps({'tracks':len(ids),'missing':len(missing),'output':str(args.output)}))


if __name__=='__main__':main()
