"""Publish fixed-order development examples from the selected corrected pilot."""
from __future__ import annotations

from collections import defaultdict
import argparse
import json
import logging
from pathlib import Path

from hindi_detector.artifacts import write_group
from hindi_detector.corrected_synthesis import sha
from hindi_detector.structured_detector import load_model
from scripts.train_corrected_detector import samples,evaluate

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint-group',choices=['pilot','preview'],default='pilot');args=parser.parse_args()
    group=ROOT/'reports/corrected_training_v16'/args.checkpoint_group;inputs=json.loads((group/'inputs.json').read_text())
    grouped=defaultdict(list);lookup={}
    for receipt in inputs['frozen_receipts']['development']:
        grouped[receipt['text_id']].append(receipt);lookup[receipt['clip_id']]=receipt
    network,state=load_model(group/'model.pt')
    contrast={r['word'] for r in json.loads((ROOT/'data/pocket_bulk_plan_v11/g2p_priors.json').read_text())}
    rows=samples(grouped,Path(inputs['features']),contrast);metrics,predictions,coverage=evaluate(network,state,rows)
    for row in predictions:
        source=lookup[row['query_clip_id']]['row']
        row.update(audio=source['audio'],audio_sha256=source['sha256'],source_pack=source['source_pack'])
    write_group(ROOT/'reports/corrected_training_v16'/(args.checkpoint_group+'_examples'),{'report.json':{'model_sha256':sha(group/'model.pt'),
        'split':'development','metrics':metrics,'site_coverage':coverage,'actual_hindi_error_accuracy':None},
        'predictions.json':predictions,'source_snapshot.json':{'sha256':sha(Path(__file__)),'source_utf8':Path(__file__).read_text()}},
        'Fixed-order development-only word listening examples for the selected pilot. predictions.json stores actual query audio identity, word intervals/source scores and site support. No reserved examples loaded; this is not test accuracy. Original WAVs stay in documented evidence group.',
        ['.venv-indicf5/bin/python -m scripts.audit_structured_pilot','Selected model and frozen development receipts only'])


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    try:main()
    except Exception:
        logging.exception('Pilot development export failed explicitly');raise
