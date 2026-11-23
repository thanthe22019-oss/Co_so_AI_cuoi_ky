"""Check completeness and budgets of the saved main experiment (no rerun)."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'results/main'
config=json.loads((folder/'config.json').read_text())
with (folder/'episodes.csv').open(encoding='utf8',newline='') as handle:
    rows=list(csv.DictReader(handle))
groups=defaultdict(list)
for row in rows:
    groups[int(row['size']),row['agent'],int(row['seed'])].append(row)
    assert 0 <= float(row['food_fraction']) <= 1, row
    assert row['outcome'] in ('win','lose','timeout'), row
    assert 1 <= int(row['steps']) <= config['max_steps'], row
    assert math.isfinite(float(row['score'])), row
expected=0
for size in config['sizes']:
    for method in config['agents']:
        for seed in config['seeds'][:1] if method=='reflex' else config['seeds']:
            data=groups[size,method,seed]
            evaluations=[r for r in data if r['phase']=='eval']
            training=[r for r in data if r['phase']=='train']
            assert len(evaluations)==config['eval_games'], (size,method,seed,len(evaluations))
            assert sorted(int(r['game']) for r in evaluations)==list(range(config['eval_games']))
            if method!='reflex':
                assert sum(int(r['steps']) for r in training)==config['budget'], (size,method,seed)
                assert all(int(r['training_steps'])==config['budget'] for r in evaluations)
            else:
                assert not training
            expected+=1
print(f'PASS: {expected} required complete configurations; {len(rows)} saved episodes; exact training budgets and valid metrics.')
