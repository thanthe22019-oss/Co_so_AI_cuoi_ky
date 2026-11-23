"""Run independent (layout, method, seed) configurations in separate processes.

Reuses complete results from main/large/shards. Never shares Python's RNG between
workers. Each worker writes its own directory; main CSV is replaced only at end.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.benchmark import completed_runs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    folder=ROOT/'results/main'
    config=json.loads((folder/'config.json').read_text())
    complete={}
    sources=[folder,ROOT/'results/large']+sorted((ROOT/'results/shards').glob('*'))
    fields=None
    for source in sources:
        if not (source/'episodes.csv').exists():continue
        other=json.loads((source/'config.json').read_text())
        if any(other[k]!=config[k] for k in ['budget','eval_games','max_steps']):continue
        with (source/'episodes.csv').open(encoding='utf8',newline='') as handle:
            reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
        valid=completed_runs(rows,config['eval_games'],config['budget'])
        grouped=defaultdict(list)
        for row in rows:
            key=int(row['size']),row['agent'],int(row['seed'])
            if key in valid:grouped[key].append(row)
        for key,data in grouped.items():
            if key not in complete:complete[key]=data
    expected=[(size,method,seed) for size in config['sizes'] for method in config['agents']
              for seed in (config['seeds'][:1] if method=='reflex' else config['seeds'])]
    missing=[key for key in expected if key not in complete]
    print(f'Reusing {len(expected)-len(missing)}/{len(expected)} required runs; {len(missing)} remaining; {args.workers} worker processes',flush=True)

    def run(key):
        size,method,seed=key
        out=ROOT/f'results/shards/{size}_{method}_{seed}'
        out.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-X','utf8',str(ROOT/'experiments/benchmark.py'),
                 '--sizes',str(size),'--agents',method,'--seeds',str(seed),
                 '--budget',str(config['budget']),'--eval-games',str(config['eval_games']),
                 '--max-steps',str(config['max_steps']),'--output',str(out)]
        with (out/'run.log').open('w',encoding='utf8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/'episodes.csv').open(encoding='utf8',newline='') as handle:
            rows=list(csv.DictReader(handle))
        if key not in completed_runs(rows,config['eval_games'],config['budget']):
            raise RuntimeError(f'Incomplete worker {key}')
        return key,rows

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,key) for key in missing]
        for i,future in enumerate(as_completed(futures),1):
            key,rows=future.result();complete[key]=rows
            print(f'{i}/{len(missing)} completed: {key}',flush=True)
    original=folder/'episodes.csv'
    backup=folder/'episodes-before-parallel.csv'
    if not backup.exists():backup.write_bytes(original.read_bytes())
    temporary=folder/'episodes-merged.csv'
    with temporary.open('w',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for key in expected:writer.writerows(complete[key])
    temporary.replace(original)
    (folder/'execution.json').write_text(json.dumps(dict(workers=args.workers,source='experiments/parallel.py',
        reused_runs=len(expected)-len(missing),new_runs=len(missing)),indent=2),encoding='utf8')
    print('Merged all required results into results/main/episodes.csv',flush=True)


if __name__=='__main__':main()
