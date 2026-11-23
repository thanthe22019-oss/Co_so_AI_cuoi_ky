"""Run Berkeley autograders, save logs and fail on missing required points."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECTS={1:'search',2:'multiagent',3:'reinforcement',5:'machinelearning'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--projects',nargs='+',type=int,choices=PROJECTS,default=[1,2,3,5])
    args=parser.parse_args()
    (ROOT/'logs').mkdir(exist_ok=True)
    passed=True
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUTF8='1')
    for n in args.projects:
        executable=ROOT/'.venv/Scripts/python.exe' if n==5 else Path(sys.executable)
        started=time.perf_counter()
        result=subprocess.run([str(executable),'-X','utf8','autograder.py','--no-graphics'],
                              cwd=ROOT/f'projects/project{n}'/PROJECTS[n],capture_output=True,env=env)
        output=(result.stdout+result.stderr).decode('utf8',errors='replace')
        (ROOT/f'logs/project{n}.txt').write_text(output,encoding='utf8')
        total=re.findall(r'Total:\s*(\d+)/(\d+)',output)
        ok=result.returncode==0 and bool(total) and int(total[-1][0])>=int(total[-1][1]) and '*** FAIL' not in output
        passed &= ok
        print(f'Project {n}: {total[-1] if total else "no score"}; {time.perf_counter()-started:.1f}s; {"PASS" if ok else "FAIL"}',flush=True)
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
