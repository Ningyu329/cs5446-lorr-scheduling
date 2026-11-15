#!/usr/bin/env python3
"""CS5446 follow-up studies: offline bandit audit, matched density contrasts,
   and a controlled room-topology pilot. Standard-library only.

Place the 'followup' folder in the CS5446-Team27 project root.
Example: python3 followup/study.py audit-bandit --runs results/runs/main
"""
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def mean_ci(xs):
    if not xs: return (float('nan'), float('nan'))
    m = statistics.mean(xs)
    if len(xs)<2: return (m, float('nan'))
    try:
        from scipy.stats import t
        crit = float(t.ppf(.975, len(xs)-1))
    except ImportError:
        # Conservative lookup for typical n in this exercise; normal approximation for n>=30.
        lookup = {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,
                  9:2.306,10:2.262,11:2.228,12:2.201,13:2.179,14:2.160,
                  15:2.145,16:2.131,17:2.120,18:2.110,19:2.101,20:2.093}
        crit=lookup.get(len(xs),2.05 if len(xs)>=21 else float('nan'))
    return m, crit*statistics.stdev(xs)/math.sqrt(len(xs))


def audit_bandit(args):
    runs=Path(args.runs)
    folders=[runs] if (runs/'stats.json.gz').exists() else sorted(runs.rglob('stats.json.gz'))
    if folders and folders[0]!=runs: folders=[p.parent for p in folders]
    rows=[]
    for folder in folders:
        try:
            with gzip.open(folder/'stats.json.gz','rt') as f: stats=json.load(f)
            if stats.get('config',{}).get('mode')!='BANDIT':continue
            with gzip.open(folder/'assigns.csv.gz','rt') as f: assigns=list(csv.DictReader(f))
        except (OSError,json.JSONDecodeError,KeyError):continue
        window=int(args.window)
        arm=stats.get('arm_pulls',[])
        completed=[]
        for r in assigns:
            try:
                dur=int(float(r['realised_ticks'])); st=int(float(r['t_assign']))
            except (ValueError,KeyError):continue
            if dur<0:continue  # Right-censored tasks excluded (not scored as short)
            sw=st//window; ew=(st+dur)//window
            if sw>=len(arm) or ew>=len(arm):continue
            completed.append((dur, ew>sw, arm[ew]!=arm[sw], ew-sw))
        if not completed:continue
        n=len(completed)
        rows.append({'run':folder.name, 'completed':n,
                     'cross_window_pct':100*sum(c[1] for c in completed)/n,
                     'different_arm_pct':100*sum(c[2] for c in completed)/n,
                     'median_task_ticks':statistics.median(c[0] for c in completed),
                     'median_windows_crossed':statistics.median(c[3] for c in completed),
                     'arms':len(arm)})
    if not rows:
        print('No BANDIT runs with stats.json.gz + assigns.csv.gz found at',runs)
        return
    dest=Path(args.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Runs audited:',len(rows),'CSV:',dest)
    for key in ('completed','cross_window_pct','different_arm_pct','median_task_ticks','median_windows_crossed'):
        print(f'  {key}: median across runs = {statistics.median(r[key] for r in rows):.2f}')
    print('Caution: completion-of-task proxy, while actual Bandit reward is per errand;')
    print('  observed delay does not by itself prove a causal credit-assignment failure.')


def analyze_criticality(args):
    d=Path(args.runs)
    dat={}
    for p in d.glob('*/summary.json'):
        with p.open() as f:s=json.load(f)
        if s.get('domain')!='warehouse' or s.get('sched') not in ('S0','S3'):continue
        if s.get('delay')!='base':continue
        k=(round(float(s['mult']),2),int(s['seed']))
        dat.setdefault(k,{})[s['sched']]=s
    effects={}
    for (mult,seed),pair in sorted(dat.items()):
        if 'S0' not in pair or 'S3' not in pair:continue
        a=pair['S0']['tasks_finished'];b=pair['S3']['tasks_finished']
        if not a:continue
        effects.setdefault(mult,{})[seed]=100*(b-a)/a
    print('Density follow-up, relative throughput S3 vs S0 (paired seeds):')
    for mult,v in sorted(effects.items()):
        mu,ci=mean_ci(list(v.values()))
        print(f' m={mult:g}: n={len(v)}, mean={mu:+.2f}% ± {ci:.2f} pp (95% t CI half-width)')
    if .5 in effects and 1.5 in effects:
        common=sorted(set(effects[.5]) & set(effects[1.5]))
        diffs=[effects[1.5][s]-effects[.5][s] for s in common]
        mu,ci=mean_ci(diffs)
        print(f' interaction (1.5× minus 0.5×): n={len(common)}, {mu:+.2f} ± {ci:.2f} percentage points')
    print('Exploratory if using previously examined seeds; independent held-out seeds are confirmatory.')


def make_bypass_map(source,dest):
    lines=source.read_text().splitlines()
    if len(lines)<68 or lines[0].strip()!='type octile':
        raise RuntimeError('Unexpected room map format')
    height=int(lines[1].split()[1]);width=int(lines[2].split()[1]);grid=[list(x) for x in lines[4:4+height]]
    if (height,width)!=(64,64):raise RuntimeError('This controlled pilot expects 64x64 room map')
    edits=[]
    # One extra door in each of 4 stretches of each of 3 internal horizontal and vertical walls.
    # Only wall cells with free neighbors on both sides are opened; no free cell is closed.
    for r in (16,32,48):
        for target_col in (8,24,40,56):
            for c in sorted(range(target_col-4,target_col+5),key=lambda x:abs(x-target_col)):
                if 1<=c<63 and grid[r][c]=='@' and grid[r-1][c]=='.' and grid[r+1][c]=='.':
                    grid[r][c]='.'; edits.append((r,c));break
    for c in (16,32,48):
        for target_row in (8,24,40,56):
            for r in sorted(range(target_row-4,target_row+5),key=lambda x:abs(x-target_row)):
                if 1<=r<63 and grid[r][c]=='@' and grid[r][c-1]=='.' and grid[r][c+1]=='.':
                    grid[r][c]='.';edits.append((r,c));break
    if len(edits)<10:raise RuntimeError('Too few safe door openings; inspect original map')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text('\n'.join(lines[:4]+[''.join(x) for x in grid])+'\n')
    return edits


def bottleneck(args):
    sys.path.insert(0,str(ROOT/'scripts'))
    from make_instances import make_instance
    binary=ROOT/'third_party/Start-Kit/build/lifelong'
    source=ROOT/'data/instances/room.domain/maps/room-64-64-16.map'
    if not source.is_file():
        source=ROOT/'third_party/Start-Kit/example_problems/room.domain/maps/room-64-64-16.map'
    if not source.is_file():raise RuntimeError(f'Original map missing: {source}; run scripts/build_startkit.sh')
    if not binary.is_file() and not args.dry:raise RuntimeError(f'Simulator missing: {binary}')
    bypass=ROOT/'data/followup_maps/room-64-64-16-bypass.map'
    doors=make_bypass_map(source,bypass)
    print(f'Map prepared, {len(doors)} NEW passages. Original map unchanged.')
    outbase=ROOT/'results/followup_bottleneck'
    for seed in args.seeds:
        orig=Path(make_instance('room',1.0,'base',seed))
        with orig.open() as f:j=json.load(f)
        alt=dict(j);alt['mapFile']=os.path.relpath(bypass,orig.parent)
        alternative=orig.with_name(orig.stem+'-bypass.json')
        alternative.write_text(json.dumps(alt,indent=2)+'\n')
        for topo,inp in [('original',orig),('bypass',alternative)]:
            for sched,lam in [('S0',0),('S1',2)]:
                d=outbase/topo/f'seed-{seed}'/sched
                d.mkdir(parents=True,exist_ok=True)
                result=d/'output.json'; log=d/'run.log';summary=d/'summary.json'
                if summary.exists() and not args.force:
                    print('SKIP',topo,seed,sched);continue
                cmd=[str(binary),'-i',str(inp),'-o',str(result),'-s','3000','-c','2','-d','3','-l',str(log)]
                if args.dry:
                    print('DRY',topo,seed,sched,'input='+str(inp));continue
                env=dict(os.environ,LORR_SCHED=sched,LORR_LAMBDA=str(lam),LORR_GAMMA='0',LORR_SEED=str(seed),LORR_HORIZON='3000')
                print('RUN',topo,seed,sched,flush=True)
                with (d/'stdout.log').open('w') as f:
                    p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                if p.returncode:raise RuntimeError(f'Simulator exit={p.returncode}. See {d}/stdout.log')
                with result.open() as f:r=json.load(f)
                out={'topology':topo,'seed':seed,'sched':sched,'tasks_finished':r['numTaskFinished'],
                     'planner_errors':r['numPlannerErrors'],'schedule_errors':r['numScheduleErrors'],
                     'entry_timeouts':r['numEntryTimeouts'],'instance':str(inp.relative_to(ROOT))}
                summary.write_text(json.dumps(out,indent=2)+'\n')
                print('TASKS',out['tasks_finished'],'ERRORS',out['planner_errors'],out['schedule_errors'],out['entry_timeouts'])
    if not args.dry: analyze_bottleneck(argparse.Namespace(runs=str(outbase)))


def analyze_bottleneck(args):
    base=Path(args.runs);d={}
    for p in base.glob('*/*/*/summary.json'):
        with p.open() as f:s=json.load(f)
        d.setdefault((s['topology'],int(s['seed'])),{})[s['sched']]=s['tasks_finished']
    effects={}
    for (topo,seed),pair in d.items():
        if 'S0' in pair and 'S1' in pair and pair['S0']:
            effects.setdefault(topo,{})[seed]=100*(pair['S1']-pair['S0'])/pair['S0']
    for topo,m in sorted(effects.items()):
        avg,ci=mean_ci(list(m.values()))
        print(f'{topo}: n={len(m)} S1 vs S0 = {avg:+.2f}% ± {ci:.2f}pp')
    if 'original' in effects and 'bypass' in effects:
        common=set(effects['original'])&set(effects['bypass'])
        diff=[effects['bypass'][i]-effects['original'][i] for i in sorted(common)]
        avg,ci=mean_ci(diff)
        print(f'Topology interaction: n={len(diff)}, {avg:+.2f} ± {ci:.2f} percentage points')
    print('Important: changed topology also changes shortest paths. This is a mechanism probe, not a proof of causality.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='cmd',required=True)
    s=sp.add_parser('audit-bandit');s.add_argument('--runs',default='results/runs/main');s.add_argument('--window',type=int,default=100);s.add_argument('--out',default='results/followup_bandit_audit.csv');s.set_defaults(func=audit_bandit)
    s=sp.add_parser('criticality-report');s.add_argument('--runs',default='results/runs/followup_criticality');s.set_defaults(func=analyze_criticality)
    s=sp.add_parser('bottleneck');s.add_argument('--seeds',nargs='+',type=int,default=[20,21,22]);s.add_argument('--dry',action='store_true');s.add_argument('--force',action='store_true');s.set_defaults(func=bottleneck)
    s=sp.add_parser('bottleneck-report');s.add_argument('--runs',default='results/followup_bottleneck');s.set_defaults(func=analyze_bottleneck)
    a=p.parse_args();a.func(a)

if __name__=='__main__':main()
