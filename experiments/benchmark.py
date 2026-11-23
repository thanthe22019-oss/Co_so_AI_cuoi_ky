"""Reproducible RL experiments using the unmodified Berkeley Pac-Man rules.

Run from workspace root: python experiments/benchmark.py --help
"""
import argparse
from collections import deque, defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'projects/project3/reinforcement'))
from layout import Layout
from pacman import GameState
from game import Actions, Directions
from ghostAgents import RandomGhost
from qlearningAgents import PacmanQAgent, ApproximateQAgent
from featureExtractors import SimpleExtractor, closestFood
import util

spec = importlib.util.spec_from_file_location('project2_agents', ROOT / 'projects/project2/multiagent/multiAgents.py')
multi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi)


def validate_layout(rows):
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError('Layout must be rectangular')
    if any(c != '%' for c in rows[0] + rows[-1]) or any(row[0] != '%' or row[-1] != '%' for row in rows):
        raise ValueError('Layout must have boundary walls')
    starts = [(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == 'P']
    food = {(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == '.'}
    if len(starts) != 1 or not food:
        raise ValueError('Layout needs one Pac-Man and at least one food')
    reached = {starts[0]}
    queue = deque(starts)
    while queue:
        x, y = queue.popleft()
        for point in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            nx, ny = point
            if rows[ny][nx] != '%' and point not in reached:
                reached.add(point)
                queue.append(point)
    walkable = {(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c != '%'}
    if reached != walkable:
        raise ValueError('All walkable cells, including food, must be connected')


def make_layout(size, seed):
    if size < 9 or size % 2 == 0:
        raise ValueError('Size must be odd and at least 9')
    rng = random.Random(seed)
    grid = [['%'] * size for _ in range(size)]
    grid[1][1] = ' '
    visited = {(1, 1)}
    stack = [(1, 1)]
    while stack:
        x, y = stack[-1]
        choices = [(nx, ny) for nx, ny in ((x+2,y),(x-2,y),(x,y+2),(x,y-2))
                   if 0 < nx < size-1 and 0 < ny < size-1 and (nx,ny) not in visited]
        if not choices:
            stack.pop()
            continue
        nx, ny = rng.choice(choices)
        grid[(y+ny)//2][(x+nx)//2] = ' '
        grid[ny][nx] = ' '
        visited.add((nx, ny))
        stack.append((nx, ny))
    # Add loops without creating disconnected cells.
    for y in range(1, size-1):
        for x in range(1, size-1):
            if grid[y][x] == '%' and rng.random() < .12:
                if (grid[y][x-1] == grid[y][x+1] == ' ' or
                        grid[y-1][x] == grid[y+1][x] == ' '):
                    grid[y][x] = ' '
    cells = [(x,y) for y in range(1,size-1) for x in range(1,size-1) if grid[y][x] == ' ']
    for x,y in cells:
        if rng.random() < .12:
            grid[y][x] = '.'
    grid[1][1] = 'P'
    grid[size-2][size-2] = 'G'
    for x,y in ((size-2,1),(1,size-2)):
        grid[y][x] = 'o'
    # A guaranteed dot also covers unlikely empty random samples.
    grid[1][2] = '.'
    rows = [''.join(row) for row in grid]
    validate_layout(rows)
    return rows


class ImprovedExtractor(SimpleExtractor):
    def __init__(self, danger=True):
        self.danger = danger

    def getFeatures(self, state, action):
        food, walls = state.getFood(), state.getWalls()
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        pos = int(x+dx), int(y+dy)
        active = [g for g in state.getGhostStates() if g.scaredTimer == 0]
        danger1 = sum(pos in Actions.getLegalNeighbors(g.getPosition(), walls) for g in active)
        features = util.Counter({'bias': .1, 'stop': .1 * (action == Directions.STOP)})
        features['ghost-one-step'] = danger1 / 10
        features['eats-food'] = .1 * (food[pos[0]][pos[1]] and not danger1)
        distance = closestFood(pos, food, walls)
        if distance is not None:
            features['closest-food'] = min(distance / (walls.width + walls.height), 1.0) / 10
        if self.danger:
            near = 0
            for ghost in active:
                one = Actions.getLegalNeighbors(ghost.getPosition(), walls)
                two = {p for q in one for p in Actions.getLegalNeighbors(q, walls)}
                near += pos in two
            features['ghost-two-steps'] = near / 10
            exits = [p for p in Actions.getLegalNeighbors(pos, walls) if p != pos]
            features['danger-dead-end'] = .1 * (len(exits) <= 1 and near > 0)
        features['eats-capsule'] = .1 * (pos in state.getCapsules())
        scared = [util.manhattanDistance(pos, g.getPosition()) for g in state.getGhostStates()
                  if g.scaredTimer > util.manhattanDistance(pos, g.getPosition())]
        features['scared-proximity'] = .1 / (1 + min(scared)) if scared else 0
        return features


def make_agent(kind):
    args = dict(epsilon=.1, alpha=.2, gamma=.9, numTraining=10**9)
    if kind == 'reflex':
        return multi.ReflexAgent()
    if kind == 'tabular':
        return PacmanQAgent(**args)
    if kind not in ('simple', 'improved', 'no-danger'):
        raise ValueError(kind)
    agent = ApproximateQAgent(extractor='SimpleExtractor', **args)
    if kind != 'simple':
        agent.featExtractor = ImprovedExtractor(danger=kind != 'no-danger')
    return agent


def episode(agent, layout, seed, max_steps, train=False):
    """One update per Pac-Man move plus the following ghost moves.

    A step limit is a truncation, not an artificial terminal state: bootstrap
    from the final nonterminal state when training. Evaluation never calls update.
    """
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    random.seed(seed)
    state = GameState()
    state.initialize(layout, 1)
    initial_food = state.getNumFood()
    learning = isinstance(agent, PacmanQAgent)
    old_epsilon, old_alpha = (agent.epsilon, agent.alpha) if learning else (None, None)
    if learning and not train:
        agent.epsilon, agent.alpha = 0., 0.
    decision_seconds = 0.
    started = time.perf_counter()
    ghost = RandomGhost(1)
    try:
        for steps in range(1, max_steps+1):
            before = time.perf_counter()
            action = agent.getAction(state)
            decision_seconds += time.perf_counter() - before
            successor = state.generateSuccessor(0, action)
            if not successor.isWin() and not successor.isLose():
                successor = successor.generateSuccessor(1, ghost.getAction(successor))
            if train and learning:
                agent.update(state, action, successor, successor.getScore() - state.getScore())
            state = successor
            # Berkeley keeps a process-global set for search expansion counting.
            GameState.explored.clear()
            if state.isWin() or state.isLose():
                break
    finally:
        if learning:
            agent.epsilon, agent.alpha = old_epsilon, old_alpha
    return dict(score=state.getScore(), steps=steps,
                outcome='win' if state.isWin() else 'lose' if state.isLose() else 'timeout',
                food_fraction=(initial_food-state.getNumFood())/initial_food,
                seconds=time.perf_counter()-started, decision_seconds=decision_seconds)


def completed_runs(rows, eval_games, budget):
    groups = defaultdict(list)
    for row in rows:
        if row['phase'] == 'eval':
            groups[int(row['size']),row['agent'],int(row['seed'])].append(row)
    return {key for key,games in groups.items() if len(games)==eval_games
            and (key[1]=='reflex' or all(int(g['training_steps'])==budget for g in games))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[15,25,35])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
    parser.add_argument('--agents', nargs='+', default=['reflex','tabular','simple','improved','no-danger'])
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--eval-games', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--output', default='results/main')
    parser.add_argument('--resume', action='store_true', help='Keep complete runs with matching configuration; retry partial runs')
    args = parser.parse_args()
    if min(args.budget,args.eval_games,args.max_steps) <= 0:
        parser.error('budgets, games and step limits must be positive')
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    config = {k:v for k,v in vars(args).items() if k!='resume'}
    retained=[]
    completed=set()
    if args.resume:
        old_config=json.loads((out/'config.json').read_text(encoding='utf8'))
        if old_config != config:
            parser.error('Resume configuration must match the original config.json')
        with (out/'episodes.csv').open(encoding='utf8',newline='') as handle:
            old_rows=list(csv.DictReader(handle))
        completed=completed_runs(old_rows,args.eval_games,args.budget)
        retained=[r for r in old_rows if (int(r['size']),r['agent'],int(r['seed'])) in completed]
        backup=out/'episodes-before-resume.csv'
        if not backup.exists():
            backup.write_bytes((out/'episodes.csv').read_bytes())
        print(f'Resuming {len(completed)} complete runs; incomplete runs retried',flush=True)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    layout_dir = ROOT / 'experiments/layouts'
    layout_dir.mkdir(parents=True,exist_ok=True)
    fields = ['size','agent','seed','phase','training_steps','game','score','steps','outcome',
              'food_fraction','seconds','decision_seconds','model_entries','train_seconds']
    with (out/'episodes.csv').open('w',newline='',encoding='utf8') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained)
        handle.flush()
        for size in args.sizes:
            rows = make_layout(size, 2026)
            (layout_dir/f'extended{size}.lay').write_text('\n'.join(rows)+'\n',encoding='utf8')
            layout = Layout(rows)
            for kind in args.agents:
                for seed in args.seeds[:1] if kind=='reflex' else args.seeds:
                    if (size,kind,seed) in completed:
                        continue
                    random.seed(seed)
                    agent = make_agent(kind)
                    used = 0
                    game = 0
                    started = time.perf_counter()
                    if kind != 'reflex':
                        while used < args.budget:
                            result = episode(agent,layout,seed*100000+game,
                                             min(args.max_steps,args.budget-used),train=True)
                            used += result['steps']
                            writer.writerow(dict(size=size,agent=kind,seed=seed,phase='train',
                                                 training_steps=used,game=game,**result))
                            game += 1
                    train_seconds = time.perf_counter()-started
                    entries = len(agent.weights) if hasattr(agent,'weights') else len(agent.qValues) if hasattr(agent,'qValues') else 0
                    for game in range(args.eval_games):
                        # Held-out stream, shared across methods and training seeds.
                        result = episode(agent,layout,10000000+game,args.max_steps)
                        writer.writerow(dict(size=size,agent=kind,seed=seed,phase='eval',training_steps=used,
                                             game=game,model_entries=entries,train_seconds=train_seconds,**result))
                    handle.flush()
                    print(f'{size} {kind} seed={seed}: trained {used} steps, evaluated {args.eval_games} games',flush=True)
    print(f'Results: {out}',flush=True)


if __name__ == '__main__':
    main()
