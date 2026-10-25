"""Offline, oracle-based evaluation of learning mechanisms (spec R2).

The host supplies an argv list (or loads one with ``load_agent_argv``), never a
shell command. ``load_tasks(evals / 'tasks')`` verifies ``evals/tasks.lock`` and
resolves workspace/oracle paths relative to each task file. Results live exclusively
in Learning's durable eval_results table. Positive decisions are recommendations
for a human; only negative decisions change a mechanism switch automatically.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid

from .learning import MECHANISMS, TIERS, error_signature


def _argv(value):
    if (not isinstance(value, list) or not value or not all(isinstance(arg, str) for arg in value)
            or not value[0] or any('\0' in arg for arg in value)):
        raise ValueError('argv must be a nonempty JSON list of strings, without NUL characters')
    return list(value)


def load_agent_argv(path):
    """Read a JSON argv template list; relative executable/script paths use the workspace."""
    return _argv(json.loads(Path(path).read_text(encoding='utf-8')))


def _task(value, base=None):
    if not isinstance(value, dict):
        raise ValueError('Task must be a JSON object')
    task = dict(value)
    if not isinstance(task.get('id'), str) or not task['id'].strip():
        raise ValueError('Task requires a nonempty string id')
    if not isinstance(task.get('prompt'), str) or '\0' in task['prompt']:
        raise ValueError('Task requires a string prompt without NUL characters')
    check = task.get('check')
    if not isinstance(check, dict) or type(check.get('expect_exit', 0)) is not int:
        raise ValueError('Task requires check.argv and an integer check.expect_exit')
    task['check'] = {**check, 'argv': _argv(check.get('argv')), 'expect_exit': check.get('expect_exit', 0)}
    if not isinstance(task.get('applies_to', ''), str):
        raise ValueError('Task applies_to must be free text')
    if task.get('oracle') is None:
        raise ValueError('Task requires an oracle directory')
    for field in ('workspace', 'oracle'):
        value = task.get(field)
        if value is not None:
            if not isinstance(value, (str, Path)) or not str(value) or '\0' in str(value):
                raise ValueError(f'Task {field} must name a directory')
            path = Path(value)
            # Keep symlink components visible until the input validation below.
            task[field] = str(((base / path) if base is not None else path).absolute())
    return task


def _safe_path(path, root):
    """Check components before resolving, so even in-root symlinks are refused."""
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise ValueError(f'Eval input is outside tasks root: {path}') from exc
    current = root
    try:
        for part in parts:
            current = current / part
            mode = current.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f'Eval inputs must be regular files or directories, without symlinks: {current}')
            if not current.resolve().is_relative_to(root):
                raise ValueError(f'Eval input is outside tasks root: {path}')
        return path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f'Cannot read eval input: {path}') from exc


def _input_files(directory, root):
    """Validate a complete tree without following links, and hash every file."""
    directory = _safe_path(directory, root)
    if not directory.is_dir():
        raise ValueError(f'Eval input is not a directory: {directory}')
    files = {}

    def visit(path):
        path = _safe_path(path, root)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child)
        else:
            # Mode matters too: an input turning executable changes what a task does.
            files[path.relative_to(root).as_posix()] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                                        + f':{stat.S_IMODE(path.stat().st_mode):o}')

    visit(directory)
    return files


def _read_task(path, root):
    path = _safe_path(path, root)
    if not path.is_file():
        raise ValueError(f'Eval task must be a regular JSON file: {path}')
    raw = path.read_bytes()
    task = _task(json.loads(raw), path.parent)
    inputs = {}
    for field in ('workspace', 'oracle'):
        if task.get(field) is not None:
            inputs.update(_input_files(Path(task[field]), root))
            task[field] = str(Path(task[field]).resolve())
    manifest = json.dumps(sorted(inputs.items()), separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    task['task_hash'] = hashlib.sha256(raw + b'\0' + manifest).hexdigest()
    task['_task_file'] = str(path)
    return task


def _verify_task(task):
    """Bind execution to the loaded JSON and input bytes, including after the agent."""
    path = task.get('_task_file')
    if not isinstance(path, str):
        raise ValueError('run_batch requires tasks from load_tasks')
    path = Path(path)
    if _read_task(path, path.parent.parent) != task:
        raise ValueError('Eval task or inputs changed since load_tasks; review and load_tasks again')


def _copy_input(source, destination):
    # Never dereference a link introduced between validation and copying.
    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)
    _input_files(destination, destination)


def _read_tasks(directory):
    directory = Path(directory).absolute()
    root = directory.parent.resolve(strict=True)
    directory = _safe_path(directory, root)
    if not directory.is_dir():
        raise ValueError(f'Eval tasks path is not a directory: {directory}')
    tasks, hashes = [], {}
    for path in sorted(directory.glob('*.json')):
        task = _read_task(path, root)
        ident = task['id']
        if ident in hashes:
            raise ValueError(f'Duplicate eval task id: {ident}')
        hashes[ident] = task['task_hash']
        tasks.append(task)
    if not tasks:
        raise ValueError(f'No eval task JSON files in {directory}')
    return sorted(tasks, key=lambda task: task['id']), hashes


def lock_tasks(directory):
    """Freeze exact JSON bytes and every workspace/oracle input in tasks.lock."""
    _, hashes = _read_tasks(directory)
    lock = Path(directory).parent / 'tasks.lock'
    # Replace atomically so an interrupted lock operation does not leave half JSON.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=lock.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(hashes, sort_keys=True, indent=2) + '\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(lock)
    finally:
        temporary.unlink(missing_ok=True)
    return hashes


def load_tasks(directory):
    """Load task dicts for run_batch, bound to exact JSON and input file bytes.

    Missing/stale locks, including added/removed tasks or inputs, are refused.
    Changing a loaded task requires an explicit edit, re-lock and reload.
    """
    lock = Path(directory).parent / 'tasks.lock'
    try:
        locked = json.loads(lock.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read tasks.lock; run lock_tasks explicitly: {lock}') from exc
    tasks, hashes = _read_tasks(directory)
    if locked != hashes:
        raise ValueError('Eval tasks changed since tasks.lock; review and lock_tasks explicitly')
    return tasks


def excluded_sources(tasks):
    return {f"eval:{task['id']}" for task in tasks}


def _substitute(argv, prompt, workspace):
    values = {'prompt': prompt, 'workspace': str(workspace)}
    return [re.sub(r'\{(prompt|workspace)\}', lambda match: values[match[1]], arg) for arg in argv]


def _text(value):
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''


def _run(argv, workspace, timeout):
    try:
        done = subprocess.run(argv, cwd=workspace, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, encoding='utf-8', errors='replace',
                              timeout=timeout, check=False, shell=False)
        return dict(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr, timed_out=False)
    except subprocess.TimeoutExpired as exc:
        return dict(returncode=None, stdout=_text(exc.stdout), stderr=_text(exc.stderr) + '\ncommand timed out',
                    timed_out=True)
    except OSError as exc:
        return dict(returncode=None, stdout='', stderr=str(exc), timed_out=False)


def _tokens(stdout):
    values = set()
    for line in stdout.splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if not isinstance(data, dict) or 'usage' not in data:
            continue
        usage = data['usage']
        value = usage.get('tokens') if isinstance(usage, dict) else None
        if type(value) is not int or value < 0:
            return None
        values.add(value)
    return values.pop() if len(values) == 1 else None


def run_batch(tasks, *, config, tier, agent_model, agent_argv, learning, skill_pack=None,
              after_run=None, timeout=600) -> str:
    """Run loaded, locked tasks with separate disposable agent and oracle dirs.

    The timeout applies separately to each agent and check. The check alone
    determines success, even if an agent exits nonzero. Missing, malformed or
    contradictory usage leaves tokens unknown (NULL).
    C's hook receives (task, result) after the row is committed, while the workspace
    still exists. The result includes process diagnostics; only counts, timing
    and the normalized failure signature are persisted. Hook errors propagate.
    """
    if config not in ('A', 'B', 'C') or tier not in TIERS:
        raise ValueError('Evaluation requires config A|B|C and tier weak|frontier')
    if not isinstance(agent_model, str) or not agent_model.strip():
        raise ValueError('Evaluation requires an agent_model')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be a positive finite number')
    argv = _argv(agent_argv)
    tasks = sorted((_task(task) for task in tasks), key=lambda task: task['id'])
    if not tasks or len({task['id'] for task in tasks}) != len(tasks):
        raise ValueError('Evaluation needs nonempty tasks with unique ids')
    for task in tasks:
        _verify_task(task)

    batch, started = uuid.uuid4().hex, time.time()
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix='second-brain-eval-') as directory:
            workspace = Path(directory)
            _verify_task(task)
            if task.get('workspace') is not None:
                _copy_input(task['workspace'], workspace)
            prompt = task['prompt']
            if config != 'A' and skill_pack is not None:
                pack = skill_pack(task)
                if not isinstance(pack, str):
                    raise ValueError('skill_pack must return text')
                if pack:
                    prompt = pack + '\n\n' + prompt
            before = time.monotonic()
            agent = _run(_substitute(argv, prompt, workspace), workspace, timeout)
            _verify_task(task)
            # Create and populate the checker directory only after the agent exits.
            with tempfile.TemporaryDirectory(prefix='second-brain-oracle-') as oracle:
                _copy_input(task['oracle'], Path(oracle))
                check = _run(_substitute(task['check']['argv'], prompt, workspace), oracle, timeout)
            seconds = time.monotonic() - before
            success = check['returncode'] == task['check']['expect_exit']
            tokens = _tokens(agent['stdout'])
            signature = '' if success else error_signature(check['stderr'] + check['stdout'])
            with learning.db:
                learning.db.execute('''INSERT INTO eval_results
                    (batch, task_id, tier, config, agent_model, success, tokens, seconds, signature, ts, task_hash)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                    (batch, task['id'], tier, config, agent_model, int(success), tokens, seconds, signature, started,
                     task['task_hash']))
            if config == 'C' and after_run is not None:
                result = dict(batch=batch, task_id=task['id'], tier=tier, config=config, agent_model=agent_model,
                              success=success, tokens=tokens, seconds=seconds, signature=signature,
                              workspace=str(workspace), agent=agent, check=check, task_hash=task['task_hash'])
                after_run(task, result)
    return batch


def _rows(learning, batch):
    rows = [dict(row) for row in learning.db.execute(
        'SELECT rowid AS _rowid, * FROM eval_results WHERE batch=? ORDER BY task_id, rowid', (batch,))]
    if not rows:
        raise ValueError(f'Unknown or empty eval batch: {batch}')
    if len({row['task_id'] for row in rows}) != len(rows):
        raise ValueError(f'Eval batch has duplicate task results: {batch}')
    return rows


def _total(rows, field):
    values = [row[field] for row in rows]
    return None if any(value is None for value in values) else sum(values)


def _cost(rows):
    successes = sum(bool(row['success']) for row in rows)
    totals = {field: _total(rows, field) for field in ('tokens', 'seconds')}
    return {field: total / successes if successes and total is not None else None for field, total in totals.items()}


def metrics(learning, batch_id):
    rows = _rows(learning, batch_id)
    started, first = min(row['ts'] for row in rows), min(row['_rowid'] for row in rows)
    previous = {(row['task_id'], row['signature']) for row in learning.db.execute('''
        SELECT e.task_id, e.signature FROM eval_results e
        JOIN (SELECT batch, min(ts) AS started, min(rowid) AS first_row FROM eval_results GROUP BY batch) b
          ON e.batch=b.batch
        WHERE e.success=0 AND e.batch<>?
          AND (b.started<? OR (b.started=? AND b.first_row<?))''', (batch_id, started, started, first))}
    failures = [row for row in rows if not row['success']]
    repeated = sum(bool(row['signature']) and (row['task_id'], row['signature']) in previous for row in failures)
    successes = len(rows) - len(failures)
    return dict(batch_id=batch_id, tasks=len(rows), successes=successes, failures=len(failures),
                success_rate=successes / len(rows), success_first_try=successes / len(rows),
                tokens=_total(rows, 'tokens'), seconds=_total(rows, 'seconds'),
                cost_per_success=_cost(rows), repeated_error_rate=repeated / len(failures) if failures else 0.0)


def _percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def decide(learning, baseline_batch, candidate_batch, *, tier, mechanism, min_gain=0.0,
           max_cost_increase=0.2, max_time_increase=0.2, resamples=2000, seed=0):
    """Paired bootstrap (5th/95th percentiles), with cost on the same paired tasks.

    No baseline successes means its cost/success is undefined, so a finite
    candidate cost passes the cost gate. Zero baseline cost permits only zero
    candidate cost, for tokens and seconds. Unknown usage never permits enabling.
    Batches must use one model/tier, identical locked tasks, and configs A/C for
    skills or B/C for reflection. Invalid comparisons never change a switch.
    """
    if tier not in TIERS or mechanism not in MECHANISMS:
        raise ValueError('Unknown tier or mechanism')
    if type(resamples) is not int or resamples < 1:
        raise ValueError('resamples must be a positive integer')
    if (not math.isfinite(min_gain) or not math.isfinite(max_cost_increase) or max_cost_increase < 0
            or not math.isfinite(max_time_increase) or max_time_increase < 0):
        raise ValueError('Decision thresholds must be finite, with nonnegative max_cost_increase and max_time_increase')
    baseline, candidate = _rows(learning, baseline_batch), _rows(learning, candidate_batch)
    if any(row['tier'] != tier for row in baseline + candidate):
        raise ValueError('Both batches must belong to the requested tier')
    models = {row['agent_model'] for row in baseline + candidate}
    if len(models) != 1 or any(not isinstance(model, str) or not model.strip() for model in models):
        raise ValueError('Both batches must use the same agent_model')
    baseline_config = 'A' if mechanism == 'skills' else 'B'
    if (any(row['config'] != baseline_config for row in baseline)
            or any(row['config'] != 'C' for row in candidate)):
        raise ValueError(f'{mechanism} requires baseline config {baseline_config} and candidate config C')
    baseline = {row['task_id']: row for row in baseline}
    candidate = {row['task_id']: row for row in candidate}
    if baseline.keys() != candidate.keys():
        raise ValueError('Both batches must contain the same task ids')
    paired = sorted(baseline)
    for key in paired:
        task_hash = baseline[key]['task_hash']
        if (not isinstance(task_hash, str) or not task_hash.strip()
                or candidate[key]['task_hash'] != task_hash):
            raise ValueError(f'Both batches require identical task_hash for task {key}')
    costs = {'baseline': _cost([baseline[key] for key in paired]),
             'candidate': _cost([candidate[key] for key in paired])}
    differences = [int(bool(candidate[key]['success'])) - int(bool(baseline[key]['success'])) for key in paired]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(differences) for _ in paired) / len(paired) for _ in range(resamples))
    result = dict(gain=sum(differences) / len(paired), ci_low=_percentile(samples, 0.05),
                  ci_high=_percentile(samples, 0.95), costs=costs, paired_tasks=len(paired))
    gates = {}
    for field, increase in (('tokens', max_cost_increase), ('seconds', max_time_increase)):
        old, new = costs['baseline'][field], costs['candidate'][field]
        known = all(row[field] is not None for key in paired for row in (baseline[key], candidate[key]))
        gates[field] = known and new is not None and (old is None or new <= old * (1 + increase))
    if result['ci_low'] > min_gain and all(gates.values()):
        return {**result, 'decision': 'recommend_enable', 'recommend': 'enable',
                'reason': '90% paired bootstrap lower bound exceeds min_gain and token/time costs are within budget; human approval required'}
    if result['ci_low'] <= min_gain:
        reason = '90% paired bootstrap lower bound does not exceed min_gain'
    elif any(row['tokens'] is None for key in paired for row in (baseline[key], candidate[key])):
        reason = 'Token usage is unknown; cost per success cannot be established'
    elif not gates['tokens']:
        reason = 'Token cost per success exceeds the allowed increase'
    else:
        reason = 'Time cost per success is unknown or exceeds the allowed increase'
    learning.set_mechanism(tier, mechanism, False, reason, 'eval')
    return {**result, 'decision': 'disable', 'reason': reason}
