"""Offline, oracle-based evaluation of learning mechanisms (spec R2).

The host supplies an argv list (or loads one with ``load_agent_argv``), never a
shell command. ``load_tasks(evals / 'tasks')`` verifies ``evals/tasks.lock`` and
resolves workspace paths relative to each task file. Results live exclusively
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
    workspace = task.get('workspace')
    if workspace is not None:
        if not isinstance(workspace, (str, Path)) or not str(workspace):
            raise ValueError('Task workspace must name a directory')
        path = Path(workspace)
        task['workspace'] = str(((base / path) if base is not None else path).resolve())
    return task


def _read_tasks(directory):
    directory = Path(directory)
    tasks, hashes = [], {}
    for path in sorted(directory.glob('*.json')):
        raw = path.read_bytes()
        task = _task(json.loads(raw), path.parent)
        ident = task['id']
        if ident in hashes:
            raise ValueError(f'Duplicate eval task id: {ident}')
        hashes[ident] = hashlib.sha256(raw).hexdigest()
        tasks.append(task)
    if not tasks:
        raise ValueError(f'No eval task JSON files in {directory}')
    return sorted(tasks, key=lambda task: task['id']), hashes


def lock_tasks(directory):
    """Explicitly freeze task JSON bytes in ``directory.parent / 'tasks.lock'``.

    Added/deleted tasks are changes too. Workspace contents are copied inputs;
    this lock covers the JSON task definitions, as specified by R2.
    """
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
    """Refuse a missing/stale lock, including added, removed or renamed task IDs."""
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
    tokens = 0
    for line in stdout.splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        usage = data.get('usage') if isinstance(data, dict) else None
        value = usage.get('tokens') if isinstance(usage, dict) else None
        if type(value) is int and value >= 0:
            tokens = value
    return tokens


def run_batch(tasks, *, config, tier, agent_model, agent_argv, learning, skill_pack=None,
              after_run=None, timeout=600) -> str:
    """Run each agent once and then its oracle in a fresh, disposable workspace.

    The timeout applies separately to each agent and check. The check alone
    determines success, even if an agent exits nonzero. Missing usage is zero
    tokens; the last valid usage line is treated as the cumulative total.
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
        if task.get('workspace') is not None and not Path(task['workspace']).is_dir():
            raise ValueError(f"Task workspace is not a directory: {task['workspace']}")

    batch, started = uuid.uuid4().hex, time.time()
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix='second-brain-eval-') as directory:
            workspace = Path(directory)
            if task.get('workspace') is not None:
                shutil.copytree(task['workspace'], workspace, dirs_exist_ok=True)
            prompt = task['prompt']
            if config != 'A' and skill_pack is not None:
                pack = skill_pack(task)
                if not isinstance(pack, str):
                    raise ValueError('skill_pack must return text')
                if pack:
                    prompt = pack + '\n\n' + prompt
            before = time.monotonic()
            agent = _run(_substitute(argv, prompt, workspace), workspace, timeout)
            check = _run(task['check']['argv'], workspace, timeout)
            seconds = time.monotonic() - before
            success = check['returncode'] == task['check']['expect_exit']
            tokens = _tokens(agent['stdout'])
            signature = '' if success else error_signature(check['stderr'] + check['stdout'])
            with learning.db:
                learning.db.execute('''INSERT INTO eval_results
                    (batch, task_id, tier, config, agent_model, success, tokens, seconds, signature, ts)
                    VALUES (?,?,?,?,?,?,?,?,?,?)''',
                    (batch, task['id'], tier, config, agent_model, int(success), tokens, seconds, signature, started))
            if config == 'C' and after_run is not None:
                result = dict(batch=batch, task_id=task['id'], tier=tier, config=config, agent_model=agent_model,
                              success=success, tokens=tokens, seconds=seconds, signature=signature,
                              workspace=str(workspace), agent=agent, check=check)
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


def _cost(rows):
    successes = sum(bool(row['success']) for row in rows)
    return {field: sum(row[field] for row in rows) / successes if successes else None for field in ('tokens', 'seconds')}


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
                tokens=sum(row['tokens'] for row in rows), seconds=sum(row['seconds'] for row in rows),
                cost_per_success=_cost(rows), repeated_error_rate=repeated / len(failures) if failures else 0.0)


def _percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def decide(learning, baseline_batch, candidate_batch, *, tier, mechanism, min_gain=0.0,
           max_cost_increase=0.2, resamples=2000, seed=0):
    """Paired bootstrap (5th/95th percentiles), with cost on the same paired tasks.

    No baseline successes means its cost/success is undefined, so a finite
    candidate cost passes the cost gate. Zero baseline token cost permits only
    zero candidate token cost. No overlap cannot establish gain and switches off.
    """
    if tier not in TIERS or mechanism not in MECHANISMS:
        raise ValueError('Unknown tier or mechanism')
    if type(resamples) is not int or resamples < 1:
        raise ValueError('resamples must be a positive integer')
    if (not math.isfinite(min_gain) or not math.isfinite(max_cost_increase) or max_cost_increase < 0):
        raise ValueError('Decision thresholds must be finite, with nonnegative max_cost_increase')
    baseline, candidate = _rows(learning, baseline_batch), _rows(learning, candidate_batch)
    if any(row['tier'] != tier for row in baseline + candidate):
        raise ValueError('Both batches must belong to the requested tier')
    baseline = {row['task_id']: row for row in baseline}
    candidate = {row['task_id']: row for row in candidate}
    paired = sorted(baseline.keys() & candidate.keys())
    costs = {'baseline': _cost([baseline[key] for key in paired]),
             'candidate': _cost([candidate[key] for key in paired])}
    result = dict(gain=None, ci_low=None, ci_high=None, costs=costs, paired_tasks=len(paired))
    if paired:
        differences = [int(bool(candidate[key]['success'])) - int(bool(baseline[key]['success'])) for key in paired]
        rng = random.Random(seed)
        samples = sorted(sum(rng.choice(differences) for _ in paired) / len(paired) for _ in range(resamples))
        result.update(gain=sum(differences) / len(paired), ci_low=_percentile(samples, 0.05),
                      ci_high=_percentile(samples, 0.95))
        old, new = costs['baseline']['tokens'], costs['candidate']['tokens']
        cost_ok = new is not None and (old is None or new <= old * (1 + max_cost_increase))
        if result['ci_low'] > min_gain and cost_ok:
            return {**result, 'decision': 'recommend_enable', 'recommend': 'enable',
                    'reason': '90% paired bootstrap lower bound exceeds min_gain and token cost is within budget; human approval required'}
        reason = ('90% paired bootstrap lower bound does not exceed min_gain'
                  if result['ci_low'] <= min_gain else 'Token cost per success exceeds the allowed increase')
    else:
        reason = 'No paired tasks; measured gain cannot be established'
    learning.set_mechanism(tier, mechanism, False, reason, 'eval')
    return {**result, 'decision': 'disable', 'reason': reason}
