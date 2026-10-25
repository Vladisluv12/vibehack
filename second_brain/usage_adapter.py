"""Run an agent CLI for eval-run and report its provider-counted tokens.

    python3 -m second_brain.usage_adapter --format claude -- claude -p "{prompt}" --output-format json

The wrapped command's output goes to stderr and its exit code passes through; stdout
carries only one line {"usage": {"tokens": N}}, counted from the CLI's own JSON (input +
output + reasoning, cache reads excluded), so no other usage-shaped line can confuse
eval-run. No usage found → empty stdout, so the cost stays unknown.
Formats, checked against real output: claude (one result object), codex (`exec --json`
events; turn.completed carries the running total), opencode (`run --format json`;
every step_finish carries that step's tokens).
"""
import argparse
import json
import os
import subprocess
import sys

FORMATS = ('claude', 'codex', 'opencode')


def _count(usage):
    if not isinstance(usage, dict):
        return None
    total, hit = 0, False
    for key in ('input_tokens', 'output_tokens', 'reasoning_output_tokens', 'input', 'output', 'reasoning'):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            total, hit = total + value, True
    return total if hit else None


def _objects(stdout):
    try:
        whole = json.loads(stdout)
        return whole if isinstance(whole, list) else [whole]
    except ValueError:
        pass
    found = []
    for line in stdout.splitlines():
        try:
            found.append(json.loads(line))
        except ValueError:
            continue
    return [o for o in found if isinstance(o, dict)]


def tokens(stdout, fmt):
    objects = _objects(stdout)
    if fmt == 'claude':
        counts = [_count(o.get('usage')) for o in objects if o.get('type') == 'result']
        return counts[-1] if counts and counts[-1] is not None else None
    if fmt == 'codex':
        counts = [_count(o.get('usage')) for o in objects if o.get('type') == 'turn.completed']
        return counts[-1] if counts and counts[-1] is not None else None
    steps = [_count((o.get('part') or {}).get('tokens')) for o in objects if o.get('type') == 'step_finish']
    steps = [s for s in steps if s is not None]
    return sum(steps) if steps else None


def main(argv=None):
    parser = argparse.ArgumentParser(prog='usage_adapter')
    parser.add_argument('--format', choices=FORMATS, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('command to run is required after --')
    done = subprocess.run(command, stdin=sys.stdin, capture_output=True, text=True, shell=False,
                          env={**os.environ, 'PWD': os.getcwd()})  # agent CLIs read their project root from PWD
    sys.stderr.write(done.stdout)
    sys.stderr.write(done.stderr)
    count = tokens(done.stdout, args.format)
    if count is not None:
        print(json.dumps({'usage': {'tokens': count}}))
    return done.returncode


if __name__ == '__main__':
    sys.exit(main())
