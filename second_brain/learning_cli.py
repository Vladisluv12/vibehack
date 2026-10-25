"""CLI commands for the learning loop. The host runs these; agents only reach MCP.

External models are argv JSON files (no shell, no keys in config): the prompt goes
to stdin, the answer is read from stdout.
"""
import json
from pathlib import Path
import subprocess

from . import evals, lint, reflection, reviewer, skills
from .brain import Brain
from .learning import MECHANISMS, TIERS, Learning

COMMANDS = {'lint', 'ingest-run', 'reflect-request', 'review', 'skill-pack', 'promote', 'stale',
            'eval-lock', 'eval-run', 'eval-metrics', 'eval-decide'}


def add_commands(cmds):
    c = cmds.add_parser('lint', help='Deterministic vault checks; exit 1 on errors')
    c.add_argument('--online', action='store_true', help='Also request http(s) sources (network)')
    c = cmds.add_parser('ingest-run', help='Store a redacted agent run summary (JSON object or list)')
    c.add_argument('path', type=Path)
    c.add_argument('--model', required=True, help='Model the host actually ran; decides the tier')
    c = cmds.add_parser('reflect-request', help='Queue reflection for one run on demand'); c.add_argument('run_id')
    c = cmds.add_parser('review', help='Process pending reflection triggers with an external model')
    c.add_argument('--llm', type=Path, required=True, help='JSON argv of a model CLI that reads the prompt on stdin')
    c.add_argument('--limit', type=int, default=5); c.add_argument('--budget', type=int, default=60000)
    c.add_argument('--timeout', type=float, default=600)
    c = cmds.add_parser('skill-pack', help='Skills to prepend to a task prompt for a model')
    c.add_argument('--model', required=True); c.add_argument('--task', required=True); c.add_argument('--run-id', required=True)
    c.add_argument('--k', type=int, default=3)
    c = cmds.add_parser('promote', help='Set the rollout stage of a reviewed skill (a human decision)')
    c.add_argument('--skill', required=True); c.add_argument('--stage', choices=('canary', 'full', 'off'), required=True)
    c = cmds.add_parser('stale', help='Notes served often but never reported as helpful')
    c.add_argument('--min-served', type=int, default=5)
    c = cmds.add_parser('eval-lock', help='Freeze eval task files into tasks.lock'); c.add_argument('tasks', type=Path)
    c = cmds.add_parser('eval-run', help='Run locked eval tasks: A no skills, B skills, C B + reflection-made skills')
    c.add_argument('--tasks', type=Path, required=True); c.add_argument('--config', choices=('A', 'B', 'C'), required=True)
    c.add_argument('--model', required=True); c.add_argument('--agent', type=Path, required=True, help='JSON argv with {prompt} and {workspace}')
    c.add_argument('--timeout', type=float, default=600)
    c = cmds.add_parser('eval-metrics'); c.add_argument('batch')
    c = cmds.add_parser('eval-decide', help='Paired bootstrap: no gain switches the mechanism off. '
                        'Compare A→C for skills and B→C for reflection')
    c.add_argument('--baseline', required=True); c.add_argument('--candidate', required=True)
    c.add_argument('--tier', choices=TIERS, required=True); c.add_argument('--mechanism', choices=MECHANISMS, required=True)


def llm_command(argv, timeout):
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ValueError('LLM config must be a JSON list of strings')

    def call(prompt):
        done = subprocess.run(argv, input=prompt, capture_output=True, text=True, timeout=timeout, shell=False)
        if done.returncode != 0:
            raise ValueError('LLM command failed')
        return done.stdout
    return call


def run(args):
    """Returns (result, exit code)."""
    learning = Learning(args.state)
    brain = None
    try:
        if args.cmd in ('lint', 'review', 'skill-pack', 'eval-run'):
            reviewer_meta = {'agent_model': args.llm.stem} if args.cmd == 'review' else None
            brain = Brain(args.vault, args.state, actor='agent:reviewer' if args.cmd == 'review' else 'human:local',
                          actor_meta=reviewer_meta)
            brain.sync()
        if args.cmd == 'lint':
            report = lint.lint(brain, online=args.online)
            return report, lint.exit_code(report)
        if args.cmd == 'ingest-run':
            data = json.loads(args.path.read_text(encoding='utf-8'))
            return [reflection.ingest_run(learning, e, args.model) for e in (data if isinstance(data, list) else [data])], 0
        if args.cmd == 'reflect-request':
            return reflection.request_reflection(learning, args.run_id), 0
        if args.cmd == 'review':
            call = llm_command(json.loads(args.llm.read_text(encoding='utf-8')), args.timeout)
            return reviewer.review_pending(brain, learning, call, limit=args.limit, budget_chars=args.budget), 0
        if args.cmd == 'skill-pack':
            return skills.skill_pack(brain, learning, tier=learning.tier(args.model), task=args.task,
                                     run_id=args.run_id, k=args.k), 0
        if args.cmd == 'promote':
            return skills.promote(learning, args.skill, args.stage, 'human:cli'), 0
        if args.cmd == 'stale':
            return learning.stale_candidates(args.min_served), 0
        if args.cmd == 'eval-lock':
            return evals.lock_tasks(args.tasks), 0
        if args.cmd == 'eval-run':
            tasks, tier = evals.load_tasks(args.tasks), learning.tier(args.model)
            pack = None
            if args.config != 'A':
                def pack(task):
                    return skills.skill_pack(brain, learning, tier=tier, task=task['applies_to'] or task['prompt'],
                                             run_id=f"eval:{task['id']}", include_reflected=args.config == 'C')['text']
            batch = evals.run_batch(tasks, config=args.config, tier=tier, agent_model=args.model,
                                    agent_argv=evals.load_agent_argv(args.agent), learning=learning,
                                    skill_pack=pack, timeout=args.timeout)
            return evals.metrics(learning, batch), 0
        if args.cmd == 'eval-metrics':
            return evals.metrics(learning, args.batch), 0
        if args.cmd == 'eval-decide':
            return evals.decide(learning, args.baseline, args.candidate, tier=args.tier, mechanism=args.mechanism), 0
    finally:
        if brain:
            brain.close()
        learning.close()
