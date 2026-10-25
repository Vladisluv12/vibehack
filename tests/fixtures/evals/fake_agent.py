"""Local eval agent: the oracle can pass only after the skill marker is supplied."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--tokens', type=int, default=10)
    parser.add_argument('--mode', choices=('skill', 'slow', 'nonzero'), default='skill')
    parser.add_argument('--no-usage', action='store_true')
    args = parser.parse_args()
    workspace = Path(args.workspace)
    if (workspace / 'ran.txt').exists():
        raise RuntimeError('workspace was reused')
    (workspace / 'ran.txt').write_text('ran', encoding='utf-8')
    (workspace / 'prompt.txt').write_text(args.prompt, encoding='utf-8')
    if args.mode == 'slow':
        time.sleep(10)
    if 'EVAL_SKILL_MARKER' in args.prompt:
        (workspace / 'answer.txt').write_bytes((workspace / 'expected.txt').read_bytes())
    print('fake agent log; no model or network')
    if not args.no_usage:
        print(json.dumps({'usage': {'tokens': args.tokens}}))
    return 7 if args.mode == 'nonzero' else 0


if __name__ == '__main__':
    sys.exit(main())
