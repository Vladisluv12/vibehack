import argparse
import json
import os
from pathlib import Path
import shutil

from .brain import Brain, restore_backup
from .github import import_github

DEMO = Path(__file__).resolve().parent.parent / 'examples' / 'demo-vault'
SHOWCASE = DEMO.parent / 'showcase-vault'


def parser():
    p = argparse.ArgumentParser(prog='second_brain', description='Local Markdown knowledge vault')
    p.add_argument('--vault', type=Path, default=Path(os.environ.get('SECOND_BRAIN_VAULT', 'vault')))
    p.add_argument('--state', type=Path, default=Path(os.environ.get('SECOND_BRAIN_STATE', 'var/state')))
    cmds = p.add_subparsers(dest='cmd', required=True)
    cmds.add_parser('sync')
    c = cmds.add_parser('search'); c.add_argument('query'); c.add_argument('--kind', default=''); c.add_argument('--offset', type=int, default=0)
    c = cmds.add_parser('read'); c.add_argument('id')
    c = cmds.add_parser('import'); c.add_argument('path', type=Path); c.add_argument('--split', action='store_true')
    c = cmds.add_parser('import-github', help='Import Markdown files of a GitHub repository (needs network)'); c.add_argument('url')
    cmds.add_parser('backup')
    c = cmds.add_parser('rollback', help='Undo journaled changes by op id or actor (e.g. agent:qwen3.8-flash)')
    g = c.add_mutually_exclusive_group(required=True); g.add_argument('--op-id'); g.add_argument('--actor')
    c = cmds.add_parser('restore'); c.add_argument('archive', type=Path); c.add_argument('destination', type=Path)
    c = cmds.add_parser('demo', help='Copy the synthetic demo vault into --vault if it does not exist')
    c.add_argument('--showcase', action='store_true', help='Larger, densely linked vault for presentations')
    c = cmds.add_parser('serve'); c.add_argument('--port', type=int, default=8765)
    cmds.add_parser('mcp', help='STDIO MCP gateway (requires fastmcp)')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cmd == 'restore':
        return print(json.dumps(restore_backup(args.archive, args.destination), ensure_ascii=False, indent=2))
    if args.cmd == 'demo' and not args.vault.exists():
        shutil.copytree(SHOWCASE if args.showcase else DEMO, args.vault)
    if args.cmd == 'serve':
        from .web import make_server
        server = make_server(args.vault, args.state, args.port)
        print(f'Second Brain: http://127.0.0.1:{args.port}/  (vault: {args.vault})', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return
    if args.cmd == 'mcp':
        from .mcp_server import run
        return run(args.vault, args.state)
    brain = Brain(args.vault, args.state)
    try:
        result = brain.sync()
        if args.cmd == 'search': result = brain.list(args.query, args.kind, args.offset)
        elif args.cmd == 'read': result = brain.get(args.id)
        elif args.cmd == 'import': result = brain.import_markdown(args.path, split=args.split)
        elif args.cmd == 'import-github': result = import_github(brain, args.url)
        elif args.cmd == 'backup': result = brain.backup()
        elif args.cmd == 'rollback': result = brain.rollback(op_id=args.op_id, actor=args.actor)
        if args.cmd in ('import', 'import-github'):
            brain.sync()
    finally:
        brain.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
