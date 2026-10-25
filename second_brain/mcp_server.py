"""Small STDIO knowledge gateway for agents (requires fastmcp).

The tools themselves live in `tools.py`; this module only exposes them over MCP.
"""
from pathlib import Path

from .tools import INSTRUCTIONS, Tools

READ_ONLY = {'readOnlyHint': True, 'destructiveHint': False, 'openWorldHint': False}


def build(vault: Path, state: Path, actor=None):
    from fastmcp import FastMCP

    mcp = FastMCP('second-brain', instructions=INSTRUCTIONS)
    tools = Tools(vault, state, actor)
    for name in ('search_knowledge', 'get_document', 'get_related'):
        mcp.tool(annotations=READ_ONLY)(getattr(tools, name))
    mcp.tool(annotations={**READ_ONLY, 'readOnlyHint': False})(tools.create_draft)
    return mcp


def run(vault, state, actor=None):
    build(Path(vault).absolute(), Path(state).absolute(), actor).run(transport='stdio', show_banner=False)
