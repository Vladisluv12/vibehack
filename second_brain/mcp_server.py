"""Small STDIO knowledge gateway for agents (requires fastmcp).

The tools themselves live in `tools.py`; this module only exposes them over MCP.
"""
from pathlib import Path

from .tools import INSTRUCTIONS, NAMES, Tools, propose_draft, readable, search  # noqa: F401 (re-exported)

READ_ONLY = {'readOnlyHint': True, 'destructiveHint': False, 'openWorldHint': False}
WRITES = {'create_draft', 'report_usage', 'reflect'}


def build(vault: Path, state: Path, actor=None, agent_model=''):
    from fastmcp import FastMCP

    mcp = FastMCP('second-brain', instructions=INSTRUCTIONS)
    tools = Tools(vault, state, actor, agent_model)
    for name in NAMES:
        annotations = {**READ_ONLY, 'readOnlyHint': False} if name in WRITES else READ_ONLY
        mcp.tool(annotations=annotations)(getattr(tools, name))
    return mcp


def run(vault, state, actor=None, agent_model=''):
    build(Path(vault).absolute(), Path(state).absolute(), actor, agent_model).run(transport='stdio', show_banner=False)
