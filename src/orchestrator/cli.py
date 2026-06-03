from __future__ import annotations

import asyncio
from typing import Optional

from dotenv import load_dotenv
import typer

load_dotenv()  # Load .env file for API keys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .engine.orchestrator import AgentOrchestrator
from .llm import LLMConfig, create_llm
from .observability.logger import setup_logging

app = typer.Typer(
    name="orchestrator",
    help="AI Agent Orchestrator — coordinate multiple agents to accomplish complex tasks",
)
console = Console()


@app.command()
def run(
    goal: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    provider: str = "huggingface",
    session: Optional[str] = None,
) -> None:
    """Execute a goal through the orchestrator."""
    setup_logging()
    llm = create_llm(LLMConfig(provider=provider, model=model))
    orchestrator = AgentOrchestrator(llm=llm)

    with console.status("[bold green]Planning and executing...") as status:
        result = asyncio.run(orchestrator.execute(goal, session_id=session))

    console.print(Panel(Markdown(result), title="[bold]Result", border_style="green"))


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    provider: str = "huggingface",
) -> None:
    """Start the FastAPI server."""
    import uvicorn
    setup_logging()
    console.print(f"[bold green]Starting orchestrator API on {host}:{port}")
    console.print(f"[dim]LLM: {provider}/{model}")
    uvicorn.run(
        "orchestrator.api.server:app",
        host=host,
        port=port,
        reload=False,
    )


@app.command()
def interactive(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    provider: str = "huggingface",
) -> None:
    """Interactive CLI session with the orchestrator."""
    setup_logging()
    llm = create_llm(LLMConfig(provider=provider, model=model))
    orchestrator = AgentOrchestrator(llm=llm)

    console.print("[bold cyan]AI Agent Orchestrator — Interactive Mode[/bold cyan]")
    console.print("[dim]Type 'exit' to quit, 'agents' to list registered agents[/dim]\n")

    while True:
        goal = typer.prompt("Goal", default="")
        if goal.lower() in ("exit", "quit"):
            break
        if goal.lower() == "agents":
            table = Table(title="Registered Agents")
            table.add_column("Name", style="cyan")
            table.add_column("Description")
            for name, agent in orchestrator._agents.items():
                table.add_row(name, agent.description)
            console.print(table)
            continue
        if not goal:
            continue

        with console.status("[bold green]Working...") as status:
            try:
                result = asyncio.run(orchestrator.execute(goal))
                console.print(Panel(Markdown(result), border_style="green"))
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    app()
