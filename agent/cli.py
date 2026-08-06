"""
Interactive CLI for the beauty dataset agent.

Run with: python -m agent.cli

Commands:
  /help        show this message
  /memory      show everything currently stored in agent memory
  /dashboard   show all logged dashboard entries
  /clear       clear the terminal
  /exit        quit
"""
import os
import sys
import time
from . import memory, dashboard
from .data import load_dataset
from .model import build_chat_llm
from .agent_builder import build_agent

HELP_TEXT = """
Commands:
  /help        show this message
  /memory      show everything currently stored in agent memory
  /dashboard   show all logged dashboard entries
  /debug       toggle showing the agent's raw Thought/Action/Observation trace
  /clear       clear the terminal
  /exit        quit
Anything else is sent to the agent as a question.
"""


def _print_header():
    print("=" * 60)
    print("Amazon Beauty Dataset Agent")
    print("=" * 60)
    print(HELP_TEXT)


def _handle_command(cmd: str, agent) -> bool:
    """Returns True if input was a command (handled here), False if it should go to the agent."""
    if cmd == "/help":
        print(HELP_TEXT)
    elif cmd == "/memory":
        data = memory.all()
        print(data if data else "(memory is empty)")
    elif cmd == "/dashboard":
        rows = dashboard.all_entries()
        if not rows:
            print("(dashboard is empty)")
        else:
            for row in rows:
                print(f"- [{row.get('timestamp')}] {row.get('parent_asin')}: {row.get('note')}")
    elif cmd == "/debug":
        agent.verbose = not agent.verbose
        print(f"Debug trace {'ON' if agent.verbose else 'OFF'}.")
    elif cmd == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
    else:
        return False
    return True


def main():
    print("Loading dataset...")
    df = load_dataset()

    print("Loading model (this can take a minute on first run)...")
    t0 = time.time()
    llm = build_chat_llm()
    agent = build_agent(df, llm)
    print(f"Ready in {time.time() - t0:.1f}s.\n")

    _print_header()

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question in ("/exit", "/quit"):
            print("Exiting.")
            break
        if question.startswith("/"):
            if not _handle_command(question, agent):
                print(f"Unknown command: {question}. Type /help for options.")
            continue

        t0 = time.time()
        try:
            result = agent.invoke({"input": question})
            print(f"\nagent> {result['output']}")
        except Exception as e:
            print(f"\nagent> Error while answering: {e}")
        print(f"({time.time() - t0:.1f}s)\n")


if __name__ == "__main__":
    sys.exit(main() or 0)
