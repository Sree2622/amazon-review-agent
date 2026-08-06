# Agent

This directory contains the interactive Amazon Beauty dataset agent. It uses **Gemini 3.5 Flash Lite** (`gemini-3.5-flash-lite`) with LangChain tools, processed product data, and persistent state.

## Agentic Features

The system is agentic because it does more than retrieve data and generate a response:

* **Reasoning loop:** it iterates through Thought → Action → Observation using `AgentExecutor`.
* **Dynamic tool selection:** it selects the next tool and arguments based on the current question and previous result.
* **Multi-step execution:** it can filter products, inspect results, refine the search, read reviews, and synthesize a recommendation.
* **Persistent memory:** it stores and retrieves user preferences across turns.
* **State-changing actions:** it can write recommendations to the dashboard.
* **Autonomous completion:** it returns a final answer after deciding it has enough information.

## Available Tools

| Tool group | Capabilities |
| --- | --- |
| Dataset queries | Ad-hoc Pandas analysis, product lookup, and review retrieval. |
| Recommendations | Product suggestions and review analysis. |
| Memory | Read, search, summarize, update, append, merge, and delete stored preferences. |
| Dashboard | Save selected recommendations for later review. |

## Recorded Agent Output

The recorded CLI session demonstrates the agent refining a moisturizer query, reading product reviews, storing a sensitive-skin and Korean-skincare preference, comparing vitamin C serums, and writing the selected recommendation to the dashboard. One completed moisturizer recommendation was returned in 6.7 seconds; the preference update completed in 1.8 seconds.

## Files

| File | Description |
| --- | --- |
| `agent_builder.py` | Builds the structured-chat agent and registers tools. |
| `cli.py` | Provides the interactive command-line interface. |
| `data.py` | Loads and merges product metadata with reviews. |
| `query_tools.py` | Provides review and memory lookup tools. |
| `process_tools.py` | Provides product lookup and recommendation tools. |
| `update_tools.py` | Provides dashboard and memory update tools. |
| `memory.py` | Manages persistent JSON-based memory. |
| `dashboard.py` | Stores recommendation dashboard entries. |
| `model.py` | Creates the Gemini chat model. |

## Run the Agent

```bash
python -m agent.cli
```

Processed review and product Parquet files must be available in `data/processed/`.
