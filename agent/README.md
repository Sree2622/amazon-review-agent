# Agent

This directory contains the interactive Amazon Beauty dataset agent. The agent combines processed product and review data with LangChain tools, persistent memory, and a chat model.

## Files

| File | Description |
| --- | --- |
| `agent_builder.py` | Builds the structured-chat agent and registers its tools. |
| `answer.py` | Exposes the dataset question-answering tool. |
| `cli.py` | Provides the interactive command-line interface. |
| `config.py` | Stores dataset paths and agent runtime settings. |
| `data.py` | Loads and merges product metadata with review data. |
| `query_tools.py` | Provides review and memory lookup tools. |
| `process_tools.py` | Provides product lookup, recommendation, and review-analysis tools. |
| `update_tools.py` | Provides dashboard and persistent-memory update tools. |
| `memory.py` | Manages persistent JSON-based agent memory. |
| `dashboard.py` | Stores recommendation dashboard entries. |
| `model.py` | Creates the Google Gemini chat model used by the agent. |
| `diagnose.py` | Checks model device placement and generation speed. |
| `upload_data.py` | Uploads the processed review dataset to the configured backend. |

## Run the Agent

```bash
python -m agent.cli
```

The processed review and product Parquet files must be available in `data/processed/`.
