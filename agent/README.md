# Agent

This directory contains the interactive Amazon Beauty dataset agent. It uses **Gemini 3.5 Flash Lite** (`gemini-3.5-flash-lite`) with LangChain tools, processed product data, and persistent state.

## Agentic Features

The system is agentic because it does more than retrieve data and generate a response:

* **Reasoning loop:** iterates through Thought → Action → Observation using `AgentExecutor`.
* **Dynamic tool selection:** picks the next tool and arguments based on the current question and previous result.
* **Multi-step execution:** filters products, inspects results, refines the search, reads reviews, and synthesizes a recommendation.
* **Persistent memory:** stores and retrieves user preferences across turns.
* **State-changing actions:** writes recommendations to a dashboard.
* **Autonomous completion:** returns a final answer after deciding it has enough information.

## Available Tools

| Tool group | Capabilities |
| --- | --- |
| Dataset queries | Ad-hoc Pandas analysis, product lookup, and review retrieval. |
| Recommendations | Product suggestions and review analysis. |
| Memory | Read, search, summarize, update, append, merge, and delete stored preferences. |
| Dashboard | Save selected recommendations for later review. |

## Sample Session

A excerpt showing the reasoning loop, a memory write, and a dashboard write:

```text
you> Find the highest-rated moisturizer under $20 with at least 200 reviews.

> Entering new AgentExecutor chain...
Thought: I need to filter products by category, price, and review count.
Action:
{
  "action": "python_repl_ast",
  "action_input": {
    "query": "moist = df[df['title_product'].str.contains('moisturizer|cream|lotion', case=False, na=False) & (df['price'] < 20)]; ..."
  }
}
Observation:
parent_asin  title_product                                    count   mean   price
B007IAM4YU   Silver Miracles Colloidal Silver Cream - 4oz     246   4.32   19.99
B08XMBHL8C   MARLOWE. 002 Extra Moisturizing Body Lotion      446   4.17   14.99

Action:
{
  "action": "Read_Reviews",
  "action_input": { "parent_asin": "B007IAM4YU", "limit": 5 }
}
Observation: [5 reviews retrieved — praised as a daily face moisturizer with healing properties]

Action:
{
  "action": "Final Answer",
  "action_input": "The highest-rated moisturizer under $20 with at least 200 reviews is the
  Silver Miracles Colloidal Silver Cream - 4oz (B007IAM4YU): 4.32 avg rating, 246 reviews, $19.99.
  Reviews highlight its use as a daily facial moisturizer and its healing properties for skin
  irritation and burns."
}
> Finished chain.
(6.7s)

you> I have sensitive skin and prefer Korean skincare. Remember this for future recommendations.

> Entering new AgentExecutor chain...
Action:
{
  "action": "Update_Memory",
  "action_input": {
    "key": "user_preferences",
    "value": "Sensitive skin, prefers Korean skincare",
    "operation": "set"
  }
}
Observation: Memory successfully updated using 'set'.

Action:
{
  "action": "Final Answer",
  "action_input": "Saved your preference for Korean skincare and sensitive skin — I'll use this
  for future recommendations."
}
> Finished chain.
(1.8s)

you> /memory
{'user_preferences': {'value': 'Sensitive skin, prefers Korean skincare', 'updated_at': '2026-08-06T02:49:26Z'}}

you> Compare the top 5 vitamin C serums and recommend only one. And add it to dashboard.

> Entering new AgentExecutor chain...
Action:
{
  "action": "Suggest_product",
  "action_input": { "query": "vitamin c serum", "top_n": 5 }
}
Observation: [5 candidate serums retrieved]

Action:
{
  "action": "Write_to_Dashboard",
  "action_input": {
    "parent_asin": "B01MRRJMGM",
    "note": "Top recommended vitamin c serum based on user preference and specifications."
  }
}
Observation: Successfully logged 'B01MRRJMGM' to the dashboard.

Action:
{
  "action": "Final Answer",
  "action_input": "Recommended: 'Best Advanced Vitamin C Serum for Face & Neck' (ASIN: B01MRRJMGM),
  5.0 rating. Added to your dashboard."
}
> Finished chain.
(5.7s)

you> /dashboard
- [2026-08-05T09:27:02Z] B01J67KBR4: Added 24K Gold Deep Day Moisturizer Cream to cart/dashboard.
- [2026-08-05T10:29:48Z] B00EISHXMI: Recommended Korean skincare product matching user preferences.
- [2026-08-06T02:49:56Z] B01MRRJMGM: Top recommended vitamin c serum based on user preference.
```

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
