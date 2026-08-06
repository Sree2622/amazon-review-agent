import pandas as pd
from langchain_classic.agents import create_structured_chat_agent, AgentExecutor
from langchain_classic.agents.structured_chat.prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonAstREPLTool
from . import config
from .query_tools import make_query_tools
from .process_tools import make_process_tools
from .update_tools import make_update_tools

# Structured chat uses JSON actions for reliable local-model tool calls.
DATASET_CONTEXT = """You are a data analyst assistant for an Amazon Beauty products + reviews dataset.

A pandas dataframe called `df` is available to the python_repl_ast tool. Its columns are:
{columns}

Sample rows:
{sample}

Do not invent data — every number in your final answer must come from an actual tool
Observation. If a tool call fails or returns nothing useful, say so rather than guessing."""

# Worked example of the required JSON action format.
FORMAT_EXAMPLE = """Here is a worked example of the expected format:

Question: suggest a face wash
Thought: The user wants a product suggestion, so I should search the catalog with Suggest_product.
Action:
```
{{
  "action": "Suggest_product",
  "action_input": {{"query": "face wash", "top_n": 5}}
}}
```
Observation: B001234: Gentle Foaming Face Wash (rating=4.5)
Thought: I have a real result from the tool, so I can give the final answer now.
Action:
```
{{
  "action": "Final Answer",
  "action_input": "I'd recommend the Gentle Foaming Face Wash (B001234) — it has a 4.5-star average rating."
}}
```

Always end with a "Final Answer" action once you have enough information. Never
leave a response without producing a final "Final Answer" action.

Strict rules — breaking any of these will cause your response to be rejected:
1. Output EXACTLY ONE json blob per turn: one opening ``` and one closing ```,
   nothing after the closing ```. Never write more than one Action in a turn.
2. Use the tool name exactly as spelled in the tool list above — do not
   rename, capitalize, pluralize, or invent a tool that isn't listed.
3. The json blob has exactly two keys, spelled exactly "action" and
   "action_input" (lowercase, with an underscore) — NOT "actionInput",
   "ActionInput", or "input". Never nest it inside another key like
   "json_blob" or wrap it in a list.
4. NEVER invent, guess, simulate, or fabricate a number, rating, product name,
   or ASIN. If you don't have real data from a tool Observation yet, call a
   tool to get it — don't write "assuming", "let's simulate", or similar and
   make something up. If a tool genuinely has nothing useful, say so plainly
   in your Final Answer instead of inventing a result.
5. Keep each Thought to one short sentence. Do not write step-by-step plans,
   alternative approaches, or notes to yourself in the Thought."""


def build_agent(df: pd.DataFrame, llm) -> AgentExecutor:
    python_tool = PythonAstREPLTool(locals={"df": df})
    tools = (
        [python_tool]
        + make_query_tools(df)
        + make_process_tools(df, llm)
        + make_update_tools(llm)
    )
    for t in tools:
        t.handle_validation_error = True
        t.handle_tool_error = True

    sample = df.head(config.AGENT_HEAD_ROWS).to_string() if config.INCLUDE_DF_IN_PROMPT else "(omitted — use python_repl_ast to inspect)"
    dataset_context = DATASET_CONTEXT.format(columns=", ".join(df.columns), sample=sample)

    system_message = (
        dataset_context
        + "\n\n"
        + PREFIX
        + "\n\n{tools}\n\n"
        + FORMAT_INSTRUCTIONS
        + "\n\n"
        + FORMAT_EXAMPLE
        + "\n\n"
        + SUFFIX
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}\n\n{agent_scratchpad}"),
        ]
    )

    agent = create_structured_chat_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=config.AGENT_DEBUG,
        max_iterations=config.AGENT_MAX_ITERATIONS,
        max_execution_time=config.AGENT_MAX_EXECUTION_TIME,
        handle_parsing_errors=True,
        early_stopping_method="force",
    )
