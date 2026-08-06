import pandas as pd
from langchain.tools import tool

from . import memory


def make_query_tools(df: pd.DataFrame) -> list:

    @tool(parse_docstring=True)
    def Read_Reviews(parent_asin: str, limit: int = 5) -> str:
        """
        Read raw customer reviews for a product.

        Args:
            parent_asin: Product identifier.
            limit: Maximum number of reviews to return.
        """

        rows = (
            df.loc[df["parent_asin"] == parent_asin, "text"]
            .dropna()
            .head(limit)
        )

        if rows.empty:
            return f"No reviews found for '{parent_asin}'."

        return "\n---\n".join(rows.tolist())


    @tool(parse_docstring=True)
    def Read_Memory(key: str) -> str:
        """
        Read a value from memory.

        Performs:
        1. Exact lookup
        2. Partial key lookup
        3. Fuzzy key lookup

        Args:
            key: Memory key or search phrase.
        """

        value = memory.get(key)

        # exact match
        if value is not None:
            if isinstance(value, dict) and "value" in value:
                return str(value["value"])
            return str(value)

        # fuzzy match
        result = memory.search(key)

        if result:
            lines = []

            for k, v in result.items():

                if isinstance(v, dict) and "value" in v:
                    v = v["value"]

                lines.append(f"{k}: {v}")

            return "\n".join(lines)

        return f"No relevant memory found for '{key}'."


    @tool(parse_docstring=True)
    def List_Memory() -> str:
        """
        List all memory keys.
        """

        keys = memory.keys()

        if not keys:
            return "Memory is empty."

        return "\n".join(keys)


    @tool(parse_docstring=True)
    def Recent_Memory(limit: int = 5) -> str:
        """
        Show the most recently updated memory entries.

        Args:
            limit: Maximum number of entries.
        """

        recent = memory.recent(limit)

        if not recent:
            return "Memory is empty."

        lines = []

        for key, value in recent.items():

            stored = value.get("value", value)

            lines.append(f"{key}: {stored}")

        return "\n".join(lines)


    @tool(parse_docstring=True)
    def Memory_Summary() -> str:
        """
        Return a high-level summary of stored memory.
        """

        return memory.summary()


    return [
        Read_Reviews,
        Read_Memory,
        List_Memory,
        Recent_Memory,
        Memory_Summary,
    ]
