from langchain.tools import tool

from . import dashboard, memory


def make_update_tools(llm) -> list:

    @tool(parse_docstring=True)
    def Write_to_Dashboard(parent_asin: str, note: str) -> str:
        """
        Log a product recommendation to the dashboard.

        Args:
            parent_asin: Product identifier.
            note: Reason or summary to store.
        """

        dashboard.write_entry(
            {
                "parent_asin": parent_asin,
                "note": note,
            }
        )

        return f"Successfully logged '{parent_asin}' to the dashboard."


    @tool(parse_docstring=True)
    def Update_Memory(
        key: str,
        value: str,
        operation: str = "set",
    ) -> str:
        """
        Update the agent's persistent memory.

        Supported operations:
        - set
        - append
        - merge
        - delete

        Args:
            key: Memory key.
            value: Value to store.
            operation: Memory update strategy.
        """

        operation = operation.lower()

        try:

            if operation == "set":
                memory.set(key, value)

            elif operation == "append":
                memory.append(key, value)

            elif operation == "merge":
                memory.merge(key, value)

            elif operation == "delete":
                deleted = memory.delete(key)

                if not deleted:
                    return f"Memory key '{key}' does not exist."

            else:
                return (
                    "Unknown operation. "
                    "Supported operations are: "
                    "set, append, merge, delete."
                )

            return f"Memory successfully updated using '{operation}'."

        except Exception as e:
            return f"Failed to update memory: {e}"


    @tool(parse_docstring=True)
    def Explain_Recommendation(
        parent_asin: str,
        reason: str,
    ) -> str:
        """
        Generate a customer-friendly explanation for a recommendation.

        Args:
            parent_asin: Product identifier.
            reason: Internal reasoning.
        """

        prompt = f"""
You are an AI shopping assistant.

Rewrite the following internal reasoning into a short,
friendly recommendation.

Requirements:
- 1-2 sentences
- Natural language
- Mention benefits
- Do not mention internal reasoning
- Do not mention the product id

Reason:
{reason}
"""

        return llm.invoke(prompt).content.strip()


    return [
        Write_to_Dashboard,
        Update_Memory,
        Explain_Recommendation,
    ]