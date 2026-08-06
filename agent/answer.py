from langchain.tools import tool
from .data import load_dataset
from .model import build_chat_llm
from .agent_builder import build_agent

df = load_dataset()
llm = build_chat_llm()
agent = build_agent(df, llm)


@tool(parse_docstring=True)
def ask_beauty_dataset(question: str) -> str:
    """Answer questions about Amazon Beauty products and reviews.

    Args:
        question: Question about products, ratings, reviews, brands, price, sentiment.
    """
    return agent.invoke({"input": question})["output"]


if __name__ == "__main__":
    questions = [
        "What are the top 10 products with the highest number of reviews?",
        "Which products have the highest average rating?",
        "What are customers saying about the most popular products?",
        "Which brands have the most products?",
        "Find products with negative customer sentiment.",
    ]
    for q in questions:
        print(f"\nQ: {q}\nA: {ask_beauty_dataset.invoke({'question': q})}")
