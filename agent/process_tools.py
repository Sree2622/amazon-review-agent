import pandas as pd
from langchain.tools import tool


def make_process_tools(df: pd.DataFrame, llm) -> list:
    """
    Creates product processing tools.
    """

    title_col = "title_product" if "title_product" in df.columns else "title"

    @tool(parse_docstring=True)
    def Fetch_product(parent_asin: str) -> str:
        """
        Fetch detailed metadata for a product.

        Args:
            parent_asin: Product identifier.
        """

        row = (
            df[df["parent_asin"] == parent_asin]
            .drop_duplicates("parent_asin")
        )

        if row.empty:
            return f"No product found for '{parent_asin}'."

        r = row.iloc[0]

        return (
            f"Title: {r.get(title_col, 'Unknown')}\n"
            f"Store: {r.get('store', 'Unknown')}\n"
            f"Price: {r.get('price', 'Unknown')}\n"
            f"Average Rating: {r.get('average_rating', 'N/A')}\n"
            f"Number of Ratings: {r.get('rating_number', 'N/A')}\n"
            f"ASIN: {parent_asin}"
        )

    @tool(parse_docstring=True)
    def Suggest_product(query: str, top_n: int = 3) -> str:
        """
        Suggest highly-rated products matching a keyword.

        Args:
            query: Search keyword.
            top_n: Maximum number of products.
        """

        matches = df[
            df[title_col]
            .fillna("")
            .str.contains(query, case=False, regex=False)
        ]

        if matches.empty:
            return f"No products found matching '{query}'."

        matches = matches.drop_duplicates("parent_asin")

        if "rating_number" in matches.columns:
            matches = matches.sort_values(
                by=["average_rating", "rating_number"],
                ascending=[False, False],
            )
        else:
            matches = matches.sort_values(
                by="average_rating",
                ascending=False,
            )

        matches = matches.head(top_n)

        results = []

        for i, (_, row) in enumerate(matches.iterrows(), 1):

            rating = row.get("average_rating")
            if pd.isna(rating):
                rating = "N/A"

            reviews = row.get("rating_number", "N/A")
            price = row.get("price", "Unknown")

            results.append(
                f"{i}. {row[title_col]}\n"
                f"   ASIN: {row.parent_asin}\n"
                f"   Rating: {rating}\n"
                f"   Reviews: {reviews}\n"
                f"   Price: {price}"
            )

        return "\n\n".join(results)

    @tool(parse_docstring=True)
    def Analyse_Suggestion(parent_asin: str) -> str:
        """
        Analyse customer reviews using Gemini.

        Args:
            parent_asin: Product identifier.
        """

        reviews = (
            df.loc[df["parent_asin"] == parent_asin, "text"]
            .dropna()
            .head(3)
            .tolist()
        )

        if not reviews:
            return f"No reviews available for '{parent_asin}'."

        reviews_text = "\n\n".join(reviews)

        prompt = (
            "You are an Amazon review analyst.\n\n"
            "Read the reviews and answer using exactly this format:\n\n"
            "Overall Sentiment: <Positive/Negative/Mixed>\n"
            "Pros: <comma separated>\n"
            "Cons: <comma separated>\n"
            "Recommendation: <one short sentence>\n\n"
            "Keep the response under 60 words.\n\n"
            "Reviews:\n"
            f"{reviews_text}"
        )

        try:
            response = llm.invoke(prompt)

            if hasattr(response, "content"):
                return response.content.strip()

            return str(response)

        except Exception as e:
            return f"Review analysis failed: {e}"

    return [
        Fetch_product,
        Suggest_product,
        Analyse_Suggestion,
    ]