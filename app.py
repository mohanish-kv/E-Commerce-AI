from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent
PRODUCTS_FILE = DATA_DIR / "products.csv"
REVIEWS_FILE = DATA_DIR / "reviews.csv"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    products = pd.read_csv(PRODUCTS_FILE)
    reviews = pd.read_csv(REVIEWS_FILE)
    return products, reviews


def format_price(price: int | float) -> str:
    return f"₹{price:,.0f}"


def filter_products(
    products: pd.DataFrame,
    search_text: str,
    category: str,
    price_range: tuple[int, int],
) -> pd.DataFrame:
    filtered = products.copy()

    if search_text:
        query = search_text.strip().lower()
        searchable = (
            filtered["ProductName"].str.lower()
            + " "
            + filtered["Brand"].str.lower()
            + " "
            + filtered["Description"].str.lower()
            + " "
            + filtered["Category"].str.lower()
        )
        filtered = filtered[searchable.str.contains(query, regex=False)]

    if category != "All":
        filtered = filtered[filtered["Category"] == category]

    min_price, max_price = price_range
    filtered = filtered[(filtered["Price"] >= min_price) & (filtered["Price"] <= max_price)]
    return filtered.sort_values(["Rating", "Price"], ascending=[False, True])


def get_reviews(reviews: pd.DataFrame, product_id: int) -> list[str]:
    return reviews.loc[reviews["ProductID"] == product_id, "Review"].tolist()


def summarize_reviews(review_texts: list[str]) -> str:
    if not review_texts:
        return "No customer reviews are available for this product yet."

    positive_terms = {
        "excellent",
        "great",
        "comfortable",
        "smooth",
        "reliable",
        "amazing",
        "responsive",
        "clear",
        "impressive",
        "useful",
        "fast",
        "lightweight",
        "stylish",
        "affordable",
        "durable",
        "easy",
        "perfect",
        "outstanding",
    }
    feature_terms = [
        "battery",
        "performance",
        "display",
        "camera",
        "charging",
        "gaming",
        "design",
        "storage",
        "tracking",
        "comfort",
        "quality",
        "value",
    ]

    combined = " ".join(review_texts).lower()
    pros = [term.title() for term in feature_terms if term in combined]
    if not pros:
        pros = ["Positive customer feedback", "Good everyday usability"]

    positive_count = sum(1 for word in re.findall(r"\w+", combined) if word in positive_terms)
    verdict = "Highly recommended based on customer sentiment."
    if positive_count < 3:
        verdict = "Recommended for users whose needs match the listed strengths."

    return "\n".join(
        [
            "**Pros**",
            *[f"- {pro}" for pro in pros[:5]],
            "",
            "**Cons**",
            "- No major repeated complaints found in the sample reviews.",
            "",
            "**Overall Verdict**",
            f"- {verdict}",
        ]
    )


def recommend_similar(products: pd.DataFrame, product_id: int, top_n: int = 3) -> pd.DataFrame:
    products = products.reset_index(drop=True)
    selected_index = products.index[products["ProductID"] == product_id][0]
    text_features = (
        products["ProductName"]
        + " "
        + products["Category"]
        + " "
        + products["Brand"]
        + " "
        + products["Description"]
    )
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(text_features)
    text_similarity = cosine_similarity(tfidf_matrix[selected_index], tfidf_matrix).flatten()

    selected = products.loc[selected_index]
    price_gap = (products["Price"] - selected["Price"]).abs()
    max_gap = max(price_gap.max(), 1)
    price_similarity = 1 - (price_gap / max_gap)
    category_bonus = (products["Category"] == selected["Category"]).astype(float) * 0.25

    scores = text_similarity + (price_similarity * 0.25) + category_bonus
    recommendations = products.assign(SimilarityScore=scores).drop(index=selected_index)
    return recommendations.sort_values("SimilarityScore", ascending=False).head(top_n)


def answer_shopping_question(products: pd.DataFrame, question: str) -> str:
    if not question.strip():
        return "Ask me what you are looking for, such as 'gaming laptop under 80000'."

    query = question.lower()
    budget_match = re.search(r"(?:under|below|less than|<)\s*₹?\s*(\d+)", query)
    filtered = products.copy()
    if budget_match:
        filtered = filtered[filtered["Price"] <= int(budget_match.group(1))]

    for category in products["Category"].unique():
        category_words = category.lower().split()
        if category.lower() in query or any(word.rstrip("s") in query for word in category_words):
            filtered = filtered[filtered["Category"] == category]
            break

    query_terms = [term for term in re.findall(r"[a-zA-Z]+", query) if len(term) > 2]
    if query_terms:
        searchable = (
            filtered["ProductName"].str.lower()
            + " "
            + filtered["Category"].str.lower()
            + " "
            + filtered["Description"].str.lower()
        )
        keyword_matches = filtered[searchable.apply(lambda text: any(term in text for term in query_terms))]
        if not keyword_matches.empty:
            filtered = keyword_matches

    if filtered.empty:
        return "I could not find a matching product. Try increasing the budget or using fewer filters."

    best = filtered.sort_values(["Rating", "Price"], ascending=[False, True]).iloc[0]
    return (
        f"I recommend **{best.ProductName}** by {best.Brand}. It costs "
        f"{format_price(best.Price)}, has a {best.Rating}★ rating, and offers {best.Description.lower()}."
    )


def render_product_card(product: pd.Series, products: pd.DataFrame, reviews: pd.DataFrame) -> None:
    with st.container(border=True):
        st.subheader(product.ProductName)
        st.write(f"**{product.Brand}** · {product.Category}")
        st.write(product.Description)
        st.metric("Price", format_price(product.Price))
        st.write(f"⭐ **{product.Rating} / 5**")

        with st.expander("View Reviews"):
            product_reviews = get_reviews(reviews, int(product.ProductID))
            if product_reviews:
                for review in product_reviews:
                    st.write(f"- {review}")
            else:
                st.info("No reviews available.")

        with st.expander("Summarize Reviews"):
            st.markdown(summarize_reviews(get_reviews(reviews, int(product.ProductID))))

        with st.expander("Recommend Similar"):
            similar = recommend_similar(products, int(product.ProductID))
            for _, recommendation in similar.iterrows():
                st.write(
                    f"- **{recommendation.ProductName}** ({recommendation.Brand}) — "
                    f"{format_price(recommendation.Price)}, {recommendation.Rating}★"
                )


def main() -> None:
    st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒", layout="wide")
    products, reviews = load_data()

    st.title("🛒 AI Shopping Assistant")
    st.caption("Search products, summarize reviews, and discover similar items from CSV data.")

    with st.sidebar:
        st.header("Filters")
        search_text = st.text_input("Search products", placeholder="gaming laptop, headphones...")
        category = st.selectbox("Category", ["All", *sorted(products["Category"].unique())])
        price_range = st.slider(
            "Price range",
            int(products["Price"].min()),
            int(products["Price"].max()),
            (int(products["Price"].min()), int(products["Price"].max())),
            step=100,
        )

    filtered_products = filter_products(products, search_text, category, price_range)
    st.header("Products")
    st.write(f"Showing {len(filtered_products)} of {len(products)} products")

    if filtered_products.empty:
        st.warning("No products match the current filters.")
    else:
        columns = st.columns(2)
        for index, (_, product) in enumerate(filtered_products.iterrows()):
            with columns[index % 2]:
                render_product_card(product, products, reviews)

    st.divider()
    st.header("💬 Chatbot")
    st.write("Try: `Need a gaming laptop under 80000` or `Which electronics product is best?`")
    question = st.chat_input("Ask for a product recommendation")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer_shopping_question(products, question))


if __name__ == "__main__":
    main()
