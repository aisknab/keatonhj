"""Generate a 3D KNN visualization of a synthetic e-commerce feed."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px


# ---------------------------------------------------------------------------
# Generate a synthetic e-commerce product feed with realistic fields
# ---------------------------------------------------------------------------

categories = {
    "mens clothing": ["shirts", "shoes", "jackets", "pants"],
    "womens clothing": ["dresses", "shoes", "bags", "tops"],
    "electronics": ["phones", "laptops", "headphones", "cameras"],
    "home & kitchen": ["cookware", "furniture", "bedding", "decor"],
    "sports": ["fitness", "outdoor", "team sports", "cycling"],
}

brands = ["Acme", "Globex", "Umbrella", "Soylent", "Initech"]
conditions = ["new", "used", "refurbished"]

np.random.seed(42)
products = []

for category, subs in categories.items():
    for sub in subs:
        for _ in range(20):  # twenty products per subcategory (~400 total)
            price = np.round(np.random.uniform(5, 500), 2)
            brand = np.random.choice(brands)
            condition = np.random.choice(conditions)
            rating = np.round(np.random.uniform(1, 5), 1)
            title = f"{brand} {sub} {np.random.randint(1000, 9999)}"
            products.append(
                (
                    title,
                    category,
                    sub,
                    brand,
                    condition,
                    price,
                    rating,
                )
            )

product_df = pd.DataFrame(
    products,
    columns=["title", "category", "subcategory", "brand", "condition", "price", "rating"],
)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

encoder = OneHotEncoder()
encoded = encoder.fit_transform(
    product_df[["category", "subcategory", "brand", "condition"]]
).toarray()

price_scaled = product_df[["price"]].values / product_df["price"].max()
rating_scaled = product_df[["rating"]].values / 5.0

features = np.hstack([encoded, price_scaled, rating_scaled])


# ---------------------------------------------------------------------------
# Dimensionality reduction and clustering
# ---------------------------------------------------------------------------

embedded = TSNE(n_components=3, random_state=42, perplexity=30).fit_transform(features)

n_clusters = 8
clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(features)

product_df["x"], product_df["y"], product_df["z"] = embedded.T
product_df["cluster"] = clusters.astype(str)

product_df["detail"] = (
    product_df["category"]
    + " > "
    + product_df["subcategory"]
    + "; Brand: "
    + product_df["brand"]
    + "; Condition: "
    + product_df["condition"]
    + "; Price: $"
    + product_df["price"].astype(str)
    + "; Rating: "
    + product_df["rating"].astype(str)
)


# ---------------------------------------------------------------------------
# 3D visualization with Plotly
# ---------------------------------------------------------------------------

fig = px.scatter_3d(
    product_df,
    x="x",
    y="y",
    z="z",
    color="cluster",
    hover_name="title",
    hover_data={
        "category": True,
        "subcategory": True,
        "brand": True,
        "condition": True,
        "price": True,
        "rating": True,
    },
    color_discrete_sequence=px.colors.qualitative.Bold,
)

fig.update_traces(marker=dict(size=4))
fig.update_layout(title="3D KNN Visualization of Products")

fig.write_html("product_knn.html", include_plotlyjs="cdn")

