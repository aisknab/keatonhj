"""Generate a 3D KNN visualization of a synthetic e-commerce feed."""

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
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

# Determine k-nearest neighbor pairs in the embedded space
parser = argparse.ArgumentParser(description="KNN visualization options")
parser.add_argument(
    "--k", type=int, default=5, help="Number of nearest neighbors to connect"
)
args = parser.parse_args()
k = args.k

neighbors = NearestNeighbors(n_neighbors=k).fit(embedded)
neighbor_indices = neighbors.kneighbors(embedded, return_distance=False)

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
    # Bright palette that pops against a dark background
    color_discrete_sequence=px.colors.qualitative.Vivid,
)

# Make markers easier to see on a dark canvas
fig.update_traces(
    marker=dict(size=5, line=dict(width=0.5, color="white"))
)


# Dark mode polish
fig.update_layout(
    title="3D KNN Visualization of Products",
    template="plotly_dark",
    paper_bgcolor="#000",
    scene=dict(
        bgcolor="#111",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        zaxis=dict(showgrid=False, zeroline=False),
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)")
)

# Add interactive neighbor highlighting via JavaScript
post_script = f"""
var neighborIndices = {neighbor_indices.tolist()};
var MAX_DEPTH = 3;
var currentLineTrace = null;
var gd = document.getElementById('knn-viz');
gd.on('plotly_click', function(event) {{
  if (currentLineTrace !== null) {{
    Plotly.deleteTraces(gd, currentLineTrace);
    currentLineTrace = null;
  }}
  var start = event.points[0].pointIndex;
  var visited = new Set([start]);
  var queue = [{{index: start, depth: 0}}];
  var xs = [], ys = [], zs = [];
  while (queue.length > 0) {{
    var node = queue.shift();
    var idx = node.index;
    var depth = node.depth;
    if (depth >= MAX_DEPTH) continue;
    var neigh = neighborIndices[idx];
    for (var i = 1; i < neigh.length; i++) {{
      var nb = neigh[i];
      xs.push(gd.data[0].x[idx], gd.data[0].x[nb], null);
      ys.push(gd.data[0].y[idx], gd.data[0].y[nb], null);
      zs.push(gd.data[0].z[idx], gd.data[0].z[nb], null);
      if (!visited.has(nb)) {{
        visited.add(nb);
        queue.push({{index: nb, depth: depth + 1}});
      }}
    }}
  }}
  if (xs.length > 0) {{
    Plotly.addTraces(gd, {{
      type: 'scatter3d',
      mode: 'lines',
      x: xs,
      y: ys,
      z: zs,
      line: {{color: 'rgba(200,200,200,0.4)', width: 2}},
      showlegend: false
    }});
    currentLineTrace = gd.data.length - 1;
  }}
}});
"""

fig.write_html("product_knn.html", include_plotlyjs="cdn", post_script=post_script, div_id="knn-viz")

