import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div
from bokeh.layouts import column
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap

# Generate a dummy e-commerce product feed
categories = {
    'mens clothing': ['shirts', 'shoes', 'jackets'],
    'womens clothing': ['dresses', 'shoes', 'bags'],
    'electronics': ['phones', 'laptops', 'headphones']
}

np.random.seed(42)
products = []
for cat, subs in categories.items():
    for sub in subs:
        for _ in range(5):  # five products per subcategory
            price = np.random.randint(20, 200)
            products.append((cat, sub, price))

# Create DataFrame
product_df = pd.DataFrame(products, columns=['category', 'subcategory', 'price'])

# Feature engineering
encoder = OneHotEncoder()
encoded = encoder.fit_transform(product_df[['category', 'subcategory']]).toarray()
prices = product_df[['price']].values
features = np.hstack([encoded, prices / prices.max()])  # normalize price

# Dimensionality reduction for visualization
embedded = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(features)

# Clustering for color coding
n_clusters = 5
clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(features)

# Prepare data for Bokeh
product_df['x'] = embedded[:, 0]
product_df['y'] = embedded[:, 1]
product_df['cluster'] = clusters.astype(str)
product_df['detail'] = (
    product_df['category'] + ' > ' + product_df['subcategory'] + '; price: $' + product_df['price'].astype(str)
)

source = ColumnDataSource(product_df)

# Create plot
p = figure(title="KNN Visualization of Products", tools="pan,wheel_zoom,reset,tap,hover", width=800, height=600)
palette = Category10[n_clusters]
p.scatter('x', 'y', source=source,
          color=factor_cmap('cluster', palette=palette, factors=[str(i) for i in range(n_clusters)]),
          size=10)

hover = p.select_one(HoverTool)
hover.tooltips = [("Product", "@detail")]

# Display details on click
info_div = Div(width=400, height=100)
callback = CustomJS(args=dict(source=source, div=info_div), code="""
    const index = cb_obj.indices[0];
    if (index != null) {
        const data = source.data;
        const desc = data['detail'][index];
        div.text = desc;
    }
""")

p.js_on_event('tap', callback)
layout = column(p, info_div)

output_file("product_knn.html")
show(layout)
