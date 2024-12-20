import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load the dataset
file_path = "data.xlsx"
data = pd.ExcelFile(file_path)

# Display sheet names to understand the structure
print("Sheet Names:", data.sheet_names)

# Load all three sheets into separate DataFrames
governance_df = data.parse('Governance Index')
finance_df = data.parse('Finance Index')
climate_df = data.parse('Climate Index')

# Apply Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 100))
numeric_columns = climate_df.select_dtypes(include=[float, int]).columns
climate_df[numeric_columns] = scaler.fit_transform(climate_df[numeric_columns])

# Merge the datasets on the 'Country' column
merged_df = governance_df.merge(finance_df, on="Country").merge(climate_df, on="Country")

# Allow user to select categories for clustering
st.title("Climate Adaptation Financial Vulnerability 2024 - Clustering Analysis")

# Short explanation about vulnerability scale
st.write("Note: In this analysis, all values are between 0 to 100. A score of **100** represents the most vulnerable conditions, while a score of **0** represents the least vulnerable conditions.")


st.write("Select one or more categories to include in clustering:")
selected_categories = st.multiselect(
    "Categories", ["Governance", "Finance", "Climate"], default=["Governance"]
)

# Create a combined DataFrame based on selected categories
selected_dfs = []
if "Governance" in selected_categories:
    selected_dfs.append(governance_df.set_index("Country"))
if "Finance" in selected_categories:
    selected_dfs.append(finance_df.set_index("Country"))
if "Climate" in selected_categories:
    selected_dfs.append(climate_df.set_index("Country"))

if selected_dfs:
    combined_df = pd.concat(selected_dfs, axis=1).reset_index()
else:
    st.error("Please select at least one category for clustering.")
    st.stop()

# Drop the "Country" column for clustering
clustering_data = combined_df.drop(columns=["Country"])

# Handle missing values by imputing with column medians
clustering_data_imputed = clustering_data.fillna(clustering_data.median())

# Normalize numeric features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(clustering_data_imputed)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
cluster_range = range(1, 11)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method using Plotly
st.subheader("Elbow Method")
elbow_figure = go.Figure()
elbow_figure.add_trace(go.Scatter(
    x=list(cluster_range),
    y=inertia,
    mode='lines+markers',
    marker=dict(size=8),
    name="Inertia"
))
elbow_figure.update_layout(
    title="Optimal Number of Clusters",
    xaxis_title="Number of Clusters",
    yaxis_title="Inertia",
    template="plotly_dark",
    title_x=0.5
)
st.plotly_chart(elbow_figure)

# Select the number of clusters (default to 4)
num_clusters = st.slider("Select Number of Clusters", 2, 10, value=4)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_features)

# Add cluster labels to the original DataFrame
combined_df["Cluster"] = cluster_labels

# Interactive Map
st.subheader("Geospatial Clustering of Countries")
country_geo_data = combined_df.copy()
country_geo_data["Cluster"] = country_geo_data["Cluster"].astype(str)  # Convert cluster to string for coloring

# Create map
map_figure = px.choropleth(
    country_geo_data,
    locations="Country",
    locationmode="country names",
    color="Cluster",
    title="   ",
    color_discrete_sequence=px.colors.diverging.Spectral,
    template="plotly_dark",
    labels={"Cluster": "Cluster Group"}
)
map_figure.update_geos(
    showcoastlines=True, coastlinecolor="White",
    showland=True, landcolor="Black",
    showocean=True, oceancolor="DarkBlue"
)
map_figure.update_layout(
    title_font_size=20,
    title_x=0.5,
    margin={"r":0,"t":40,"l":0,"b":0}
)
st.plotly_chart(map_figure)

# Boxplots for Feature Distribution by Cluster
st.subheader("Feature Distribution by Cluster")
feature = st.selectbox("Select a Feature", clustering_data_imputed.columns)
boxplot_figure = px.box(
    combined_df,
    x="Cluster",
    y=feature,
    points="all",
    color="Cluster",
    hover_data=["Country"],  # Include country names in tooltips
    title=f"Distribution of {feature} by Cluster",
    template="plotly_dark"
)
st.plotly_chart(boxplot_figure)

# Radar Charts for Governance, Finance, and Climate
st.subheader("Radar Charts")

# Select only numeric columns for aggregation
numeric_df = combined_df.select_dtypes(include=[np.number])
cluster_means = numeric_df.groupby(combined_df["Cluster"]).mean()

# Select a cluster
cluster_filter = st.selectbox("Select a Cluster", combined_df["Cluster"].unique())

# Governance Radar Chart
governance_features = [
    "Government Effectiveness", "Rule of Law", "Regulatory quality",
    "Political Stability and Absence of Violence/Terrorism", "Voice and accountability\nControl of Corruption",
    "Security Threats Index", "Factionalized Elites Index"
]
if "Governance" not in selected_categories:
    st.write("Governance data has not been selected.")
else:
    if all(feature in cluster_means for feature in governance_features):
        governance_data = cluster_means.loc[cluster_filter, governance_features]
        governance_data_values = np.append(governance_data.values, governance_data.values[0])
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=governance_data_values,
            theta=governance_features + [governance_features[0]],
            fill='toself',
            name='Governance',
            fillcolor="rgba(0, 128, 255, 0.5)",  # Semi-transparent blue
            line=dict(color="rgb(0, 128, 255)", width=2)  # Solid blue
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),  # Fixed range 0-100
            title="Governance Radar Chart",
            template="plotly_dark"
        )
        st.plotly_chart(fig)

# Finance Radar Chart
finance_categories = {
    "Debt Sustainability": [
        "Total Debt/GDP", "External Debt/GDP", "Debt/Exports", "Debt Service/Exports",
        "Debt/Budget Revenue", "Real interest rate-growth differential", "Short term debt/External Debt",
        "Short term debt/Reserves", "Debt Service/Revenue"
    ],
    "Financial Integration": [
        "Trade/Global Trade", "Current Account/GDP", "FDI/GDP", "Portfolio Equity Inflows/GDP",
        "Portfolio Debt Inflows/GDP", "Foreign Claims of Banks", "Trade Credit/GDP",
        "Membership in IFIs/Total IFIs"
    ],
    "Financial Sophistication": [
        "Domestic Credit to Private Sector/GDP", "Number of New Listed Companies/Global",
        "Market Cap /GDP", "Market Cap /Global Market Cap", "Turnover Ratio of Stock Exchanges/ Global"
    ]
}
if "Finance" not in selected_categories:
    st.write("Finance data has not been selected.")
else:
    finance_data = {}
    for category, columns in finance_categories.items():
        finance_data[category] = cluster_means[columns].mean(axis=1).loc[cluster_filter]
    finance_values = list(finance_data.values()) + [list(finance_data.values())[0]]
    finance_labels = list(finance_data.keys()) + [list(finance_data.keys())[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=finance_values,
        theta=finance_labels,
        fill='toself',
        name='Finance',
        fillcolor="rgba(0, 255, 128, 0.5)",
        line=dict(color="rgb(0, 255, 128)", width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),  # Fixed range 0-100
        title="Finance Radar Chart",
        template="plotly_dark"
    )
    st.plotly_chart(fig)

# Climate Radar Chart
climate_features = [
    "INFORM CC Risk Index 2022", "Infom Risk / CC Risk Change - 75/25",
    "Damages_MortalityCosts_RCP8_5", "Damages_MortalityCosts_RCP4_5", "popbelow5m(WDI)"
]
if "Climate" not in selected_categories:
    st.write("Climate data has not been selected.")
else:
    if all(feature in cluster_means for feature in climate_features):
        climate_data = cluster_means.loc[cluster_filter, climate_features]
        climate_data_values = np.append(climate_data.values, climate_data.values[0])
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=climate_data_values,
            theta=climate_features + [climate_features[0]],
            fill='toself',
            name='Climate',
            fillcolor="rgba(255, 165, 0, 0.5)",
            line=dict(color="rgb(255, 165, 0)", width=2)
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),  # Fixed range 0-100
            title="Climate Radar Chart",
            template="plotly_dark"
        )
        st.plotly_chart(fig)

# Heatmap for Correlation Analysis
st.subheader("Explore Correlation of Categories for Selected Cluster")

# Dropdowns to select categories for x and y axes
axis_options = {
    "Governance": governance_df.columns.tolist(),
    "Finance": list(finance_categories.keys()),  # Use only finance categories
    "Climate": climate_df.columns.tolist()
}
x_axis_category = st.selectbox("Select X-Axis Category", options=axis_options.keys(), index=0)
y_axis_category = st.selectbox("Select Y-Axis Category", options=axis_options.keys(), index=1)

# Filter numeric_df by the selected cluster
cluster_filtered_data = numeric_df[combined_df["Cluster"] == cluster_filter]

# Get selected columns for x and y axes
if x_axis_category == "Finance":
    x_columns = [col for cat in finance_categories.values() for col in cat if col in cluster_filtered_data.columns]
else:
    x_columns = [col for col in axis_options[x_axis_category] if col in cluster_filtered_data.columns]

if y_axis_category == "Finance":
    y_columns = [col for cat in finance_categories.values() for col in cat if col in cluster_filtered_data.columns]
else:
    y_columns = [col for col in axis_options[y_axis_category] if col in cluster_filtered_data.columns]

# Check if columns exist
if not x_columns or not y_columns:
    st.warning("Selected categories do not have matching numeric columns in the dataset for the selected cluster.")
else:
    # Subset cluster_filtered_data based on selected columns
    correlation_matrix = cluster_filtered_data[x_columns + y_columns].corr()

    # Mask the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    correlation_matrix_masked = correlation_matrix.copy()
    correlation_matrix_masked[mask] = np.nan  # Set upper triangle values to NaN

    # Create a Plotly heatmap
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix_masked.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale="Spectral",  # Updated to a valid colorscale
        zmin=-1,  # Set range for color scale
        zmax=1,  # Set range for color scale
        colorbar=dict(title="Correlation", thickness=20, len=0.5)
    ))

    heatmap_fig.update_layout(
        title=f"Correlation Heatmap for Cluster {cluster_filter} ({x_axis_category} vs {y_axis_category})",
        xaxis=dict(title="Features", tickangle=45, showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(title="Features", showgrid=False, tickfont=dict(size=10)),
        template="plotly_dark",
        margin=dict(l=100, r=100, t=50, b=100),
        height=600
    )

    # Display the heatmap
    st.plotly_chart(heatmap_fig)

# Interactive Filters and Dropdowns
st.subheader("Explore Variables for Selected Cluster")
governance_columns = [
    "Country", "Government Effectiveness", "Rule of Law", "Regulatory quality",
    "Political Stability and Absence of Violence/Terrorism", "Voice and accountability\nControl of Corruption",
    "Security Threats Index", "Factionalized Elites Index"
]
finance_columns = [
    "Country", "Total Debt/GDP", "External Debt/GDP", "Debt/Exports", "Debt Service/Exports", "Debt/Budget Revenue",
    "Real interest rate-growth differential", "Short term debt/External Debt", "Short term debt/Reserves",
    "Debt Service/Revenue", "Trade/Global Trade", "Current Account/GDP", "FDI/GDP", "Portfolio Equity Inflows/GDP",
    "Portfolio Debt Inflows/GDP", "Foreign Claims of Banks", "Trade Credit/GDP", "Membership in IFIs/Total IFIs",
    "Domestic Credit to Private Sector/GDP", "Number of New Listed Companies/Global", "Market Cap /GDP",
    "Market Cap /Global Market Cap", "Turnover Ratio of Stock Exchanges/ Global"
]
climate_columns = [
    "Country", "INFORM CC Risk Index 2022", "Change in risk", "Infom Risk / CC Risk Change - 50/50",
    "Infom Risk / CC Risk Change - 67/33", "Infom Risk / CC Risk Change - 75/25",
    "Damages_MortalityCosts_RCP8_5", "Damages_MortalityCosts_RCP4_5", "popbelow5m(WDI)"
]

category_options = {
    "Governance": governance_columns,
    "Finance": finance_columns,
    "Climate": climate_columns
}
selected_categories = st.multiselect("Select Categories", options=category_options.keys(), default=["Governance"])

# Combine columns from selected categories
selected_columns = []
for category in selected_categories:
    selected_columns.extend(category_options[category])

columns_to_include = st.multiselect(
    "Select Columns to Display", selected_columns, default=["Country"]
)
filtered_data = combined_df[combined_df["Cluster"] == cluster_filter][columns_to_include]
st.write(filtered_data)
