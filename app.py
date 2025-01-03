import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Climate Adaptation Finance 2024",
    page_icon="🌍",
    layout="centered",   # Optional: Choose 'centered' or 'wide' layout
)

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
print(numeric_columns)
climate_df[numeric_columns] = scaler.fit_transform(climate_df[numeric_columns])

# Merge the datasets on the 'Country' column
merged_df = governance_df.merge(finance_df, on="Country").merge(climate_df, on="Country")

# Display the content
st.markdown("""
# Understanding Countries' Financial Vulnerability to Climate Adaptation - Clustering Analysis 2024

Climate change poses significant challenges to countries worldwide, requiring robust governance, sustainable financial systems, and adaptive resilience strategies. This dashboard provides tools to analyze and visualize the financial vulnerabilities of nations as they navigate the complexities of climate adaptation. By leveraging publicly available data on governance, finance, and climate metrics, this platform applies clustering techniques to group countries with shared vulnerability profiles.

### What is Financial Vulnerability to Climate Adaptation?

Financial vulnerability in the context of this dashboard refers to a country's capacity to mobilize and allocate resources to adapt to the adverse effects of climate change. It encompasses:
- **Financial Stability**: The resilience of financial markets, public debt levels, and economic diversification.
- **Governance Quality**: The efficiency, transparency, and adaptability of institutions managing climate risks.
- **Climate Exposure**: The degree of risk posed by climate hazards.

### Key Features of the Dashboard:

1. **Data Integration**:
   - Combines governance indicators (e.g., government effectiveness), financial metrics (e.g., debt-to-GDP ratio), and climate data (e.g., INFORM risk).
   - Offers a comprehensive perspective on the interplay between these critical domains.

2. **Data Transformation**:
   - Scales numeric features to a standardized range (0 to 100), where:
     - **100**: Indicates the **most vulnerable** or least favorable conditions.
     - **0**: Represents the **least vulnerable** or most favorable conditions.
   - Missing values are imputed using the column median to ensure consistency and completeness.

3. **Clustering Techniques**:
   - Groups countries based on shared characteristics using **K-Means Clustering**.
   - Allows for the exploration of vulnerability patterns and identification of peer groups for benchmarking.

4. **Principal Component Analysis (PCA)**:
   - Reduces data complexity while retaining key information.
   - Enables users to select the optimal number of components to balance interpretability and data accuracy.

5. **Geospatial Visualization**:
   - Interactive maps reveal spatial patterns in vulnerability, highlighting regional trends and hotspots.
   - Facilitates targeted analysis of clusters with similar governance, financial, and climate vulnerabilities.

6. **Advanced Insights**:
   - **Feature Distribution by Cluster**: Understand which factors contribute most to vulnerability across clusters.
   - **Radar Charts**: Visualize the strengths and weaknesses of clusters across governance, finance, and climate metrics.
   - **Correlation Analysis**: Explore how governance, finance, and climate metrics interact within clusters, revealing key dependencies.

7. **Variable Exploration**:
   - Drill down into specific variables or categories to gain detailed insights.
   - Compare country profiles within clusters to identify unique strengths or critical vulnerabilities.

### Why This Matters:

Understanding financial vulnerability to climate adaptation is essential for:
- **Policy Design**: Informing governments and organizations about resource allocation and strategic priorities.
- **International Cooperation**: Identifying countries with similar challenges to foster collaboration and shared learning.
- **Investment Decisions**: Guiding stakeholders in prioritizing climate-resilient projects and interventions.

### How to Get Started:

1. **Select Categories**:
   Focus your analysis by choosing governance, finance, and/or climate data.

2. **Optimize Parameters**:
   - Use **PCA Explained Variance** to refine the dimensionality of your dataset.
   - Apply the **Elbow Method** to determine the optimal number of clusters.

3. **Visualize and Explore**:
   - Navigate geospatial maps to understand global and regional trends.
   - Analyze feature distributions and radar charts for detailed cluster insights.
   - Examine cross-category correlations and delve into specific variables for nuanced understanding.
""")

st.write("Further documentation of variables can be found [here](https://google.com).")

# Allow user to select categories for clustering
st.subheader("Select Categories for Clustering")
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

# Perform PCA and K-Means Clustering
# Drop the "Country" column for clustering
clustering_data = combined_df.drop(columns=["Country"])

# Handle missing values by imputing with column medians
clustering_data_imputed = clustering_data.fillna(clustering_data.median())

# Normalize numeric features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(clustering_data_imputed)

# Optimize the number of PCA components
explained_variance = []
for n in range(1, min(len(normalized_features[0]), 11)):
    pca = PCA(n_components=n)
    pca.fit(normalized_features)
    explained_variance.append(sum(pca.explained_variance_ratio_))

# Plot Explained Variance to Find Optimal Components
st.subheader("Explained Variance for PCA Components")
st.write("Principal Component Analysis (PCA) reduces the dimensionality of the dataset by transforming it into a smaller set of components while retaining most of the variance. The explained variance indicates how much information (variance) each principal component captures. This section helps you identify the optimal number of components to balance simplicity and data representation.")
pca_fig = go.Figure()
pca_fig.add_trace(go.Scatter(x=list(range(1, len(explained_variance) + 1)), y=explained_variance, mode='lines+markers', name='Explained Variance'))
pca_fig.update_layout(title="PCA Explained Variance", xaxis_title="Number of Components", yaxis_title="Cumulative Explained Variance", template="plotly_dark")
st.plotly_chart(pca_fig)

# Use optimal number of components for PCA
optimal_components = st.slider("Select Number of PCA Components", min_value=1, max_value=len(explained_variance), value=2)
pca = PCA(n_components=optimal_components)
pca_features = pca.fit_transform(normalized_features)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
cluster_range = range(1, 11)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method using Plotly
st.subheader("Elbow Method")

st.write("The Elbow Method is a technique used to determine the optimal number of clusters for K-Means clustering. It plots the sum of squared distances (inertia) between data points and their assigned cluster centers as the number of clusters increases. The 'elbow point,' where the rate of decrease slows significantly, suggests the best cluster count, balancing accuracy and simplicity.")


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
st.subheader("Geospatial Visualization of Clustering of Countries")

st.write("Geospatial maps allows us to visualize and analyze patterns spatially across countries by grouping them based on governance, finance, and climate vulnerabilities. This helps identify regional trends and relationships that are vital for targeted policy-making and resource allocation.")

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

# Short explanation about vulnerability scale
st.write("Note: In these  charts, a score of **100** represents the most vulnerable or least favorable conditions, while a score of **0** represents the least vulnerable or most favorable conditions.")

# Boxplots for Feature Distribution by Cluster
st.subheader("Feature Distribution by Cluster")
st.write("This visualization shows the distribution of selected features across different clusters. By examining the spread of values for each feature, users can gain insights into the characteristics that differentiate clusters and identify patterns within each cluster group.")
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

# Radar Charts Note
st.write("Radar charts provide a visual summary of how countries perform across various governance, finance, and climate metrics. Each axis represents a feature, and the filled area indicates the average values for the selected cluster, scaled between 0 (least vulnerable) and 100 (most vulnerable). These charts help identify strengths and vulnerabilities within clusters.")


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

st.write("This section examines the correlations between selected categories within a specific cluster. By analyzing these correlations, users can understand how features within governance, finance, or climate interact, identifying strong relationships or dependencies that might influence policy or decision-making.")


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

st.write("This section allows users to investigate specific variables within a selected cluster. By focusing on particular features, users can delve deeper into the detailed metrics and characteristics that define the cluster, aiding in nuanced analysis and targeted insights.")


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

# Add a copyright line at the bottom of the page
st.markdown(
    "<div style='text-align: center; margin-top: 50px; font-size: 12px; color: gray;'>"
    "© 2024 Columbia Climate School, Columbia University. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)
