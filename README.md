# Clustering Analysis of Countries' Financial Vulnerability to Climate Adaptation - 2024

This repository contains a Streamlit application that performs clustering analysis on country-level governance, finance, and climate data. The goal is to identify clusters of countries based on their financial vulnerability to climate adaptation.

## Features

- **Data Integration**: Combines governance, finance, and climate data for a comprehensive analysis.
- **Scalable Clustering**: Supports user-defined category selection for clustering.
- **Dimensionality Reduction**: Includes PCA to optimize feature space.
- **Interactive Visualizations**:
  - Explained Variance and Elbow Method for PCA and clustering.
  - Geospatial maps for cluster distribution.
  - Radar charts for cluster-specific insights.
  - Correlation heatmaps and boxplots for feature exploration.
- **Customizable Analysis**: Users can select specific categories, features, and clusters for detailed exploration.

## Dataset

The application uses three primary datasets:
1. **Governance Index**
2. **Finance Index**
3. **Climate Index**

These datasets are merged on the `Country` column for analysis.

