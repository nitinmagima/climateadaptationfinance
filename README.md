# Clustering Analysis Dashboard: Understanding Countries' Financial Vulnerability to Climate Adaptation**

Climate change poses significant challenges to countries worldwide, requiring robust governance, sustainable financial systems, and adaptive resilience strategies. This dashboard provides tools to analyze and visualize the financial vulnerabilities of nations as they navigate the complexities of climate adaptation. By leveraging publicly available data on governance, finance, and climate metrics, this platform applies clustering techniques to group countries with shared vulnerability profiles.

This repository contains a Streamlit application that performs the clustering analysis. The goal is to identify clusters of countries based on their financial vulnerability to climate adaptation.

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
