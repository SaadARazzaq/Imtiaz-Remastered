# Imtiaz Remastered: Data Science for Supermarket Customer Retention

![image](https://github.com/user-attachments/assets/638c7451-dd0f-4a97-ad67-1a2db2e2c390)

This repository contains the code and analysis for the **Imtiaz Remastered** project, aimed at enhancing customer retention in the electronics section of Imtiaz Mall through data-driven insights. The project leverages clustering techniques to segment customers based on purchasing behavior, providing actionable strategies for targeted marketing.

## **Project Overview**

The objective of this project is to segment customers into distinct groups based on their purchasing patterns. These segments are then used to tailor marketing strategies that can improve customer retention and drive sales growth.

## **Whats in the code?**

    - **Data Acquisition and Preprocessing**: Cleaning, transformation, and feature engineering of raw sales data.
    - **Exploratory Data Analysis (EDA)**: Detailed analysis of customer behavior through various visualizations.
    - **Clustering Analysis**: Application and comparison of `K-Means`, `DBSCAN`, and `K-Means++ clustering` algorithms.
    - **Recommendations**: Actionable insights and marketing strategies based on the analysis.

## **Key Steps**

1. **Data Acquisition and Preprocessing**: 
    - Imported and cleaned the raw data.
    - Transformed and engineered features to better understand customer behavior.
    
    Example code:
    ```python
    import numpy as np
    import pandas as pd

    # Loading data
    data = pd.read_csv('customer_data.csv')

    # Data cleaning
    data = data.dropna()

    # Feature engineering
    data['TotalSpend'] = data['Quantity'] * data['UnitPrice']
    ```

2. **Exploratory Data Analysis (EDA)**:
    - Performed univariate, bivariate, and temporal analyses to uncover key trends.
    - Visualized customer purchase behavior using plots and charts.
    
    Example code:
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Distribution of Total Spend
    plt.figure(figsize=(10, 6))
    sns.histplot(data['TotalSpend'], kde=True)
    plt.title('Distribution of Total Spend')
    plt.show()
    ```

3. **Clustering Analysis**:
    - Applied K-Means, DBSCAN, and K-Means++ algorithms to segment customers.
    - Compared clustering results to determine the best-performing model.
    
    Example code:
    ```python
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data[['TotalSpend', 'Frequency']])
    silhouette_avg = silhouette_score(data[['TotalSpend', 'Frequency']], clusters)
    print(f'K-Means Silhouette Score: {silhouette_avg}')
    ```

4. **Recommendations**:
    - Provided actionable insights based on the identified customer segments.
    - Suggested strategies for personalized marketing and improved customer retention.

## **Installation and Usage**

To replicate the analysis or run it on your local machine, follow these steps:

1. Clone this repository:
    ```python
    git clone https://github.com/saadarazzaq/imtiaz-remastered.git
    ```

2. Install the necessary dependencies from `requirements.txt`:
    ```python
    pip install -r requirements.txt
    ```

3. Launch the Jupyter Notebook to explore the analysis:
    ```python
    jupyter notebook src.ipynb
    ```

## **Performance of Algorithms**

- Cluster Shapes and Separation:
    - K-Means and K-Means++ seem to create relatively well-defined and separated cluster with distinct shapes.
    - DBSCAN have identified some noise points as a separate cluster.
- Cluster Sizes and Distribution:
    - K-Means and K-Means++ have clusters of roughly similar sizes.
    - DBSCAN might have clusters of varying densities and sizes.
- Outliers:
    - K-Means and K-Means++ might be sensitive to outliers, potentially pulling cluster centers towards them.
    - DBSCAN is generally more robust to outliers, as it can classify them as noise.
    
## **Results and Impact**

Through this analysis, Imtiaz Mall gained valuable insights into customer purchasing behavior. The segmentation enabled the creation of targeted marketing strategies aimed at retaining customers and increasing sales in the electronics section.

## **Conclusion and Recommendations**

*Customer Segments in the Electronics Section:*

- *Segment 1 - Young Trendsetters (Age: 17.9-42.8)*
  - *Product Preferences:* Primarily favor "Low" preference products.
  - *Purchase Behavior:* Likely prioritize affordability, trendiness, or entry-level features.
  
- *Segment 2 - Middle-Aged Balancers (Age: 42.8-55.2)*
  - *Product Preferences:* Balanced distribution with a slight inclination towards "Medium" and "High" preference products.
  - *Purchase Behavior:* Consider factors like quality, functionality, and brand reputation significantly.

- *Segment 3 - Senior Enthusiasts (Age: 55.2-80.0)*
  - *Product Preferences:* Strong inclination towards "High" preference products.
  - *Purchase Behavior:* Prioritize reliability, durability, advanced features, or specific needs related to accessibility or health.

*Differentiating Factors & Purchase Behavior Patterns:*

- *Income Influence:* Higher-income groups consistently spend more on electronics.
- *Brand Affinity:* Brand C remains the most popular, especially during winter and fall.
- *Seasonal Variation:* Winter and fall months observe peak sales, particularly for "High" preference products.

*Strategies for Customer Retention & Growth:*

1. *Personalized Recommendations:* Leverage age-specific preferences for targeted product recommendations and marketing strategies.
2. *Seasonal Campaigns:* Create promotional offers aligned with seasonal trends, especially during peak purchase months.
3. *Brand Partnerships:* Strengthen collaborations with Brand C, focusing on its popularity during peak sales seasons.

*Clustering Analysis Applications:*

- *Targeted Marketing:* Tailor marketing campaigns based on identified segments for maximum impact and relevance.
- *Loyalty Programs:* Design loyalty programs catering to each segment's preferences for increased engagement.

*Personalized Product Recommendations:*
  * Leverage cluster insights to recommend products tailored to each customer segment's preferences.
  * Utilize collaborative filtering or content-based recommendation systems trained on purchase history and product attributes.
  * Offer Tech Enthusiasts new arrivals and trendy gadgets.
  * Suggest functional and reliable options for Balanced Buyers.
  * Recommend premium and advanced features for Quality Seekers.
  * Highlight seasonal deals and gift ideas for Seasonal Shoppers.

By implementing these recommendations and continuously analyzing customer behavior, the electronics section can enhance customer retention, drive sales growth, and optimize its marketing and product offerings for each segment.

