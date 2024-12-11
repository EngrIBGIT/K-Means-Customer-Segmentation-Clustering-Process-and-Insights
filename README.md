# K-Means Customer Segmentation: Clustering Process and Insights

***[Notebook Link for K-Means Customer Segmentation](https://github.com/EngrIBGIT/K-Means-Customer-Segmentation-Clustering-Process-and-Insights/blob/main/K_Means%20Customer%20Segmentation%20Analysis_D.ipynb)***

***[K-Means Customer Segmentation Live App Link](https://k-means-customer-segmentation-clustering.onrender.com/)***

This report outlines the K-Means clustering process for customer segmentation, utilizing an unsupervised machine learning technique. The goal is to identify distinct groups of customers based on various attributes such as income, spending behavior, and other demographic factors. This can enable targeted marketing, personalized recommendations, and improved customer relationship management.


## 1. Data Overview

The dataset used for the clustering process is the **Customer Segmentation Dataset** from Kaggle, containing 2240 customers with attributes like:

- **Demographics**: Age, income, marital status, education level.
- **Purchase behaviors**: Spending on products like wine, fruits, meat, etc., number of deals purchased, web visits, and accepted campaigns.

### Key Data Attributes:

- **Income**: Customer income, a major factor in segmentation.
- **Spending**: Amount spent on different product categories.
- **Recency**: How recently the customer made a purchase.
- **Web Activity**: Number of web visits per month and catalog purchases.
- **Demographic**: Age, marital status, education level.

## 2. Data Preprocessing

### 2.1 Handling Missing Values

The dataset has some missing values, particularly in the Income column. These missing values were imputed using the **median** value, ensuring that there were no gaps in the data, which could negatively impact the clustering process.

### 2.2 Standardizing the Data

To ensure that all features have the same scale and avoid bias due to different ranges in values (e.g., income ranges from 17,300 to 666,666 while spending is in much smaller values), we used `StandardScaler` to standardize the features before applying the clustering algorithm.

### 2.3 Feature Selection and Transformation

**Dimensionality Reduction (PCA)** was applied to reduce the number of features and retain the most significant variance in the data. This step was necessary to ensure that the K-Means algorithm could operate efficiently without the curse of dimensionality.

## 3. K-Means Clustering

### 3.1 Choosing the Number of Clusters

To determine the optimal number of clusters, we used the **Elbow Method**. By plotting the **Within-Cluster Sum of Squares (WCSS)** for different values of k, we identified the point where the WCSS curve started to flatten. This indicated that adding more clusters beyond this point did not provide significant improvement.

From the **Elbow plot**, we selected **k=4** as the optimal number of clusters. This represents the most distinct and meaningful segmentation based on the variance explained by the clusters.

### 3.2 Clustering Execution

Using KMeans with **k=4**, we performed the clustering and assigned each customer to one of the four segments. These segments represent customer groups with similar behaviors and characteristics.

## 4. Cluster Analysis and Insights

### 4.1 Cluster Profiles

After clustering, we analyzed the characteristics of each cluster by examining the mean values of various features for each segment. The analysis revealed the following customer segments:

#### Cluster 0: High-Spending, Middle-Aged Customers

- **Profile**: High-income individuals who spend significantly on wine, meat, and fruits. They are in their 40s to 50s and have been customers for a long time.
- **Marketing Insight**: Target these customers with high-end, premium product offerings and loyalty programs.

#### Cluster 1: Low Income, Frequent Shoppers

- **Profile**: Younger customers with lower incomes but a high frequency of purchases. They often engage with promotions and are more likely to buy frequently at a low cost.
- **Marketing Insight**: Focus on budget-friendly promotions, bundle offers, and loyalty rewards to increase retention.

#### Cluster 2: Middle-Aged, Moderate Spending

- **Profile**: Customers with moderate income levels and spending habits. They typically purchase fruits, wines, and have moderate engagement with online promotions.
- **Marketing Insight**: Engage these customers with balanced offers and targeted discounts based on their moderate spending habits.

#### Cluster 3: Low Activity, Low Spending

- **Profile**: Customers who show low engagement, rarely purchase, and have lower spending levels. They are relatively inactive and may have a higher recency score, indicating they haven’t purchased recently.
- **Marketing Insight**: Re-engagement strategies such as personalized offers, reminders, or reactivation campaigns would be beneficial.

### 4.2 Cluster Visualization

Using **PCA (Principal Component Analysis)**, we visualized the customer segments in a 2D space. This visualization helped to confirm the distinctness of the clusters. It visually affirmed that the clusters were well-separated, meaning the K-Means algorithm performed well in segmenting the customers.

## 5. Conclusion and Actionable Insights

The K-Means clustering process provided valuable insights into the customer base. The customer segments identified can now be used for:

- **Targeted Marketing**: Tailoring marketing campaigns for each cluster based on their spending behavior, demographics, and activity levels.
- **Personalized Offers**: Offering special deals and offers to high-spending customers, while crafting budget-friendly promotions for price-sensitive customers.
- **Customer Retention**: Engaging inactive customers through strategies like personalized email marketing and loyalty rewards.


## Customer Segmentation Resources

### [K-Means Customer Segmentation Live App Link](https://k-means-customer-segmentation-clustering.onrender.com/)
Explore a live interactive app where you can upload your dataset, customize clustering parameters, and visualize customer segmentation insights. The app allows you to:

1. **Upload your dataset** in `.csv` or `.xlsx` format.
2. **Select features** for clustering, such as `Education`, `Income`, `Age`, etc.
3. **Customize the number of clusters** and visualize the results using advanced techniques like PCA.
4. **Download the processed data** and clustering results in formats like Excel, CSV, PDF, or DOC.

This app is user-friendly and ideal for exploring customer behavior patterns to inform marketing and business strategies.

---

### [Notebook Link for K-Means Customer Segmentation](https://github.com/EngrIBGIT/K-Means-Customer-Segmentation-Clustering-Process-and-Insights/blob/main/K_Means%20Customer%20Segmentation%20Analysis_D.ipynb)
This repository contains the complete Jupyter Notebook used for the K-Means Customer Segmentation analysis. It includes:

1. **Data Preparation**: Cleaning and preprocessing steps for raw datasets.
2. **Clustering Process**: Implementation of K-Means clustering and parameter tuning.
3. **Insights and Visualizations**: Detailed insights from clustering, including visualizations and metrics like silhouette scores.
4. **Code Walkthrough**: Fully documented Python code to replicate the process or adapt it for your own datasets.

Use this notebook to understand the methodology, replicate the analysis, or customize the clustering process for specific needs.



## 6. Future Steps

To further improve the segmentation:

- **Fine-tune the clustering model**: Experiment with other clustering algorithms like DBSCAN or hierarchical clustering to compare results.
- **Incorporate Additional Features**: Additional behavioral data such as customer feedback, website interactions, or customer service interactions could enhance segmentation.
- **Deployment**: The K-Means model can be deployed to **Heroku** or a similar platform for real-time customer segmentation in production environments.

By leveraging this clustering model, businesses can develop deeper insights into their customers, enhance marketing efforts, and ultimately improve customer satisfaction and loyalty.

