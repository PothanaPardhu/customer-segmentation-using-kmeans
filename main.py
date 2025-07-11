import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation using K-Means Clustering")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load default data
if uploaded_file is None:
    st.info("No file uploaded. Using default Mall_Customers.csv")
    data = pd.read_csv("Mall_Customers.csv")
else:
    data = pd.read_csv(uploaded_file)

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(data.head())

# Data Cleaning
st.subheader("Data Quality Check")
st.write("Missing Values:")
st.write(data.isnull().sum())

duplicates = data.duplicated().sum()
st.write(f"Duplicate Rows: {duplicates}")

# EDA Section
if st.checkbox("Show Exploratory Data Analysis"):
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    st.subheader("Gender Distribution")
    gender_counts = data['Gender'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'pink'])
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("Feature Distributions")
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(data['Age'], kde=True, ax=ax2[0])
    sns.histplot(data['Annual Income (k$)'], kde=True, ax=ax2[1])
    sns.histplot(data['Spending Score (1-100)'], kde=True, ax=ax2[2])
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(6,4))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

# Feature Selection
st.subheader("Select Features for Clustering")
features = st.multiselect("Choose at least two features:", 
                          options=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], 
                          default=['Annual Income (k$)', 'Spending Score (1-100)'])

if len(features) >= 2:
    X = data[features].values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose clusters
    k = st.slider("Select number of clusters (k)", 2, 10, 5)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    data['Cluster'] = labels

    # Silhouette Score
    sil_score = silhouette_score(X_scaled, labels)
    st.write(f"Silhouette Score: **{sil_score:.2f}**")

    # 2D Scatter Plot
    fig4, ax4 = plt.subplots(figsize=(6,4))
    scatter = ax4.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis')
    centers = kmeans.cluster_centers_
    ax4.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=100, label='Centroids')
    ax4.set_xlabel(features[0])
    ax4.set_ylabel(features[1])
    ax4.set_title("Customer Segments")
    ax4.legend()
    st.pyplot(fig4)

    # 3D Scatter Plot (if 3 features)
    if len(features) == 3:
        fig5 = plt.figure(figsize=(8,6))
        ax5 = fig5.add_subplot(111, projection='3d')
        ax5.scatter(X_scaled[:,0], X_scaled[:,1], X_scaled[:,2], c=labels, cmap='viridis', s=30)
        ax5.set_xlabel(features[0])
        ax5.set_ylabel(features[1])
        ax5.set_zlabel(features[2])
        st.pyplot(fig5)

    # Cluster Profiling
    st.subheader("Cluster Profiling Summary")
    cluster_summary = data.groupby('Cluster')[features].mean()
    st.write(cluster_summary)

    # Download Button
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", csv, "clustered_customers.csv", "text/csv")

else:
    st.warning("Please select at least two features.")

# Project Explanation
st.markdown("""
### About Customer Segmentation
Customer segmentation helps businesses divide their customers into distinct groups based on behavior and demographics. This allows targeted marketing, personalized offers, and improved customer service.

### How to Use this App:
1. Upload your customer dataset or use the default.
2. Explore the data with visualizations.
3. Select features and choose the number of clusters.
4. View results and download clustered data for business insights.
""")
