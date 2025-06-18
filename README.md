# K-means Clustering from Scratch

A complete implementation of the K-means clustering algorithm from scratch in Python, applied to loan applicant data for customer segmentation.

## 📋 Project Overview

This project implements K-means clustering without using scikit-learn's built-in functions to demonstrate a deep understanding of the algorithm. The goal is to classify loan applicants based on their income and loan amount to identify distinct customer segments.

## 🎯 Objective

To classify loan applicants into clusters based on:
- **Applicant Income**: The applicant's monthly income
- **Loan Amount**: The requested loan amount (in thousands)

## 📊 Dataset

The project uses `clustering.csv` which contains loan application data with the following relevant features:
- `ApplicantIncome`: Applicant's income
- `LoanAmount`: Requested loan amount
- Other features: Gender, Married status, Education, Employment type, etc.

**Dataset size**: 383 records

## 🛠️ Implementation Details

### Algorithm Steps

1. **Data Preparation**: Load and extract relevant features (ApplicantIncome, LoanAmount)
2. **Initialization**: Randomly select initial centroids from the dataset
3. **Iterative Process**:
   - Calculate Euclidean distance from each point to all centroids
   - Assign each point to the nearest centroid
   - Update centroids by calculating the mean of assigned points
   - Repeat until convergence (centroids stop moving)

### Key Features

- **From Scratch Implementation**: No scikit-learn clustering functions used
- **Convergence Detection**: Monitors centroid movement to detect algorithm completion
- **Visualization**: Multiple plots showing data distribution and clustering results
- **Prediction Function**: Ability to classify new applicants based on trained model

## 📈 Results

The algorithm identifies **3 distinct customer segments**:

1. **Cluster 1**: Low Income, Low Loan Applicants
2. **Cluster 2**: Medium Income, Medium Loan Applicants  
3. **Cluster 3**: High Income, High Loan Applicants

## 🚀 Usage

### Prerequisites

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
```

### Running the Code

1. **Load the data**:
```python
df = pd.read_csv('clustering.csv')
```

2. **Extract features**:
```python
X = df[['LoanAmount','ApplicantIncome']]
```

3. **Run K-means algorithm** (implemented in the main loop)

4. **Make predictions for new applicants**:
```python
# Example: High income, high loan applicant
new_cluster = predict_cluster(8000, 200, centroids)
```

### Prediction Examples

The model can predict clusters for new applicants:

- **High Income (8000), High Loan (200)** → Cluster 3
- **Low Income (2000), Low Loan (50)** → Cluster 1  
- **Medium Income (5000), Medium Loan (120)** → Cluster 2

## 📊 Visualizations

The project includes several visualizations:

1. **Initial Data Distribution**: Scatter plot of all applicants
2. **Initial Centroids**: Random starting points for clustering
3. **Final Clusters**: Color-coded clusters with final centroids
4. **Prediction Visualization**: New applicants plotted with their predicted clusters

## 🔧 Technical Implementation

### Distance Calculation
```python
euclidean_distance = np.sqrt((centroid_income - point_income)² + (centroid_loan - point_loan)²)
```

### Convergence Criteria
The algorithm stops when the sum of centroid movements becomes zero:
```python
difference = loan_change + income_change
```

### Cluster Assignment
Each point is assigned to the cluster with the minimum distance to its centroid.

## 📁 Files Structure

```
├── K_meansClusteringFromScratch.ipynb    # Main implementation notebook
├── clustering.csv                        # Dataset
└── README.md                            # This file
```

## 🎓 Learning Outcomes

This project demonstrates:
- Understanding of K-means algorithm internals
- Implementation of distance calculations
- Centroid update mechanisms
- Convergence detection
- Data visualization techniques
- Practical application to business problems

## 🔮 Future Enhancements

- Implement K-means++ initialization for better initial centroids
- Add elbow method for optimal K selection
- Include silhouette analysis for cluster validation
- Extend to handle categorical features
- Add anomaly detection capabilities

## 📝 Notes

- The algorithm converges when centroids stop moving significantly
- Results may vary due to random initialization
- Consider running multiple times and selecting best result
- Suitable for spherical clusters with similar sizes

## 🤝 Business Applications

This clustering approach can help financial institutions:
- **Customer Segmentation**: Identify different risk profiles
- **Targeted Marketing**: Customize loan products for each segment
- **Risk Assessment**: Understand applicant patterns
- **Product Development**: Design loans for specific customer groups

---

*This implementation provides a solid foundation for understanding clustering algorithms and their practical applications in financial data analysis.*
