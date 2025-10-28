# 🧠 Alzheimer's Disease Detection System

A Machine Learning–based system for **early detection of Alzheimer's Disease (AD)** using MRI-related attributes and patient health data.  
This project demonstrates the use of **data preprocessing, EDA, classification, clustering, and association rule mining** to uncover disease patterns and predict cognitive decline.

---

## 🚀 Project Overview

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder that affects memory, thinking, and behavior.  
Early detection plays a crucial role in improving patient outcomes and treatment efficiency.

This project builds a complete data pipeline — from raw data preprocessing to advanced analytics and model training — using both **Python** and **RapidMiner**.

---

## 🧩 Features Implemented

| Sr. No. | Task | Tool / Language |
|----------|------|----------------|
| 1 | Design a Star & Snowflake Schema | DB Diagram / Draw.io |
| 2 | Data Preprocessing | Python (`pandas`, `sklearn`) |
| 3 | Exploratory Data Analysis (EDA) | Python (`matplotlib`, `seaborn`) |
| 4 | Classification (Decision Tree, Naïve Bayes) | RapidMiner |
| 5 | Classification (Decision Tree, Naïve Bayes) | Python |
| 6 | Clustering (K-Means, Agglomerative, DBSCAN) | RapidMiner |
| 7 | Clustering (K-Means, Agglomerative, DBSCAN) | Python |
| 8 | Association Rule Mining (Apriori Algorithm) | RapidMiner |
| 9 | Association Rule Mining (Apriori Algorithm) | Python |

---

## 🧠 Machine Learning Components

### 🩺 1. **Data Preprocessing**
- Missing value imputation  
- Label encoding for categorical features  
- Feature scaling using `StandardScaler`  
- Artifact saving (`scaler.pkl`, `encoder.pkl`)

### 📊 2. **Exploratory Data Analysis (EDA)**
- Distribution and correlation plots  
- Relationship between risk factors and Alzheimer’s diagnosis  
- Pairplots, heatmaps, and boxplots

### 🌳 3. **Classification Models**
- **Decision Tree Classifier**
- **Naïve Bayes Classifier**
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### 🧩 4. **Clustering Algorithms**
- **K-Means**, **Agglomerative**, and **DBSCAN**
- Cluster visualization using patient attributes (Age, BMI, MMSE, etc.)
- Performance measured with **Silhouette Score**

### 🔗 5. **Association Rule Mining**
- Implemented using **Apriori Algorithm**
- Discovers hidden patterns like:
  > *“If Age > 80 and Sleep Quality is Poor → High risk of Alzheimer’s”*
- Metrics: Support, Confidence, Lift

---

## 🧰 Technologies Used

- **Languages:** Python, SQL  
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib, mlxtend  
- **Tools:** RapidMiner, Git, GitHub, dbdiagram.io  
- **IDE:** Visual Studio Code, Jupyter Notebook  
- **Version Control:** GitHub  

---

## 📂 Project Structure
<img width="379" height="541" alt="image" src="https://github.com/user-attachments/assets/5ab56e36-5534-48d3-802d-b71ead560c84" />

