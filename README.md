# Indian Railways Train Schedule Analysis 🚆

A Data Science and Machine Learning project that analyzes Indian Railways train schedules using Python. The project explores train routes, distances, running times, speeds, station activity, and uses classification algorithms to predict train categories.

## 📌 Project Overview

This project uses the December 2017 Indian Railways train schedule dataset to perform data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning.

The main goal is to understand railway scheduling patterns and predict whether a train belongs to the **Special, Mail/Express, or Superfast** category.

## 🎯 Project Objectives

1. Identify the longest and shortest train routes.
2. Find the top 10 busiest railway stations.
3. Identify the most common starting and ending stations.
4. Compare average speeds across train categories.
5. Analyze the relationship between route distance, number of stops, and halt time.
6. Predict train categories using machine learning classification models.

## 📊 Dataset Information

- **Dataset:** Indian Railways Train Schedule
- **Time Period:** December 2017
- **Total Trains:** 577
- **Raw Records:** 10,222 train-stop records
- **Cleaned Records:** 10,220 train-stop records

### Important Features

- Train Number
- Train Name
- Station Name
- Arrival Time
- Departure Time
- Distance
- Source Station
- Destination Station

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## ⚙️ Project Workflow

1. **Data Loading:** Import the railway dataset using Pandas.
2. **Data Cleaning:** Fix broken rows, handle missing values, remove duplicates, and clean text data.
3. **Preprocessing:** Convert time and numerical columns into suitable formats.
4. **Feature Engineering:** Calculate running hours, average halt time, average speed, and stops per 100 km.
5. **Exploratory Data Analysis:** Study route distances, train speeds, station activity, and correlations.
6. **Machine Learning:** Train and evaluate five classification algorithms.
7. **Model Comparison:** Compare model performance using accuracy, precision, recall, and macro F1-score.

## 🤖 Machine Learning Algorithms

The project uses five classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest Classifier
5. Decision Tree Classifier

### Model Performance

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 83.62% | 84.05% |
| Decision Tree | 74.14% | 74.33% |
| SVM (RBF) | 68.10% | 68.24% |
| KNN | 66.38% | 64.23% |
| Logistic Regression | 60.34% | 61.01% |

**Best Performing Model:** Random Forest Classifier

Random Forest achieved the highest accuracy of **83.62%** and macro F1-score of **84.05%**.

## 📈 Key Findings

- Superfast trains have the highest average speed.
- Longer routes generally require more running hours.
- Longer routes tend to have more stops.
- Kalyan Junction is among the busiest stations in the dataset.
- Random Forest performs best among the five classification models.

## 📂 Project Structure

```text
Indian-Railways-Train-Analysis/
│
├── project-code.py
├── train.csv
├── README.md
└── PPT.pptx
```

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Open the Project Folder

```bash
cd Indian-Railways-Train-Analysis
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the Python File

```bash
python project-code.py
```

Make sure `train.csv` is in the same folder as `project-code.py`.

## 📌 Conclusion

This project demonstrates how data science and machine learning can be used to analyze railway schedules and discover useful operational patterns.

By comparing five classification algorithms, Random Forest was identified as the best-performing model for predicting train categories.

## 🔮 Future Scope

- Integration of real-time railway data.
- Train delay analysis.
- Arrival-time prediction.
- Analysis using updated railway datasets.
- Development of an interactive analytics dashboard.

## 👨‍💻 Author

**Hari**  
B.Tech Computer Science and Engineering  
Lovely Professional University

---

⭐ If you find this project interesting, consider giving the repository a star!
