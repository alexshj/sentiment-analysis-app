# 🧠 Sentiment Analysis App (Jupyter-based)

This project is a complete sentiment analysis pipeline built using Python, NLTK, Scikit-learn, and Jupyter Notebooks. It performs preprocessing, model training, evaluation, and live predictions on movie reviews using the IMDB dataset.

The application is modular, interpretable, and built according to internship project requirements.

---

## 📁 Project Structure

sentiment-analysis-project/
├── data/
│ ├── raw/ # Original IMDB dataset
│ └── processed/ # Cleaned & preprocessed data
├── models/ # Saved ML model and TF-IDF vectorizer
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ ├── 04_evaluation_visualization.ipynb
│ └── 05_interactive_demo.ipynb
├── sentiment_env/ # Virtual environment (optional)
├── requirements.txt
└── README.md

---

## ✨ Features

- 📊 Data exploration with visualizations
- 🧹 Text preprocessing: HTML removal, stopwords, tokenization, lemmatization
- 🔍 Model comparison: Logistic Regression, Naive Bayes, Random Forest, SVM
- 🧠 TF-IDF feature extraction
- 📉 Confusion matrix, classification report, and top keywords
- 🎤 Interactive review sentiment prediction using Jupyter widgets

---

## 📦 Dataset

- **Source**: [Kaggle - IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Samples**: 50,000 (25k positive, 25k negative)

---

## 🚀 How to Run

1. Clone or download the folder.
2. Create and activate a virtual environment:
   ```bash
   python -m venv sentiment_env
   sentiment_env\Scripts\activate   # (Windows)
Install dependencies:

pip install -r requirements.txt
Launch Jupyter Notebook:


jupyter notebook
Run the notebooks in order:

01_data_exploration.ipynb

02_preprocessing.ipynb

03_model_training.ipynb

04_evaluation_visualization.ipynb

05_interactive_demo.ipynb

🧪 Requirements
Python 3.8+

Jupyter Notebook

pandas, numpy, scikit-learn, nltk

matplotlib, seaborn, wordcloud

ipywidgets, joblib

Install all with:


pip install -r requirements.txt
👨‍💻 Author
Alex Shaju
B.Tech in Artificial Intelligence & Data Science
This project was developed as part of an internship for a sentiment analysis application using the IMDB dataset.

📌 License
This project is for educational and academic use only.
