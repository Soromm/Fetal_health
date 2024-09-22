## Fetal Health Classification Project

This project focuses on predicting fetal health using various machine learning models. The dataset used contains several fetal health indicators, and the models are evaluated based on accuracy and Matthews Correlation Coefficient (MCC).

Table of Contents
•	Introduction
•	Dataset
•	Modeling Approach
•	Evaluation
•	Requirements
•	Results
•	Usage

Introduction
This project aims to analyze fetal health data and classify it into different health categories using machine learning algorithms. The primary goal is to predict the health of a fetus by using models like SVM, Random Forest, XGBoost, and Blending techniques.

Dataset
The dataset contains several features that reflect fetal health, including:
•	Baseline value
•	Accelerations
•	Fetal movements
•	Uterine contractions
•	Light decelerations
•	Severe decelerations
•	Abnormal short-term variability
•	Mean value of short-term variability
•	Long-term variability
•	Histogram width, etc.

The target variable is the fetal health status, which can be categorized as:
1.	Normal
2.	Suspect
3.	Pathological

The dataset is saved as fetal_health.csv.

Modeling Approach
The following models were built and evaluated:
•	Support Vector Machine (SVM)
•	Random Forest
•	XGBoost
•	Blending
Each model was fine-tuned and evaluated based on both Accuracy and Matthews Correlation Coefficient (MCC).

Evaluation
To evaluate the models, we used:
1.	Accuracy: The proportion of correctly classified instances.
2.	Matthews Correlation Coefficient (MCC): A measure that takes into account true and false positives and negatives, providing a more balanced evaluation.
Here are two visual representations of the results:
•	MCC Comparison:
•	Accuracy Comparison:

Requirements
To run this project, you need the following dependencies:
pip install numpy pandas matplotlib seaborn plotly scikit-learn xgboost

Results
The best-performing model based on MCC and Accuracy was Blending, which achieved the highest metrics.
•	SVM and Random Forest models performed well, but they lagged behind XGBoost and Blending techniques.
•	XGBoost provided competitive results in both Accuracy and MCC.

Usage
1.	Clone the repository and navigate to the project folder.
2.	Install the required packages.
3.	Run the Jupyter notebook Main.ipynb to reproduce the analysis and models.


License
This project is licensed under the MIT License.

