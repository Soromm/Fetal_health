# Import lib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

fetal_data = pd.read_csv("fetal_health.csv")
fetal_data.head()

## DESCRIPTIVE STATISTICS

fetal_data.info()

fetal_data['fetal_health'] = fetal_data['fetal_health'].astype(int)

## STATISTICS

fetal_data.describe()

from scipy.stats import shapiro

stat, p = shapiro(fetal_data)
print(f'Statistic: {stat}, p-value: {p}')

if p > 0.05:
    print("Data follows a normal distribution (Fail to reject H0).")
else:
    print("Data does not follow a normal distribution (Reject H0).")


from scipy.stats import normaltest

stat, p = normaltest(fetal_data)
print(f'Statistic: {stat}, p-value: {p}')

if p.all() > 0.05:
    print("Data follows a normal distribution (Fail to reject H0).")
else:
    print("Data does not follow a normal distribution (Reject H0).")


## DATA VISUALIZATION

## Fetal health: 1 - Normal 2 - Suspect 3 - Pathological

ax = sns.countplot(x='fetal_health', data = fetal_data, palette=["#FFBF00", "#DFFF00", "#FF6600"] )
total = len(fetal_data["fetal_health"])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width() / 3 - 0.05
    y = p.get_y() + p.get_height() + 5
    ax.annotate(percentage, (x, y))

plt.show()


featureplot = fetal_data.select_dtypes(include=['Float64'])
featureplot.head()

rows = (len(featureplot.columns)+1) // 2
fig, axes = plt.subplots(rows, 2, figsize=(8, rows * 4))

for i, col in enumerate(featureplot.columns):
    hor = i % 2
    var = i // 2

    sns.histplot(x=featureplot[col], kde=True, hue=fetal_data['fetal_health'], palette='bright', ax=axes[var,hor])
    axes[var, hor].set_title(f'Distribution of {col}', size=5)

    plt.legend(fetal_data['fetal_health'].value_counts().index-1,title_fontsize='20', fontsize='20')

plt.tight_layout()


plt.figure(figsize=(20, 15))
sns.heatmap(featureplot.corr(),cmap='Greens', annot=True, fmt='.2f', annot_kws={'size':10})
plt.show()


## features to drop
#from the heatmap we dropped column with high correlations.

featureplot = featureplot.drop(columns=['light_decelerations','histogram_mode','histogram_width','histogram_mean','histogram_median','histogram_number_of_peaks','histogram_variance'])
fetal_data.head()

plt.figure(figsize=(20, 15))
sns.heatmap(featureplot.corr(),cmap='Greens', annot=True, fmt='.2f', annot_kws={'size':10})
plt.show()


## Feature Engineering

from sklearn.preprocessing import PowerTransformer
feature_train = featureplot.copy()

transformer = PowerTransformer(method= 'yeo-johnson')

feature_train = transformer.fit_transform(feature_train)

feature_train = pd.DataFrame(feature_train, columns = featureplot.columns)
feature_train

rows = (len(feature_train.columns)+1) // 2
fig, axes = plt.subplots(rows, 2, figsize=(8, rows * 4))

for i, col in enumerate(feature_train.columns):
    hor = i % 2
    var = i // 2

    sns.histplot(x=feature_train[col], kde=True, hue=fetal_data['fetal_health'], palette='bright',  ax=axes[var,hor])
    axes[var, hor].set_title(f'Distribution of {col}', size=5)

    plt.legend(fetal_data['fetal_health'].value_counts().index-1,title_fontsize='20', fontsize='20')

plt.tight_layout()

## Modeling 

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, matthews_corrcoef, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle

x = featureplot
y = fetal_data['fetal_health']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.25, random_state=42)
print(X_test.shape, X_train.shape, Y_test.shape, Y_train.shape)

X_test.head()


def multi_class_roc(model, X_test, Y_test, num_classes = 3):
    Y_test_bin = label_binarize(Y_test, classes=[1, 2, 3])
    y_pred_proba = model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], y_pred_proba[:,i])

        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))  
    colors = cycle(['red', 'blue', 'green'])

    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i+1} (AUC = {roc_auc[i]:.4f})')

    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

#confusion matrix function

def confusionmatrix(Y_test, y_pred, model_name):
    cm = confusion_matrix(Y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


## RandomForestclassifer

model_rf = RandomForestClassifier(random_state=42)

model_rf.fit(X_train, Y_train)
predict_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(Y_test, predict_rf)

print("Random Forest Classification Report")
print(classification_report(Y_test, predict_rf))

mcc_rf = matthews_corrcoef(Y_test, predict_rf)

print(f"Matthews Correlation Coefficient (MCC): {mcc_rf:.4f}")

confusionmatrix(Y_test, predict_rf, 'Random Forest')
multi_class_roc(model_rf, X_test, Y_test, num_classes=3)


## Support Vector Machine

svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_train, Y_train)

svm_y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(Y_test,svm_y_pred)

print(svm_accuracy)
print('SVM Classification Report')
print(classification_report(Y_test,svm_y_pred))

mcc_svm = matthews_corrcoef(Y_test,svm_y_pred)


print(f"Matthews Correlation Coefficient (MCC): {mcc_svm:.4f}")

confusionmatrix(Y_test, svm_y_pred, 'SVM')
multi_class_roc(svm_model, X_test, Y_test, num_classes=3)


## XGBOOST

xg_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
Y_test_xg = Y_test.copy()
Y_train_xg = Y_train.copy()
Y_train_xg = Y_train_xg - 1
Y_test_xg = Y_test_xg - 1
xg_model.fit(X_train, Y_train_xg)
xg_y_predi = xg_model.predict(X_test)
xg_accuracy = accuracy_score(Y_test_xg, xg_y_predi)

print("XGBoost Classification Report")
print(classification_report(Y_test_xg, xg_y_predi, zero_division=0))

mcc_xgb = matthews_corrcoef(Y_test_xg, xg_y_predi)

print(f"Matthews Correlation Coefficient (MCC): {mcc_xgb:.4f}")

confusionmatrix(Y_test_xg, xg_y_predi, 'XGBOOST')
multi_class_roc(xg_model, X_test, Y_test, num_classes=3)


## voting Classifier

voting_model = VotingClassifier(
    estimators=[('xgb', xg_model), ('rf', model_rf)],
    voting='soft'  # soft voting averages the predicted probabilities
)
voting_model.fit(X_train, Y_train)

voting_y_pred = voting_model.predict(X_test)

voting_accuracy = accuracy_score(Y_test, voting_y_pred)

print("Blending (Voting) Classification Report")
print(classification_report(Y_test, voting_y_pred))

print(f"Blending Accuracy: {voting_accuracy:.4f}")

mcc_voting = matthews_corrcoef(Y_test,voting_y_pred)

print(f"Matthews Correlation Coefficient (MCC): {mcc_voting:.4f}")


confusionmatrix(Y_test, voting_y_pred, 'Voting classifier')
multi_class_roc(voting_model, X_test, Y_test, num_classes=3)


## Model comparison 

import plotly.graph_objs as go
import plotly.subplots as sp

models = ['SVM', 'Random Forest', 'XGBoost', 'Blending']
accuracies = [svm_accuracy, accuracy_rf, xg_accuracy, voting_accuracy]

bar_fig = go.Figure()

bar_fig.add_trace(go.Bar(
    x=models,
    y= accuracies,
    marker_color= '#66CA25'
))

bar_fig.update_layout(
    title='Model Comparison - Accuracy',
    xaxis_title='Models',
    yaxis_title='Accuracy',
    yaxis=dict(range=[0.8, 1.0]),  # Set y-axis limits
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    width=800,  # Increase width
    height=800   # Increase height

)

bar_fig.show()



models = ['SVM', 'Random Forest', 'XGBoost', 'Blending']
mcc = [mcc_rf, mcc_svm, mcc_xgb, mcc_voting]

mcc_fig = go.Figure()

mcc_fig.add_trace(go.Bar(
    x=models,
    y=mcc,
    marker_color= '#00712D'
))


mcc_fig.update_layout(
    title='Model Comparison - MCC',
    xaxis_title='Models',
    yaxis_title='Matthews Correlation Coefficient (MCC)',
    yaxis=dict(range=[0.0, 1.0]),
    xaxis_tickangle=-45,  
    width=800, 
    height=700 
)

# Show the figure
mcc_fig.show()
