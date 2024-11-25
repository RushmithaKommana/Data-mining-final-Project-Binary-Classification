#!/usr/bin/env python
# coding: utf-8

# # CS 634 Data Mining Final Term Project
# 
# ## Overview
# 
# This Python notebook aims to implement and compare the performance of three different machine learning algorithms for binary classification on a dataset of your choice. The three algorithms to be implemented are:
# 
# 1. Random Forest Classifier
# 2. Long Short-Term Memory (LSTM) network (a deep learning model)
# 3. Support Vector Machine (SVM)
# 
# ### The notebook will perform the following tasks:
# 
# 1. Load the selected dataset from a reputable source (as mentioned in the "Additional Option: Sources of Data" section).
# 2. Split the data into features (X) and target (y).
# 3. Implement the three classification algorithms using existing libraries (e.g., scikit-learn, TensorFlow/Keras).
# 4. Perform 10-fold cross-validation for each algorithm.
# 5. Calculate the performance metrics for each fold, including Accuracy, Precision, Recall, F1-score, ROC-AUC, True Skill Statistic (TSS), and Heidke Skill Score (HSS).
# 6. Calculate the average performance metrics for each algorithm across all folds.
# 7. Visualize the results using Matplotlib, showing the per-fold and average performance metrics for each algorithm.

# In[4]:


get_ipython().system('pip install tensorflow')


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# In[6]:


# Load the Pima Indians Diabetes Dataset
url = 'diabetes.csv'
data = pd.read_csv(url)


# In[7]:


#displaying first 5 rows
data.head()


# In[8]:


data.shape


# In[9]:


data.isnull().sum()


# In[10]:


data.info()


# In[11]:


#unique values in each column
for column in data.columns:
    print("{} has {} unique values".format(column, len(data[column].unique())))


# In[12]:


#describing dataset 
data.describe()


# In[15]:


# Value counts for the 'Outcome' column
df = data['Outcome'].value_counts()

# Bar plot
plt.figure(figsize=(8, 5))
df.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Distribution of Diabetes Outcome')
plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[17]:


import seaborn as sns  # Import Seaborn


# In[19]:


# Correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')  # Correlation matrix for the DataFrame
plt.title('Correlation Matrix', fontsize=16)
plt.show()


# In[21]:


# Box plot for each column except the last one
for column in data.columns[:-1]:  # Assuming the last column ('Outcome') is categorical
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=data, x=column)  # Use the DataFrame and specify the column
    plt.title(f'Box plot of {column}', fontsize=16)
    plt.show()


# In[22]:


# Violin plot for each column except the last one
for column in data.columns[:-1]:  # Exclude the last column ('Outcome') if it's categorical
    plt.figure(figsize=(8, 4))
    sns.violinplot(data=data, x=column)  # Pass the DataFrame and specify the column
    plt.title(f'Violin plot of {column}', fontsize=16)
    plt.show()


# In[23]:


# Histogram plot for each column except the last one
for feature in data.columns[:-1]:  # Iterate over columns, excluding the last one ('Outcome')
    plt.figure(figsize=(8, 4))
    sns.histplot(data[feature], kde=True)  # Use the correct loop variable
    plt.xlabel(feature, fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.title(f'Histogram plot of {feature}', fontsize=16)
    plt.show()


# In[19]:


# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('diabetes.csv')  # Replace with your dataset path

# Define features (X) and target variable (y)
X = data.drop(columns=['Outcome'])  # Exclude the target column
y = data['Outcome']  # Target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display the shapes of the datasets
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')



# In[27]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
sc = StandardScaler()

# Standardize the training and test data
X_train = sc.fit_transform(X_train)  # Fit to training data and transform
X_test = sc.transform(X_test)        # Transform test data using the same scaler


# ### Implement the 3 algorithms

# In[28]:


# 1. Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[35]:


# 2. LSTM (from the "Additional Option: Deep Learning" list)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential([
    LSTM(64, input_shape=(10, 20)),  # Old approach
    Dense(1, activation='sigmoid')
])

model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[36]:


# 3. Support Vector Machine (SVM) (from the "Additional Option: Algorithms" list)
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)


# In[37]:


# Perform 10-fold cross-validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


# In[38]:


# Initialize lists to store the performance metrics
rf_metrics = []
lstm_metrics = []
svm_metrics = []


# In[39]:


# Initialize lists to store the metrics for each fold
rf_fold_metrics = []
lstm_fold_metrics = []
svm_fold_metrics = []

for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
    print(f'Fold {i}:')

    # Split the data into training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train and evaluate the Random Forest Classifier
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
    tss = (tp / (tp + fn)) - (fp / (fp + tn))
    hss = 2 * ((tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)))
    rf_fold_metrics.append({'Accuracy': accuracy_score(y_test, y_pred_rf),
                           'Precision': precision_score(y_test, y_pred_rf),
                           'Recall': recall_score(y_test, y_pred_rf),
                           'F1-score': f1_score(y_test, y_pred_rf),
                           'ROC-AUC': roc_auc_score(y_test, y_pred_rf),
                           'TSS': tss,
                           'HSS': hss})
    rf_metrics.append(rf_fold_metrics[-1])

    # Train and evaluate the LSTM model
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
    y_pred_lstm = (model.predict(X_test_lstm) > 0.5).astype("int32")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lstm).ravel()
    tss = (tp / (tp + fn)) - (fp / (fp + tn))
    hss = 2 * ((tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)))
    lstm_fold_metrics.append({'Accuracy': accuracy_score(y_test, y_pred_lstm),
                             'Precision': precision_score(y_test, y_pred_lstm),
                             'Recall': recall_score(y_test, y_pred_lstm),
                             'F1-score': f1_score(y_test, y_pred_lstm),
                             'ROC-AUC': roc_auc_score(y_test, y_pred_lstm),
                             'TSS': tss,
                             'HSS': hss})
    lstm_metrics.append(lstm_fold_metrics[-1])

    # Train and evaluate the SVM model
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svm).ravel()
    tss = (tp / (tp + fn)) - (fp / (fp + tn))
    hss = 2 * ((tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)))
    svm_fold_metrics.append({'Accuracy': accuracy_score(y_test, y_pred_svm),
                            'Precision': precision_score(y_test, y_pred_svm),
                            'Recall': recall_score(y_test, y_pred_svm),
                            'F1-score': f1_score(y_test, y_pred_svm),
                            'ROC-AUC': roc_auc_score(y_test, y_pred_svm),
                            'TSS': tss,
                            'HSS': hss})
    svm_metrics.append(svm_fold_metrics[-1])


# In[40]:


# Calculate the average performance metrics
rf_avg_metrics = {k: np.mean([d[k] for d in rf_metrics]) for k in rf_metrics[0]}
lstm_avg_metrics = {k: np.mean([d[k] for d in lstm_metrics]) for k in lstm_metrics[0]}
svm_avg_metrics = {k: np.mean([d[k] for d in svm_metrics]) for k in svm_metrics[0]}


# In[41]:


# Print the results
print('Random Forest Classifier:')
print(rf_avg_metrics)
print('\nLSTM:')
print(lstm_avg_metrics)
print('\nSVM:')
print(svm_avg_metrics)


# In[42]:


# Print the results in tabular format
print('Performance Metrics')
print('=' * 20)
print('{:^20} | {:^20} | {:^20} | {:^20}'.format('Metric', 'Random Forest', 'LSTM', 'SVM'))
print('-' * 80)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'TSS', 'HSS']:
    print('{:^20} | {:^20.3f} | {:^20.3f} | {:^20.3f}'.format(metric, rf_avg_metrics[metric], lstm_avg_metrics[metric], svm_avg_metrics[metric]))


# ### Visualization

# In[43]:


# Random Forest Classifier
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot([m['Accuracy'] for m in rf_fold_metrics], label='Accuracy')
plt.plot([m['Precision'] for m in rf_fold_metrics], label='Precision')
plt.plot([m['Recall'] for m in rf_fold_metrics], label='Recall')
plt.plot([m['F1-score'] for m in rf_fold_metrics], label='F1-score')
plt.title('Random Forest Classifier - Per Fold')
plt.xlabel('Fold')
plt.ylabel('Metric')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'TSS', 'HSS'],
       [rf_avg_metrics['Accuracy'], rf_avg_metrics['Precision'], rf_avg_metrics['Recall'],
        rf_avg_metrics['F1-score'], rf_avg_metrics['ROC-AUC'], rf_avg_metrics['TSS'], rf_avg_metrics['HSS']])
plt.title('Random Forest Classifier - Average')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# In[44]:


# LSTM
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot([m['Accuracy'] for m in lstm_fold_metrics], label='Accuracy')
plt.plot([m['Precision'] for m in lstm_fold_metrics], label='Precision')
plt.plot([m['Recall'] for m in lstm_fold_metrics], label='Recall')
plt.plot([m['F1-score'] for m in lstm_fold_metrics], label='F1-score')
plt.title('LSTM - Per Fold')
plt.xlabel('Fold')
plt.ylabel('Metric')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'TSS', 'HSS'],
       [lstm_avg_metrics['Accuracy'], lstm_avg_metrics['Precision'], lstm_avg_metrics['Recall'],
        lstm_avg_metrics['F1-score'], lstm_avg_metrics['ROC-AUC'], lstm_avg_metrics['TSS'], lstm_avg_metrics['HSS']])
plt.title('LSTM - Average')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# In[31]:


# SVM
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot([m['Accuracy'] for m in svm_fold_metrics], label='Accuracy')
plt.plot([m['Precision'] for m in svm_fold_metrics], label='Precision')
plt.plot([m['Recall'] for m in svm_fold_metrics], label='Recall')
plt.plot([m['F1-score'] for m in svm_fold_metrics], label='F1-score')
plt.title('SVM - Per Fold')
plt.xlabel('Fold')
plt.ylabel('Metric')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'TSS', 'HSS'],
       [svm_avg_metrics['Accuracy'], svm_avg_metrics['Precision'], svm_avg_metrics['Recall'],
        svm_avg_metrics['F1-score'], svm_avg_metrics['ROC-AUC'], svm_avg_metrics['TSS'], svm_avg_metrics['HSS']])
plt.title('SVM - Average')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# ## Conclusion
# 
# This Python notebook successfully implemented and compared the performance of three different machine learning algorithms for binary classification on a selected dataset. The three algorithms evaluated were:
# 
# 1. Random Forest Classifier
# 2. Long Short-Term Memory (LSTM) network
# 3. Support Vector Machine (SVM)
# 
# Through the 10-fold cross-validation process, the notebook calculated and analyzed several performance metrics for each algorithm, including Accuracy, Precision, Recall, F1-score, ROC-AUC, True Skill Statistic (TSS), and Heidke Skill Score (HSS). The results were presented in both tabular and visual formats to facilitate a comprehensive comparison.

# Conclusion:
# Looking at the metrics, Random Forest stands out as the best-performing model overall:
# 
# It excels in most important areas, including Accuracy, Recall, F1-score, ROC-AUC, TSS, and HSS.
# SVM does well in Precision, but struggles in other key areas like Recall and F1-score when compared to Random Forest.
# LSTM, on the other hand, trails behind in all metrics, particularly in Recall, F1-score, and ROC-AUC, making it less effective for this task.

# **Random Forest proves to be the most reliable model for this classification task, to its strong overall performance. It strikes a good balance across all metrics, making it the most effective choice for this dataset.
