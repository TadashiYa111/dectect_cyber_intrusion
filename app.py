import sklearn
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Preprocess the Dataset
data_frame = pd.read_csv('./SIMARGL2021.csv')

selected_columns = [
'DST_TOS', 'SRC_TOS', 'TCP_WIN_SCALE_OUT', 'TCP_WIN_SCALE_IN', 'TCP_FLAGS','TCP_WIN_MAX_OUT', 'PROTOCOL', 'TCP_WIN_MIN_OUT', 'TCP_WIN_MIN_IN','TCP_WIN_MAX_IN', 'LAST_SWITCHED', 'TCP_WIN_MSS_IN', 'TOTAL_FLOWS_EXP','FIRST_SWITCHED', 'FLOW_DURATION_MILLISECONDS', 'LABEL'
]
data_frame = data_frame[selected_columns]
print(data_frame.head())

# Remove Duplicated dataset
print("Duplicated Values : ", data_frame.duplicated().sum())
data_frame.drop_duplicates(inplace=True)
print(data_frame.head())

grouped_data = data_frame.groupby('LABEL').size()
fig, ax = plt.subplots(1)
ax.bar(grouped_data.index, grouped_data.values)
ax.set(xlabel='LABEL', ylabel='Distinct Count')
plt.xticks(rotation=90)
plt.show()

grouped_data = data_frame['LABEL'].value_counts()
fig, ax = plt.subplots(figsize=(8,8))
ax.pie(grouped_data.values, labels=grouped_data.index, autopct='%1.1f%%', textprops={'fontsize': 5})
ax.set_title('Distribution of LABEL')
plt.show()

normalized_data = data_frame.copy()
numerical_columns = normalized_data.select_dtypes(include=['float64', 'int64']).columns

non_numerical_columns = normalized_data.select_dtypes(exclude=['float64', 'int64']).columns

label_encoder = LabelEncoder()
normalized_data[non_numerical_columns] = normalized_data[non_numerical_columns].apply(label_encoder.fit_transform)

scaler = StandardScaler()
normalized_data[numerical_columns] = scaler.fit_transform(normalized_data[numerical_columns])
print(normalized_data.head())

X = normalized_data.drop(columns=['LABEL'], axis=1)
Y = normalized_data['LABEL']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=30)

start_time = time.time()
rf_classifier.fit(X_train, y_train)
end_time = time.time()
training_timeRFC = end_time - start_time

y_pred = rf_classifier.predict(X_test)

accuracyRFC = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, "weighted")
precision = precision_score(y_test, y_pred, "weighted")
recall = recall_score(y_test, y_pred, "weighted")
print(accuracyRFC)
print(f1)
print(precision)
print(recall)

dt_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=4)

start_time = time.time()
dt_classifier.fit(X_train, y_train)
end_time = time.time()
training_timeDT = end_time - start_time
print(training_timeDT)

y_pred = dt_classifier.predict(X_test)

accuracyDT = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracyDT)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall", recall)

nb_classifier = GaussianNB()

start_time = time.time()
nb_classifier.fit(X_train, y_train)
end_time = time.time()
training_timeNB = end_time - start_time
print(training_timeNB)

y_pred = nb_classifier.predict(X_test)

accuracyNB = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
print("Accuracy:", accuracyNB)
print("F1 Score:", f1_score)
print("Recall Score:", recall)

accuracy_scores =[accuracyRFC, accuracyDT, accuracyNB]
training_times = [training_timeRFC, training_timeDT, training_timeNB]

algorithm_names = ["Random Forest", "Decision Tree", "Gaussian Naive Bayes"]

plt.bar(algorithm_names, accuracy_scores)
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Algorithm')
plt.show()