# 3. Given a bank customer, build a neural network-based classifier that can determine whether
# they will leave or not in the next 6 months. Dataset Description: The case study is from an open-source dataset from Kaggle. The dataset contains 10,000 sample points with 14 distinct features such as CustomerId, CreditScore, Geography, Gender, Age, Tenure, Balance, etc.
# Link to the Kaggle project: https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling
# Perform following steps:
# 1. Read the dataset.
# 2. Distinguish the feature and target set and divide the data set into training and test sets.
# 3. Normalize the train and test data.
# 4. Initialize and build the model. Identify the points of improvement and implement the same.
# 5. Print the accuracy score and confusion matrix .


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load and preprocess the dataset
data = pd.read_csv('dataset/Churn_Modelling.csv')  # Correct file path

# Encoding categorical variables if needed (example)
X = data.drop(['CustomerId', 'Exited'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding categorical features

y = data['Exited']

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Build and compile the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 6: Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Accuracy and Confusion Matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))
