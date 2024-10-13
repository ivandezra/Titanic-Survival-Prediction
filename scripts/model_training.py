from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from data_processing import preprocess_data

# Load preprocessed data
X, y = preprocess_data('./data/train.csv')

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open('./models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Validate the model
y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%")
