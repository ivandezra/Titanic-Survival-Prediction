import pandas as pd
import pickle
import argparse

def preprocess_test_data(file_path):
    df = pd.read_csv(file_path)
    
    # Fill missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    
    # Convert categorical features to numerical
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Select features
    X = df[['Pclass', 'Age', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
    return X

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(input_file, model_path):
    # Load model
    model = load_model(model_path)
    
    # Preprocess test data
    X_test = preprocess_test_data(input_file)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Load test file for PassengerId
    df_test = pd.read_csv(input_file)
    df_test['Survived_Prediction'] = predictions
    
    # Save predictions to a CSV
    df_test[['PassengerId', 'Survived_Prediction']].to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default= './data/test.csv', help='Path to the input CSV file containing test data')
    parser.add_argument('--model', default='./models/logistic_regression_model.pkl', help='Path to the saved model')
    args = parser.parse_args()
    
    predict(args.input, args.model)
