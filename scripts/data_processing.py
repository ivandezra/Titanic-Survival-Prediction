import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Fill missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Drop 'Cabin' column due to too many missing values
    df.drop('Cabin', axis=1, inplace=True)
    
    # Convert categorical features to numerical
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Select features
    X = df[['Pclass', 'Age', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
    y = df['Survived']
    
    return X, y

if __name__ == "__main__":
    preprocess_data('./data/train.csv')