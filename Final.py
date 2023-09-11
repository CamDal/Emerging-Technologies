import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np
import warnings

class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == 'Linear Regression':
            return LinearRegression()
        elif model_name == 'Support Vector Machine':
            return SVR()
        elif model_name == 'Random Forest':
            return RandomForestRegressor()
        elif model_name == 'Gradient Boosting Regressor':
            return GradientBoostingRegressor()
        elif model_name == 'XGBRegressor':
            return XGBRegressor()
        elif model_name == 'Lasso':
            return Lasso()
        elif model_name == 'Ridge':
            return Ridge()
        else:
            raise ValueError(f"Model '{model_name}' not recognized!")

def load_data(file_name):
    return pd.read_excel(file_name)

def create_pairplot(data):
    sns.pairplot(data)
    plt.show()

def preprocess_data(data):
    if data.isnull().any().any():
        raise ValueError("The data contains missing values. Please ensure the data is cleaned before processing.")

    X = data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Gender', 'Age', 'Income', 'Credit Card Debt', 'Healthcare Cost', 'REITs', 'Net Worth'], axis=1)
    Y = data['Net Worth']
    
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    
    return X_scaled, y_scaled, sc, sc1

def split_data(X_scaled, y_scaled):
    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    model_names = [
        'Linear Regression',
        'Support Vector Machine',
        'Random Forest',
        'Gradient Boosting Regressor',
        'XGBRegressor',
        'Ridge',
        'Lasso'
    ]
    
    models = {}
    for name in model_names:
        print(f"Training model: {name}")
        model = ModelFactory.get_model(name)
        model.fit(X_train, y_train.ravel())
        models[name] = model
        print(f"{name} trained successfully.")
        
    return models

def evaluate_models(models, X_test, y_test):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
        
    return rmse_values

def plot_model_performance(rmse_values):
    plt.figure(figsize=(10,7))
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    bars = plt.bar(models, rmse, color=['blue', 'green', 'red', 'purple', 'orange'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_best_model(models, rmse_values):
    best_model_name = min(rmse_values, key=rmse_values.get)
    best_model = models[best_model_name]
    dump(best_model, "Net_Worth.joblib")

def load_best_model():
    return load("Net_Worth.joblib")

def retrain_model(new_data, X_train, y_train):
    # Add the new data to the existing training data
    X_train = np.vstack((X_train, new_data[:, :-1]))
    y_train = np.vstack((y_train, new_data[:, -1].reshape(-1, 1)))

    # Retrain the models with the updated training data
    models = train_models(X_train, y_train)

    return models

def gather_user_inputs():
    Inherited = int(input("Enter inherited amount: "))
    Stocks = int(input("Enter stock value: "))
    Bonds = float(input("Enter bonds amount: "))
    Mutual_Funds = float(input("Enter mutual funds: "))
    ETFs = float(input("Enter ETFs value: "))
    
    return Inherited, Stocks, Bonds, Mutual_Funds, ETFs

def scale_user_inputs(user_inputs, sc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        user_input_scaled = sc.transform([user_inputs])
    
    return user_input_scaled

def make_prediction(user_input_scaled):
    predicted_amount = loaded_model.predict(user_input_scaled)
    print("Predicted Net Worth:", predicted_amount[0])

if __name__ == "__main__":
    try:
        # Load data
        data = load_data(r"C:\Users\cam10\Downloads\Net_Worth_Data.xlsx")

        # Graph relationships
        create_pairplot(data)

        #Preprocess data
        X_scaled, y_scaled, sc, sc1 = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
        
        # Train models and evaluate
        models = train_models(X_train, y_train)
        rmse_values = evaluate_models(models, X_test, y_test)
        plot_model_performance(rmse_values)
        save_best_model(models, rmse_values)
        
        # Load the best model
        loaded_model = load_best_model()
        
        #Retrain models
        retrain_models = retrain_model

        # Gather user inputs
        user_inputs = gather_user_inputs()
        
        # Scale user inputs
        scaled_inputs = scale_user_inputs(user_inputs, sc)

        # Make prediction
        make_prediction(scaled_inputs)
        
    except ValueError as ve:
        print(f"Error: {ve}")
