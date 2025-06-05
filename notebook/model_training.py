import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Models
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv('notebook/data/stud_with_scores.csv')
print(df.head())

# Define features and target variable
X = df.drop(columns=['math score', 'Total Score', 'Average Score'])
y = df['math score']
print(X.head())

# Creating column Transformers with 3 transformers
numeric_features = X.select_dtypes(exclude=['object']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
onehot_transformer = OneHotEncoder()

# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('OneHotEncoder', onehot_transformer, categorical_features),
        ('StandardScaler', numeric_transformer, numeric_features)
    ]
)

X = preprocessor.fit_transform(X)
print(X.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define function to give model metrics
def evaluate_model(true, predicted):
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    mae = mean_absolute_error(true, predicted)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    return mse, rmse, r2, mae

# Model List
models = {
    'KNeighborsRegressor': KNeighborsRegressor(),
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'SVR': SVR(),
    'CatBoostRegressor': CatBoostRegressor(verbose=False),
    'XGBRegressor': XGBRegressor()
}
model_list = []
r2_scores = []

# Train and evaluate each model
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    # Predict on the test set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate the model
    model_train_mse, model_train_rmse, model_train_r2, model_train_mae = evaluate_model(y_train, y_train_pred)
    model_test_mse, model_test_rmse, model_test_r2, model_test_mae = evaluate_model(y_test, y_test_pred)

    print(f"\nModel: {list(models.keys())[i]}")
    model_list.append(list(models.keys())[i])

    print('Model Performance on Training Set:')
    print('- Root Mean Squared Error: {:.4f}'.format(model_train_rmse))
    print('- Mean Absolute Error: {:.4f}'.format(model_train_mae))
    print('- R^2 Score: {:.4f}'.format(model_train_r2))
    print(' '*50)
    print('Model Performance on Test Set:')
    print('- Root Mean Squared Error: {:.4f}'.format(model_test_rmse))
    print('- Mean Absolute Error: {:.4f}'.format(model_test_mae))
    print('- R^2 Score: {:.4f}'.format(model_test_r2))
    r2_scores.append(model_test_r2)
    print('='*100)
    print('\n')
# Create a DataFrame to store model names and their R^2 scores
model_performance = pd.DataFrame({
    'Model': model_list,
    'R^2 Score': r2_scores
})
# print(model_performance)
# Sort the DataFrame by R^2 Score in descending order
model_performance = model_performance.sort_values(by='R^2 Score', ascending=False)
# Print the sorted DataFrame
print(model_performance)


# Difference between Actual and Predicted Values
pred_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Difference': y_test - y_test_pred
})