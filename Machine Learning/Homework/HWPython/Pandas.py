import pandas as pd
from sqlalchemy import create_engine
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

def create_or_define_dataframes():
    # Replace with your database credentials and details
    username = 'root'
    password = ''
    host = 'localhost'
    database = 'employees'

    # Create the connection string
    connection_string = f'mysql+pymysql://{username}:{password}@{host}/{database}'

    # Create the connection engine
    engine = create_engine(connection_string)

    # Define the query
    query = 'SELECT * FROM salaries WHERE to_date != "9999-01-01" ORDER BY from_date DESC'

    # Execute the query and load the data into a pandas DataFrame
    df = pd.read_sql(query, engine)

    # Return the DataFrame
    return df

def perform_linear_regression(df):
    # Extract year from from_date
    df['year'] = pd.to_datetime(df['from_date']).dt.year

    # Drop rows with missing values
    df = df.dropna(subset=['year', 'salary'])

    # Prepare the data for regression
    X = jnp.array(df['year'].values)
    y = jnp.array(df['salary'].values)

    # Normalize the data
    X_mean = jnp.mean(X)
    X_std = jnp.std(X)
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std

    # Initialize parameters
    theta = jnp.array([0.0, 0.0])

    # Define the model
    def model(theta, x):
        return theta[0] + theta[1] * x

    # Define the loss function
    def loss_fn(theta, x, y):
        predictions = model(theta, x)
        return jnp.mean((predictions - y) ** 2)

    # Compute the gradient of the loss function
    grad_loss_fn = jit(grad(loss_fn))

    # Training loop
    learning_rate = 0.001
    num_iterations = 5000
    for _ in range(num_iterations):
        gradients = grad_loss_fn(theta, X, y)
        theta = theta - learning_rate * gradients

    # Predict the salary for the year 2025
    year_2025 = (jnp.array([2025]) - X_mean) / X_std
    predicted_salary = model(theta, year_2025)

    # Denormalize the prediction
    predicted_salary = predicted_salary * y_std + y_mean

    return predicted_salary[0]

# Example usage
df_salaries = create_or_define_dataframes()
predicted_salary_2025 = perform_linear_regression(df_salaries)
print(f"Predicted salary for the year 2025: {predicted_salary_2025}")