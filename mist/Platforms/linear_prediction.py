from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Prepare the input (X) and output (y) data

def train_linear_model(X, y):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    ml_model = LinearRegression()
    ml_model.fit(X_train, y_train)

    # Make predictions
    y_pred = ml_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Display the coefficients
    print(f"Coefficients: {ml_model.coef_}")
    print(f"Intercept: {ml_model.intercept_}")
    return ml_model

def train_neural_network(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    ml_model = MLPRegressor( hidden_layer_sizes=(20, 20), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
    ml_model.fit(X_train, y_train)

    # Make predictions
    y_pred = ml_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Display the coefficients
    print(f"Coefficients: {ml_model.coef_}")
    print(f"Intercept: {ml_model.intercept_}")
    return ml_model 