import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = 'Nairobi Office Price Ex.csv'
data = pd.read_csv(file_path)

# Extract relevant columns: SIZE as feature and PRICE as target
x = data['SIZE'].values
y = data['PRICE'].values

# Define Mean Squared Error (MSE) function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define Gradient Descent function to update slope (m) and intercept (c)
def gradient_descent(x, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * x + c
    # Calculate gradients
    dm = (-2 / n) * np.sum(x * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)
    # Update parameters
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Initialize parameters randomly
m = np.random.randn()
c = np.random.randn()
learning_rate = 0.0001  # Reduced learning rate
epochs = 10

# Train the model and show progressive predictions
plt.scatter(x, y, color="blue", label="Data Points")

for epoch in range(epochs):
    # Predict the current y values with current m and c
    y_pred = m * x + c
    # Calculate and print Mean Squared Error
    error = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch+1}, MSE: {error}")
    
    # Plot the line of best fit for the current epoch to show progress
    plt.plot(x, y_pred, label=f"Epoch {epoch+1}")
    
    # Update m and c using gradient descent
    m, c = gradient_descent(x, y, m, c, learning_rate)

# Add labels and legend
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.legend()
plt.show()

# Final prediction for office size = 100 sq. ft.
office_size = 100
predicted_price = m * office_size + c
print(f"Predicted office price for {office_size} sq. ft.: {predicted_price}")
