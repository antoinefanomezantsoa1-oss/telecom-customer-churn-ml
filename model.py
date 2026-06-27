import csv
import math

# 1. THE SIGMOID FUNCTION
def sigmoid(z):
    # Clips z to prevent math overflow errors
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

# 2. LOAD DATA FROM YOUR CSV
X = [] # Features: [Bias, Tenure, MonthlyCharges]
Y = [] # Target: Churn (0 or 1)

with open('d:/dev antoine/MLproject/churn_data.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader) # Skip header row
    for row in reader:
        # Features normalized manually so the math stays stable
        tenure_normalized = float(row[5]) / 72.0   # Index 5 is tenure (divided by max months approx 72)
        charges_normalized = float(row[18]) / 100.0 # Index 18 is MonthlyCharges

        # Convert 'Yes' to 1.0 and 'No' to 0.0 using the correct Index 20 (Churn)
        label = 1.0 if row[20] == 'Yes' else 0.0
        
        X.append([1.0, tenure_normalized, charges_normalized]) # 1.0 is the bias
        Y.append(label) # Append the numeric label we calculated above

# 3. INITIALIZE WEIGHTS (Start at 0)
weights = [0.0, 0.0, 0.0]
learning_rate = 0.3
epochs = 500

# 4. OFFLINE GRADIENT DESCENT LOOP
print("Starting offline training loop...")
for epoch in range(epochs):
    gradients = [0.0, 0.0, 0.0]
    
    # Calculate gradients for every data row
    for i in range(len(X)):
        # Dot product: z = w0*x0 + w1*x1 + w2*x2
        z = sum(X[i][j] * weights[j] for j in range(len(weights)))
        prediction = sigmoid(z)
        error = prediction - Y[i]
        
        for j in range(len(weights)):
            gradients[j] += error * X[i][j]
            
    # Update weights
    for j in range(len(weights)):
        weights[j] -= (learning_rate / len(X)) * gradients[j]

print("Training finished!")
print(f"Final Weights: Bias={weights[0]:.4f}, Tenure={weights[1]:.4f}, Charges={weights[2]:.4f}")