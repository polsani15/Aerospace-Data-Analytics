import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# 1. GENERATE SYNTHETIC FLIGHT DATA (Boeing-style FDM data)
# Features: [Altitude(ft), Airspeed(knots), Vertical_Speed(fpm)]
# Labels: 0: Taxi, 1: Takeoff, 2: Cruise, 3: Landing
def generate_flight_data(samples=1000):
    np.random.seed(42)
    # Altitude, Speed, Vertical Speed
    data = np.random.rand(samples, 3) 
    labels = []
    for i in range(samples):
        alt = np.random.randint(0, 35000)
        speed = np.random.randint(0, 500)
        vs = np.random.randint(-2000, 2000)
        
        if alt < 500 and speed < 40: label = 0 # Taxi
        elif vs > 500 and alt < 10000: label = 1 # Takeoff/Climb
        elif alt > 25000: label = 2 # Cruise
        else: label = 3 # Landing/Descent
        labels.append(label)
    
    return pd.DataFrame(data, columns=['Alt', 'Speed', 'VS']), np.array(labels)

# 2. PREPARE DATA
X, y = generate_flight_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. BUILD DEEP LEARNING MODEL (Multi-Layer Perceptron)
# This hits the "Neural Network" requirement in the JD
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, activation='relu')
print("[PROCESS] Training Neural Network for Flight Phase Classification...")
mlp.fit(X_train_scaled, y_train)

# 4. EVALUATE
predictions = mlp.predict(X_test_scaled)
print("\n--- Boeing Analytics Model Report ---")
print(classification_report(y_test, predictions, target_names=['Taxi', 'Takeoff', 'Cruise', 'Landing']))