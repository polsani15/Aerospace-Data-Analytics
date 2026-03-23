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
    data = []
    labels = []
    for _ in range(samples):
        choice = np.random.choice([0, 1, 2, 3]) # Force equal distribution
        if choice == 0: # Taxi
            alt, speed, vs = np.random.randint(0, 50), np.random.randint(0, 30), np.random.randint(-10, 10)
        elif choice == 1: # Takeoff/Climb
            alt, speed, vs = np.random.randint(500, 10000), np.random.randint(150, 250), np.random.randint(1000, 2500)
        elif choice == 2: # Cruise
            alt, speed, vs = np.random.randint(28000, 38000), np.random.randint(400, 480), np.random.randint(-50, 50)
        else: # Landing
            alt, speed, vs = np.random.randint(500, 5000), np.random.randint(130, 180), np.random.randint(-1500, -500)
        
        data.append([alt, speed, vs])
        labels.append(choice)
    
    return pd.DataFrame(data, columns=['Alt', 'Speed', 'VS']), np.array(labels)

# 2. PREPARE DATA
X, y = generate_flight_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. BUILD DEEP LEARNING MODEL (Multi-Layer Perceptron)
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, activation='relu')
print("[PROCESS] Training Neural Network for Flight Phase Classification...")
mlp.fit(X_train_scaled, y_train)

# 4. EVALUATE (Fixed for missing classes)
predictions = mlp.predict(X_test_scaled)
present_classes = np.unique(np.concatenate((y_test, predictions)))
all_labels = ['Taxi', 'Takeoff', 'Cruise', 'Landing']
target_names = [all_labels[i] for i in present_classes]

print("\n--- Boeing Analytics Model Report ---")
print(classification_report(y_test, predictions, target_names=target_names))
