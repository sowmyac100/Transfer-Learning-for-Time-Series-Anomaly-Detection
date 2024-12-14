import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_spacecraft_telemetry_data():
    np.random.seed(42)
    
    # Generate time index
    time_idx = np.arange(0, 1000)
    
    # Simulate normal telemetry data
    base_signal = np.sin(time_idx / 50) * 10 + 50  # Base signal with a sinusoidal pattern
    drift = time_idx / 200  # Linear drift
    noise = np.random.normal(scale=0.5, size=len(time_idx))  # Small random noise
    
    telemetry_signal = base_signal + drift + noise
    
    # Introduce anomalies
    anomalies = np.zeros_like(telemetry_signal)
    
    # Sudden spikes
    telemetry_signal[200:210] += np.random.normal(10, 2, size=10)
    anomalies[200:210] = 1
    
    # Sudden drops
    telemetry_signal[500:510] -= np.random.normal(10, 2, size=10)
    anomalies[500:510] = 1
    
    # Erratic behavior
    telemetry_signal[700:720] = np.random.normal(50, 5, size=20)
    anomalies[700:720] = 1
    
    # Create a DataFrame
    telemetry_data = pd.DataFrame({
        "time_idx": time_idx,
        "telemetry": telemetry_signal,
        "anomaly": anomalies
    })
    
    return telemetry_data

# Generate the dataset
telemetry_data = create_spacecraft_telemetry_data()

# Plot the dataset
plt.figure(figsize=(15, 6))
plt.plot(telemetry_data["time_idx"], telemetry_data["telemetry"], label="Telemetry Signal")
plt.scatter(
    telemetry_data["time_idx"][telemetry_data["anomaly"] == 1],
    telemetry_data["telemetry"][telemetry_data["anomaly"] == 1],
    color="red",
    label="Anomalies",
    zorder=5
)
plt.title("Synthetic Spacecraft Telemetry Data with Anomalies")
plt.xlabel("Time Index")
plt.ylabel("Telemetry Signal")
plt.legend()
plt.show()

# Display the first few rows of the dataset
telemetry_data.head()
