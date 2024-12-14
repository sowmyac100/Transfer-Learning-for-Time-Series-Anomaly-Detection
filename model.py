import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support

#Split the data into training and test sets
training_cutoff = int(len(spacecraft_data) * 0.8)
train_data = spacecraft_data[:training_cutoff]
test_data = spacecraft_data[training_cutoff:]

#Define the TimeSeriesDataSet for PyTorch Forecasting
max_encoder_length = 30
max_prediction_length = 10

training = TimeSeriesDataSet(
    train_data,
    time_idx="time_idx",
    target="temperature",  # Using temperature as the primary target
    group_ids=["time_idx"],  # Single group for this synthetic data
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["temperature", "pressure", "vibration"],
    target_normalizer=GroupNormalizer(groups=["time_idx"]),  # Normalize target values
)

batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)

#Fine-tune a pretrained transformer model
model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1,  # Regression task
    loss=torch.nn.MSELoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

trainer = Trainer(
    max_epochs=10,
    gpus=1 if torch.cuda.is_available() else 0,
    gradient_clip_val=0.1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
)

#Evaluate the model on the test set
test_dataloader = training.to_dataloader(train=False, batch_size=batch_size)
predictions = model.predict(test_dataloader)

# Add predictions to the test dataset
test_data["predictions"] = predictions
test_data["anomaly_score"] = np.abs(test_data["predictions"] - test_data["temperature"])

# Set a threshold for anomalies
threshold = test_data["anomaly_score"].quantile(0.95)
test_data["predicted_anomaly"] = (test_data["anomaly_score"] > threshold).astype(int)

# Evaluate performance
precision, recall, f1, _ = precision_recall_fscore_support(
    test_data["anomaly"], test_data["predicted_anomaly"], average="binary"
)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Visualization
def plot_results(test_data, threshold):
    """Plot the actual values, anomalies, and anomaly predictions."""
    plt.figure(figsize=(15, 6))
    
    # Plot actual time series
    plt.plot(test_data["time_idx"], test_data["temperature"], label="Temperature", alpha=0.8)
    
    # Plot actual anomalies
    plt.scatter(
        test_data["time_idx"][test_data["anomaly"] == 1],
        test_data["temperature"][test_data["anomaly"] == 1],
        color="red",
        label="Actual Anomalies",
        zorder=5,
    )
    
    # Plot predicted anomalies
    plt.scatter(
        test_data["time_idx"][test_data["predicted_anomaly"] == 1],
        test_data["temperature"][test_data["predicted_anomaly"] == 1],
        color="orange",
        label="Predicted Anomalies",
        marker="x",
        zorder=5,
    )
    
    # Plot anomaly scores as a secondary axis
    ax2 = plt.gca().twinx()
    ax2.plot(
        test_data["time_idx"], 
        test_data["anomaly_score"], 
        color="green", 
        label="Anomaly Score", 
        alpha=0.6
    )
    ax2.axhline(y=threshold, color="purple", linestyle="--", label="Threshold", alpha=0.8)
    ax2.set_ylabel("Anomaly Score")
    
    # Add labels, legend, and title
    plt.title("Spacecraft Telemetry Anomaly Detection Results")
    plt.xlabel("Time Index")
    plt.ylabel("Temperature")
    plt.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.show()

plot_results(test_data, threshold)
