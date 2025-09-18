import mlflow
import joblib
from pipeline_orchestrator import prepare_data, check_metrics_with_mlflow_model

# --- Configuration ---
# 1. Set the tracking URI to running server
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 2. Define the path to existing model file
EXISTING_MODEL_PATH = "model/stem_classifier_postprocessing_v1.pkl"

# 3. Choose a name for model in the registry
REGISTERED_MODEL_NAME = "StemClassifierPostprocessing"

# 4. Folder with initial training data
INITIAL_TRAINING_DATA_FOLDER = "used_gts"

# --- Main Script ---
# Load pre-trained model from the file
print(f"Loading model from: {EXISTING_MODEL_PATH}")
model = joblib.load(EXISTING_MODEL_PATH)

# Process the initial training data
x,y=prepare_data(INITIAL_TRAINING_DATA_FOLDER)
mean_error = check_metrics_with_mlflow_model(x,y, model)

# Start a new MLflow run. This is a container for logging activity.
with mlflow.start_run(run_name="Registering Initial Production Model") as run:
    print("Logging and registering the model...")

    # Log the mean error
    print(f"Logging mean error: {mean_error}")
    mlflow.log_metric("mean_error", mean_error)

    # Log the model to MLflow
    # This is the key step: MLflow copies the model to its artifact store
    # and registers it under the name you provide.
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model", # This is a subfolder within the run's artifacts
        registered_model_name=REGISTERED_MODEL_NAME
    )
    
    run_id = run.info.run_id
    print(f"✅ Model successfully logged under Run ID: {run_id}")
    print(f"✅ Model registered as '{REGISTERED_MODEL_NAME}' Version 1.")