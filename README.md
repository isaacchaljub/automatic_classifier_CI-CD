# Automated MLOps Retraining Pipeline with MLflow

This project implements a complete, automated MLOps pipeline for a classification model. It actively monitors the model's performance on new, incoming data and automatically triggers a retraining, evaluation, and deployment cycle if performance degrades below a set threshold. The entire machine learning lifecycle is tracked and managed using **MLflow**.

---

## Features

- **Automated Monitoring**: Continuously evaluates the production model's performance on new data batches.
- **Conditional Retraining**: Triggers a model retraining job only when the performance metric (mean error) crosses a predefined threshold.
- **Fair Model Comparison**: Re-evaluates the old production model and the new challenger model on the exact same test set to ensure a true apples-to-apples comparison.
- **Automated Promotion**: If the new model is better, it is automatically promoted to the "production" stage in the MLflow Model Registry using aliases.
- **Data Archiving**: Automatically moves processed data from the new data folder to an archive folder upon successful model promotion.
- **Centralized Tracking**: Uses a central MLflow Tracking Server to log all experiments, metrics, parameters, and model artifacts.
- **Scheduled Execution**: Designed to be run automatically on a schedule using tools like `cron`.

---

## Project Architecture & Workflow

The pipeline operates in a continuous cycle, orchestrated by a single Python script.

1.  **Trigger**: The pipeline checks the `new_gts/` folder. It only proceeds if there is a minimum number of new data files.
2.  **Monitor**: It loads the current **production** model from the MLflow Model Registry and evaluates its performance on the new data. The performance metrics are logged to an MLflow run called `ModelMonitoring`.
3.  **Decide**: The script compares the production model's performance against a predefined threshold. If performance is acceptable, the pipeline stops.
4.  **Retrain**: If performance has degraded, a retraining job is triggered.
    - A temporary folder is created.
    - All historical data from `used_gts/` and new data from `new_gts/` are copied into it.
    - A new model is trained on this combined dataset.
    - The new model, its metrics, and its signature are logged to a new MLflow run called `ModelRetraining` and registered.
5.  **Compare & Promote**: The new "challenger" model and the old "production" model are both evaluated on a consistent test set.
    - If the challenger is better, the `production` alias is moved to the new model's version in the MLflow Model Registry.
6.  **Cleanup**: If the promotion was successful, the processed files are moved from `new_gts/` to `used_gts/`, completing the cycle.

---

## Directory Structure

```
.
├── model/
│   └── stem_classifier_postprocessing_v1.pkl   # Your initial pre-trained model
├── new_gts/                                    # Folder for new, incoming data files
├── used_gts/                                   # Archive for data used in production models
├── mlflow_server/                              # (Optional) For running the MLflow server
│   ├── artifacts/
│   └── backend_db/
├── .gitignore
├── pipeline_orchestrator.py                    # The main pipeline controller script
├── register_first_model.py                     # Script to register the initial v1 model
├── train_classification_model.py               # Your custom model training logic
└── requirements.txt                            # Project dependencies
```

---

## 🛠Getting Started

Follow these steps to set up and run the pipeline locally.

### 1. Prerequisites

- Python 3.8+
- `pip` for package installation

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 3. Setup the Environment

First, create a `requirements.txt` file with the following content:

```txt
mlflow
scikit-learn
pandas
numpy
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Initial Folder and Model Setup

1.  Create the necessary data folders:
    ```bash
    mkdir new_gts used_gts
    ```
2.  Place your initial training data files into the `used_gts/` folder.
3.  Place your pre-trained `stem_classifier_postprocessing_v1.pkl` file into a `model/` folder.

### 5. Start the MLflow Server

In a dedicated terminal, start the central MLflow server. This server will track everything.

```bash
# Optional: Create a directory for the server
mkdir mlflow_server
cd mlflow_server

# Start the server
mlflow server \
    --backend-store-uri sqlite:///backend_db/mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0
```
**Keep this terminal running.**

### 6. Register Your First Model

Before the pipeline can run, it needs a baseline "production" model in the registry.

1.  In a **new terminal**, run the registration script:
    ```bash
    python register_first_model.py
    ```
2.  Go to the MLflow UI at `http://127.0.0.1:5000`.
3.  Navigate to the **Models** tab, click on `StemClassifierPostprocessing`, and find **Version 1**.
4.  Assign it the alias **`production`**.

---

## Usage

### 1. Add New Data

Place 15 or more new `.csv` data files into the `new_gts/` folder. (This number can be changed in pipeline_orchestrator.py)

### 2. Run the Pipeline Manually

Execute the orchestrator script to run one full cycle of the pipeline.

```bash
python pipeline_orchestrator.py
```
Observe the terminal output and check the MLflow UI to see the new runs and model versions.

### 3. Automate with a Scheduler (Production)

For true automation, run the orchestrator on a schedule using `cron`.

1.  Open your crontab: `crontab -e`
2.  Add a line to run the script daily at 2 AM and log the output:
    ```bash
    0 2 * * * /usr/bin/python3 /path/to/your/project/pipeline_orchestrator.py >> /path/to/your/project/pipeline.log 2>&1
    ```

---

## 📜 Key Scripts

- **`pipeline_orchestrator.py`**: The main controller. It contains the primary workflow logic for monitoring, triggering, and managing the MLOps cycle.
- **`train_classification_model.py`**: Your self-contained, custom logic for training the classification model from a folder of data.
- **`register_first_model.py`**: A one-time setup script to get your initial pre-trained model into the MLflow Model Registry with its baseline metrics.

---

## 📄 License

This project is licensed under the MIT License.
