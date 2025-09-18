# pipeline_orchestrator.py

import mlflow
import mlflow.sklearn
import os
import glob
import shutil
import tempfile
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils import resample

# We will import the custom training function directly
from train_classification_model import train_classification_model

# ===========================================================================
# --- 1. CONFIGURATION ---
# ===========================================================================
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "StemClassifierPostprocessing"
USED_GTS_FOLDER = "used_gts"
NEW_GTS_FOLDER = "new_gts"
FILE_TRIGGER_COUNT = 15
# We will use mean_error as the key metric. The goal is to minimize it.
# Retraining is triggered if the error is HIGHER than this threshold.
MEAN_ERROR_THRESHOLD = 10.0 # e.g., 10% mean error

# ===========================================================================
# --- 2. ADAPTED METRICS FUNCTION ---
# ===========================================================================
# This function contains the logic from the 'check_postprocessing_metrics.py',
# but it's modified to accept a loaded model object from MLflow.

def prepare_data(folder_path):
    """
    This function prepares the data for the model.

    Args:
        folder_path (str): The path to the folder with new data ('new_gts').

    Returns:
        x (pd.DataFrame): The features for the model.
        y (pd.Series): The target for the model.
    """
    
    print(f"Processing folder '{folder_path}'")
    
    # The data loading and feature engineering logic is copied directly from the script
    # --- START of copied logic ---
    folder_path = os.path.join(os.getcwd(), folder_path)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    files = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(folder_path, f))
        df.columns=['object_id', 'centroids_x', 'centroids_y', 'aspect_ratio', 'area', 'frames', 'scores', 'counted', 'ground_truth']
        df['name']=f.split('.')[0]
        files.append(df)
    
    def assign_velocities(df):
        # Calculate velocities for each group and store in a new column
        df['velocity'] = df.groupby('object_id', group_keys=False).apply(
            lambda x: pd.Series(
                x['centroids_x'].diff() / x['frames'].diff(),
                index=x.index
            )
        )
        return df
    
    for f in files:

        assign_velocities(f)

        if f['velocity'].mean()<0:
            f['velocity']=-f['velocity']

        frame_counts = f.groupby('object_id')['frames'].apply(lambda x: x.max() - x.min())
        frame_info=f.groupby('object_id')['frames'].min()

        f['frames']=[frame_counts.get(f.loc[i,'object_id']) for i in range(len(f))]
        f['first_frame']=[frame_info.get(f.loc[i,'object_id']) for i in range(len(f))]



        f['mean_velocity_fps']=[f.groupby('object_id')['velocity'].mean().get(f.loc[i,'object_id']) for i in range(len(f))]
        f['std_area']=[f.groupby('object_id')['area'].std().get(f.loc[i,'object_id']) for i in range(len(f))]
        f['adj_std_area']=f['std_area']/f['frames']

    ps=[]

    for file in files:
        ps.append(file[file['counted']==True].reset_index(drop=True))

    for f in ps:
        f['ground_truth']=f['ground_truth'].map({'True Positive':1, 1:1,'False Positive':0,0:0}).astype(bool)
        f.drop('counted',axis=1,inplace=True)


        ffp=pd.DataFrame(f.groupby('object_id')['first_frame'].mean())
        # print(ffp)
        # print(ffp.mean())
        ffp['dist']=[ffp.iloc[i,0] if i==0 else ffp.iloc[i,0]-ffp.iloc[i-1,0] for i in range(len(ffp))]
        f['dist_between_apps']=[ffp['dist'].get(f.loc[i,'object_id']) for i in range(len(f))]

        f.drop('first_frame',axis=1,inplace=True)

        normalized_speed=MinMaxScaler().fit_transform(pd.DataFrame(f['mean_velocity_fps']))

        f['adjusted_frames']=[f['frames'][i]*normalized_speed[i][0] for i in range(len(normalized_speed))]


    # Reorganize columns so that 'ground_truth' and 'preds' are the second to last and last columns
    columns = ['object_id','ground_truth','name'] + [col for col in ps[0].columns if col not in ['object_id','ground_truth','name']]

    for i in range(len(ps)):
        ps[i]=ps[i][columns]

    robust_scaled=[]
    for df in ps:
        robust_scaled.append(RobustScaler().fit_transform(df.drop(['object_id','ground_truth','name'],axis=1)))

    scaled_cols=ps[0].drop(['object_id','ground_truth','name'],axis=1).columns

    scaled_positive_stems=[]
    for i, df in enumerate(ps):
        temp=df.copy()
        temp.set_index('object_id', inplace=True)
        temp[scaled_cols]=robust_scaled[i]
        scaled_positive_stems.append(temp)

    data=pd.concat([df for df in scaled_positive_stems],axis=0)
    data.reset_index(inplace=True)
    # data.drop('object_id',axis=1,inplace=True)
    data.dropna(inplace=True)
    data.set_index('name',inplace=True)
    

    #Split the data into training and test sets
    x,y=data.drop(['ground_truth'],axis=1),data['ground_truth']

    return x,y

def check_metrics_with_mlflow_model(x,y, model):
    """
    This function is an adaptation of check_postprocessing_metrics.py.

    Args:
        x (pd.DataFrame): The features for the model.
        y (pd.Series): The target for the model.
        model: The loaded MLflow model object (from the registry).

    Returns:
        float: The mean error.
    """


    #Upsample the minority class (ground_truth=0) to balance the dataset
    from sklearn.utils import resample
    x_0 = x[y == 0]
    x_1 = x[y == 1]
    y_0 = y[y == 0]
    y_1 = y[y == 1]
    x_0_upsampled, y_0_upsampled = resample(x_0, y_0, replace=True, n_samples=len(x_1), random_state=42)
    x = pd.concat([x_0_upsampled, x_1])
    y = pd.concat([y_0_upsampled, y_1])

    object_ids=x['object_id']
    x.drop('object_id',axis=1,inplace=True)

    #Make predictions on the test set
    y_pred = model.predict(x)

    #Create a dataframe with file_names, y and y_pred
    predictions=pd.DataFrame({'file_name':y.index.tolist(), 'object_id':object_ids, 'y':y, 'y_pred':y_pred})

    predictions['y_pred'] = predictions.groupby(['file_name', 'object_id'])['y_pred'].transform(lambda x: 1 if x.mean() > 0.5 else 0)

    predictions= predictions.groupby(['file_name','object_id']).mean()
    predictions=predictions.groupby(['file_name']).sum()
    predictions['diff']=predictions['y']-predictions['y_pred']
    predictions['error']=100*abs(predictions['diff'])/predictions['y']

    max_diff=abs(predictions['diff']).max()
    mean_diff=abs(predictions['diff']).mean()
    max_error=abs(predictions['error']).max()
    mean_error=abs(predictions['error']).mean()

    return mean_error



# ===========================================================================
# --- 3. THE MAIN ORCHESTRATOR LOGIC ---
# ===========================================================================

def run_orchestrator():
    """
    The main pipeline function that orchestrates the entire MLOps cycle.
    """
    print("Starting pipeline orchestrator...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # --- Step 1: Check Trigger ---
    new_files_list = glob.glob(os.path.join(NEW_GTS_FOLDER, "*.csv"))
    if len(new_files_list) < FILE_TRIGGER_COUNT:
        print(f"Found {len(new_files_list)} files. Waiting for {FILE_TRIGGER_COUNT}. Exiting.")
        return

    # --- Step 2: Monitor Production Model ---
    print(f"Found {len(new_files_list)} new files. Monitoring production model performance.")
    try:
        prod_model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"
        production_model = mlflow.sklearn.load_model(prod_model_uri)
        print("Successfully loaded 'production' model from MLflow Registry.")
    except Exception as e:
        print(f"Could not load a 'production' model. Please deploy a model first. Error: {e}")
        return

    # Call our adapted metrics function
    x,y=prepare_data(NEW_GTS_FOLDER)
    current_mean_error = check_metrics_with_mlflow_model(x,y, production_model)
    # current_mean_error = monitoring_metrics.get("mean_error", float('inf'))

    with mlflow.start_run(run_name="ModelMonitoring") as monitoring_run:
        print(f"Logging monitoring metric: {current_mean_error}")
        mlflow.log_metric("mean_error", current_mean_error)

    # --- Step 3: Decide to Retrain ---
    if current_mean_error <= MEAN_ERROR_THRESHOLD:
        print(f"Mean Error {current_mean_error:.2f}% is within the threshold of {MEAN_ERROR_THRESHOLD}%. No retraining needed.")
        return

    print(f"Mean Error {current_mean_error:.2f}% is above the threshold of {MEAN_ERROR_THRESHOLD}%. Retraining triggered!")

    # --- Step 4: Retrain the Model ---
    with mlflow.start_run(run_name="ModelRetraining", nested=True) as retraining_run:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temporary training directory: {temp_dir}")
            
            # Copy all files to the temporary directory
            used_files = glob.glob(os.path.join(USED_GTS_FOLDER, "*"))
            for f in used_files + new_files_list:
                shutil.copy(f, temp_dir)
            
            print(f"Copied {len(used_files) + len(new_files_list)} files for training.")

            # Call your original training function from the imported file
            challenger_model, _, _, mean_err, input_example,new_x,new_y = train_classification_model(temp_dir)
            
        print("Temporary training directory removed.")
        
        challenger_metrics = {"mean_error": mean_err}
        print(f"New model trained. Metrics: {challenger_metrics}")
        mlflow.log_metrics(challenger_metrics)
        
        # Log and register the new model
        model_info = mlflow.sklearn.log_model(
            sk_model=challenger_model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=input_example
        )

    # --- Step 5: Compare & Promote ---
    print("Comparing challenger model with production model...")
    challenger_mean_error = challenger_metrics.get("mean_error", float('inf'))
    
    # Get the current production version details to find its logged error
    # prod_version_details = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
    # prod_run = client.get_run(prod_version_details.run_id)
    # prod_mean_error = prod_run.data.metrics.get("mean_error", float('inf'))
    prod_mean_error = check_metrics_with_mlflow_model(new_x,new_y, production_model)

    print(f"Challenger Model Mean Error: {challenger_mean_error:.2f}%")
    print(f"Production Model Mean Error: {prod_mean_error:.2f}%")

    # A LOWER error is BETTER
    if challenger_mean_error < prod_mean_error:
        print("New model is better! Promoting to Production.")

        # Get the version of the challenger model we just created
        challenger_version_info = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[0]
        challenger_version = challenger_version_info.version

        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="production",
            version=challenger_version
        )
        
        # --- Step 6: Cleanup ---
        print(f"Promotion successful. Moving files from '{NEW_GTS_FOLDER}' to '{USED_GTS_FOLDER}'.")
        for f in new_files_list:
            shutil.move(f, os.path.join(USED_GTS_FOLDER, os.path.basename(f)))
    else:
        print("New model is not better. Keeping current production model.")

    print("Pipeline run finished.")


if __name__ == "__main__":
    run_orchestrator()