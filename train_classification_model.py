import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import joblib

def train_classification_model(training_data_path:str):
    '''
    Train a classification model to predict anomalies in the training data.

    Parameters
    ----------
    training_data_path : str
        Path to the training data folder.

    Returns
    -------
    model : RandomForestClassifier
        Trained classification model.ç
    predictions : dataframe
        Predictions of the model.
    max_diff : float
        Maximum difference in trees between the true and predicted values.
    mean_diff : float
        Mean difference in trees between the true and predicted values.
    max_error : float
        Maximum error between the true and predicted values.
    mean_error : float
        Mean error between the true and predicted values.
    '''

    def assign_velocities(df):
        # Calculate velocities for each group and store in a new column
        df['velocity'] = df.groupby('object_id', group_keys=False).apply(
            lambda x: pd.Series(
                x['centroids_x'].diff() / x['frames'].diff(),
                index=x.index
            )
        )
        return df

    # Read all the CSV files in the folder and add a 'name' column to each DataFrame with the file name
    folder_path = os.path.join(os.getcwd(), training_data_path)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    files = []
    # names=[]
    for f in csv_files:
        df = pd.read_csv(os.path.join(folder_path, f))
        df.columns=['object_id', 'centroids_x', 'centroids_y', 'aspect_ratio', 'area', 'frames', 'scores', 'counted', 'ground_truth']
        df['name']=f.split('.')[0]
        files.append(df)
    # data = pd.concat(data_list, ignore_index=True)

    # file_names=pd.concat(names, ignore_index=True)

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

    #Define the classification forest model
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True)

    #Train the model
    model.fit(x, y)

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

    x_with_object_ids=x.copy()
    x_with_object_ids['object_id']=object_ids


    return model, predictions, max_error, mean_error, x.head(1),x_with_object_ids, y

if __name__ == "__main__":
    model, preds, max_error, mean_error,_,x,y = train_classification_model("used_gts")
    print(preds)
    # print(f"Max difference in trees: {max_diff}")
    # print(f"Mean difference in trees: {mean_diff}")
    print(f"Max error: {max_error}")
    print(f"Mean error: {mean_error}")
    print(x.head(1))
    # print(x.columns)
    print(y.head(1))

    # joblib.dump(model, 'model/stem_classifier_postprocessing_v1.pkl')