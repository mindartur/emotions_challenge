import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('data.composer')

FILES_DIR = 'data/train/'
COMPOSED_TABLES_DIR = 'data/'

labels_path = os.path.join(FILES_DIR, 'labels')
face_path = os.path.join(FILES_DIR, 'face_nn')
audio_path = os.path.join(FILES_DIR, 'audio')
eyes_path = os.path.join(FILES_DIR, 'eyes')
kinect_path = os.path.join(FILES_DIR, 'kinect')


def find_idxmin(dt, df_features):
    # TODO: add time checking - if the nearest neighbour is too far by time
    return (df_features.index - dt).to_series().reset_index(drop=True).abs().idxmin()


def merge_two_dataframes(df_labels, df_features, create_mask=False):
    # merge two dataframes. Chooses the closest row from df_features and assigns it to the df_labels
    # df_features.fillna(0)
    closest_values = df_labels.apply(lambda row: df_features.iloc[find_idxmin(row.name, df_features)], axis=1)
    if create_mask:
        closest_values['mask'] = (closest_values == 0).all(axis=1)
        closest_values['mask'] = closest_values['mask'].astype(int)

    return pd.concat([df_labels, closest_values], axis=1)


def read_feature_dataframes(file_name, row_number):
    to_merge = []
    try:
        to_merge.append(pd.read_csv(os.path.join(face_path, file_name), delimiter=',', skiprows=1, header=None))
    except IOError as e:
        # no data, init empty dataframe
        to_merge.append(pd.DataFrame(0, columns=list(range(101)), index=list(range(row_number))))
    try:
        to_merge.append(pd.read_csv(os.path.join(audio_path, file_name), delimiter=',', skiprows=1, header=None))
    except IOError as e:
        # no data, init empty dataframe
        to_merge.append(pd.DataFrame(0, columns=list(range(37)), index=list(range(row_number))))

    try:
        to_merge.append(pd.read_csv(os.path.join(eyes_path, file_name), delimiter=',', skiprows=1, header=None))
    except IOError as e:
        # no data, init empty dataframe
        to_merge.append(pd.DataFrame(0, columns=list(range(7)), index=list(range(row_number))))
        
    try:
        to_merge.append(pd.read_csv(os.path.join(kinect_path, file_name), delimiter=',', skiprows=1, header=None))
    except IOError as e:
        # no data, init empty dataframe
        to_merge.append(pd.DataFrame(0, columns=list(range(28)), index=list(range(row_number))))
    return to_merge


def get_big_table(row_number=None):

    for file_name in os.listdir(labels_path):
        logger.info(file_name)
        df_labels = pd.read_csv(os.path.join(labels_path, file_name), delimiter=',')
        if not row_number:
            row_number = len(df_labels)
        to_merge = read_feature_dataframes(file_name, row_number)
        # to_merge = [df_labels] + to_merge
        
        min_dt = df_labels['Time'].iloc[0]
        max_dt = df_labels['Time'].iloc[-1]

        merged_df = pd.DataFrame(index=np.arange(min_dt, max_dt + 0.01, (max_dt-min_dt) / (row_number-1)),
                                 columns=['Test'])
        
        merged_df = merge_two_dataframes(merged_df, df_labels)
        for df_feature in to_merge:
            # df_feature = df_feature[df_feature.iloc[:, 1:].all(axis=1) != 0]  # delete 0-rows (needs to be tested)
            merged_df = merge_two_dataframes(merged_df, df_feature, create_mask=True)

        yield file_name, merged_df

def transform_data():
    for i, (file_name, df) in enumerate(get_big_table()):
        print(i, file_name, df.shape)
        df.to_csv(os.path.join('transformed_data_1', file_name))
    
        
if __name__ == '__main__':
    transform_data()
