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


def merge_two_dataframes(df_labels, df_features):
    # merge two dataframes. Chooses the closest row from df_features and assigns it to the df_labels
    closest_values = df_labels.apply(lambda row: df_features.iloc[find_idxmin(row.name, df_features)], axis=1)
    return pd.concat([df_labels, closest_values], axis=1)


def read_feature_dataframes(file_name, row_number):
    to_merge = []
    try:
        to_merge.append(pd.read_csv(os.path.join(face_path, file_name), delimiter=',', skiprows=1, header=None))
    except IOError as e:
        # no data, init empty dataframe
        to_merge.append(pd.DataFrame(columns=list(range(100)), index=list(range(row_number))))

    try:
        to_merge.append(pd.read_csv(os.path.join(audio_path, file_name), delimiter=',', skiprows=1, header=None))
    except IOError as e:
        # no data, init empty dataframe
        to_merge.append(pd.DataFrame(columns=list(range(36)), index=list(range(row_number))))

    try:
        to_merge.append(pd.read_csv(os.path.join(eyes_path, file_name), delimiter=',', skiprows=1, header=None))
    except IOError as e:
        # no data, init empty dataframe
        to_merge.append(pd.DataFrame(columns=list(range(6)), index=list(range(row_number))))

    try:
        to_merge.append(pd.read_csv(os.path.join(kinect_path, file_name), delimiter=',', skiprows=1, header=None))
    except IOError as e:
        # no data, init empty dataframe
        to_merge.append(pd.DataFrame(columns=list(range(27)), index=list(range(row_number))))
    return to_merge


def get_big_table(row_number):

    for file_name in os.listdir(labels_path):
        logger.info(file_name)

        to_merge = read_feature_dataframes(file_name, row_number)
        df_labels = pd.read_csv(os.path.join(labels_path, file_name), delimiter=',')
        to_merge = [df_labels] + to_merge

        min_dt = df_labels['Time'].iloc[0]
        max_dt = df_labels['Time'].iloc[-1]

        merged_df = pd.DataFrame(index=np.arange(min_dt, max_dt + 0.01, (max_dt-min_dt) / (row_number-1)),
                                 columns=['Test'])

        for df_feature in to_merge:
            # df_feature = df_feature[df_feature.iloc[:, 1:].all(axis=1) != 0]  # delete 0-rows (needs to be tested)
            merged_df = merge_two_dataframes(merged_df, df_feature)

        yield merged_df


if __name__ == '__main__':
    print(next(get_big_table(1000)))
