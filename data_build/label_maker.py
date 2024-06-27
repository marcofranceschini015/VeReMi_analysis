import pandas as pd
import numpy as np

# Euclidean norm to replace
# pos, spd, acl, hed
def euclidean_norm(vector):
    return np.linalg.norm(vector)


def apply_euclidean_norms(df):
    for col in df.columns:
        if col in ['pos', 'spd', 'acl', 'hed']:
            # Apply the norm to each row in the column
            df.loc[:, col] = df[col].apply(lambda x: euclidean_norm(np.array(x)))
    return df


def merge_data(messages, truth, columns):
    # Merge data by messageID
    merged_df = pd.merge(messages, truth, on='messageID', how='left', suffixes=('', '_truth'))

    # Creating a new list that includes both the original and their _truth counterparts
    columns_to_keep = columns + [col + '_truth' for col in columns if col + '_truth' in merged_df.columns]

    # Filter the merged DataFrame to keep only these columns
    final_df = merged_df[columns_to_keep]

    return final_df


def compare_data(df, columns):
    df['label'] = 0

    # Iterate over each column, compare with its _truth counterpart, and update 'label'
    for col in columns:
        if col == 'sendTime':
            # Specific condition for 'sendTime' where the absolute difference must be greater than 0.01
            df['label'] = np.where((df['sendTime'] - df['sendTime_truth']).abs() > 0.01, 1, df['label'])
        else:
            # General condition for all other columns where any difference sets the label to 1
            df['label'] = np.where(df[col] != df[col + '_truth'], 1, df['label'])
    return df


def label_data(messages, truth):
    print("\nProcessing data ...")
    columns_compare = ['sendTime', 'senderPseudo', 'pos', 'spd', 'acl', 'hed']
    columns_keep = ['sendTime', 'senderPseudo', 'pos', 'spd', 'acl', 'hed', 'veh']
    messages = apply_euclidean_norms(messages)
    truth = apply_euclidean_norms(truth)
    merged_df = merge_data(messages, truth, columns_keep)

    final_df = compare_data(merged_df, columns_compare)
    columns_keep.append('label')
    return final_df[columns_keep]