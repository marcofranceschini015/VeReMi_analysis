import pandas as pd
import numpy as np
from tqdm import tqdm

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
        # We use np.where to vectorize the comparison for each column pair
        df['label'] = np.where(df[col] != df[col + '_truth'], 1, df['label'])
    
    return df


def label_data(messages, truth):
    print("\nProcessing data ...")
    columns = ['sendTime', 'senderPseudo', 'pos', 'spd', 'acl', 'hed']
    columns_to_compare = ['senderPseudo', 'pos', 'spd', 'acl', 'hed']
    messages = apply_euclidean_norms(messages)
    truth = apply_euclidean_norms(truth)
    merged_df = merge_data(messages, truth, columns)

    final_df = compare_data(merged_df, columns_to_compare)
    columns.append('label')
    return final_df[columns]