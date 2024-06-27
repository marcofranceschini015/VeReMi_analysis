from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import code #code.interact(local=dict(globals(), **locals()))

def create_sequences(df, time_steps=2):
    X, y = [], []
    grouped = df.groupby(['veh', 'senderPseudo'])
    
    for _, group in tqdm(grouped, desc="Create sequences"):
        # Ensure we have an even number of records to form complete pairs
        if len(group) % time_steps != 0:
            #take = - (len(group) % time_steps)
            group = group.iloc[:-1]  # Drop last row if odd number of entries
        # Use only the normalized features for creating sequences
        group = group.sort_values(by='sendTime')
        for i in range(0, len(group) - time_steps + 1, time_steps):
            X.append(group.iloc[i:i + time_steps][['pos', 'spd', 'acl', 'hed']].values)
            y.append(group.iloc[i + time_steps - 1]['label'])
    return np.array(X), np.array(y)

# How much rows per label
n = 110000

# Load the dataset
df = pd.read_csv('/Data/ConstPos_0709.csv')

# Sort by senderPseudo and sendTime
df.sort_values(by=['veh', 'senderPseudo', 'sendTime'], inplace=True)

# Take a balanced part
df_sorted = df.sort_values(by='label')
df_head = df_sorted.head(n)
df_tail = df_sorted.tail(n)

# Concatenate the head and tail DataFrame sections
df = pd.concat([df_head, df_tail], ignore_index=True)

# Normalize the continuous features
scaler = StandardScaler()
df[['pos', 'spd', 'acl', 'hed']] = scaler.fit_transform(df[['pos', 'spd', 'acl', 'hed']])

X, y = create_sequences(df)

np.save('/X.npy', X)
np.save('/y.npy', y)