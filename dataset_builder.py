# Import
import sys
from reader import *
from label_maker import *
import code #code.interact(local=dict(globals(), **locals()))

####################
# MAIN
if len(sys.argv) > 1:  # Check if at least one argument is provided
    folder = sys.argv[1]
    print("Create the dataset for:", folder)
else:
    folder = 'ConstPos_1416'
    print("Default folder: ", folder)

folder = 'Data/' + folder

vehicle_df = read_json_files(folder, "traceJSON")
filtered_vehicle_df = vehicle_df[vehicle_df["type"] == 3]

truth_df = read_json_files(folder, "traceGroundTruth")

df = label_data(filtered_vehicle_df, truth_df)
perc = (df["label"].sum()/len(df))*100
print(f"\nPercentage of attacks or error: {perc}%")
df.to_csv(folder + ".csv", index=False)