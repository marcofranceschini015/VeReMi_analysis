import pandas as pd
import code #code.interact(local=dict(globals(), **locals()))
import matplotlib.pyplot as plt
import sys

def plot_behaviour(df, name):
    malign = 0
    benign = 0
    mixed = 0
    for pseudo in df['senderPseudo'].unique():
        oneshot = df[df['senderPseudo'] == pseudo]
        if len(oneshot['label'].unique()) > 1:
            mixed = mixed + 1
        elif oneshot['label'].unique() == 1:
            malign = malign + 1
        else:
            benign = benign + 1
    
    tot = malign + benign + mixed
    perc_mal = round((malign / (tot)) * 100, 2)
    perc_ben = round((benign / (tot)) * 100, 2)
    perc_mixed = round((mixed / (tot)) * 100, 2)

    percentages = [perc_ben, perc_mal, perc_mixed]
    labels = [f'{perc_ben}%', f'{perc_mal}', f'{perc_mixed}']
    categories = ['Benign', 'Malign', 'Mixed']

    # Plotting
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, percentages, color=['blue', 'green', 'red'])

    # Adding labels on top of the bars
    for bar, label in zip(bars, labels):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, label,
                ha='center', va='bottom')

    plt.xlabel('Vehicles Categories')
    plt.ylabel('Percentage')
    # Adding grid
    plt.grid(True)
    plt.title(name)

    # Save the plot as PDF
    plt.savefig(f'/Data/{name}.pdf')



#### MAIN
if len(sys.argv) > 1:  # Check if at least one argument is provided
    file = sys.argv[1]
else:
    file = 'ConstPos_1416'

df = pd.read_csv(f"/Data/{file}.csv")
plot_behaviour(df, file)