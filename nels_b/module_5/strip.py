import pandas as pd

path = "/home/nbuhrley/CSE-450-Machine-learning/nels_b/module_5/NorthWind_smallcnn100-random-Nels-module5-holdout-predictions.csv"

data = pd.read_csv(path, usecols=[1])

print(data.head())

path = path.replace("holdout-predictions", "holdout-predictions-stripped")

data.to_csv(path, index=False)

# np.savetxt(path, data[:,1], delimiter=',')