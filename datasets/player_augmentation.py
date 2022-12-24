import pandas as pd

for i in range(1, 560):
    url = "./datasets/players/" + str(i) + ".csv"

    t = pd.read_csv(url, index_col= 0).astype(float)
    t = t.fillna(0)

    t.to_csv("./datasets/player_temp/" + str(i) + ".csv" , sep = ",")