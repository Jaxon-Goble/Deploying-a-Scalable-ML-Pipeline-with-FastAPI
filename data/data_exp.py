#Brief data exploration

import os
import pandas as pd
import numpy as np

project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")

df = pd.read_csv(data_path)

print(df.head(1))
print(df.shape)