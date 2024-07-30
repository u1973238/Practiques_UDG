import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("")

df.head()

'''
del df['Unnamed_ 0']
df.head()
'''

df.corr(method="pearson")
df.corr()

plt.matshow(df.corr())