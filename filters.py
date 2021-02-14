import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.random((200,3)))
df['date'] = pd.date_range('2000-1-1', periods=200, freq='D',index='date')
mask = (df['date'] > '2000-6-1') & (df['date'] <= '2000-6-10')
df2=df.loc[mask]
print(df.loc[mask])
print(df.dtypes)
print(df2.index)
print(df.index)