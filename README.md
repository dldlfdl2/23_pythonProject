# 23_pythonProject

## 💖 2023-06-17

데이터의 개념과 이해 `pandas`, `matplotlib`

``` python
import matplotlib.pyplot as plt
import random
import pandas as pd

path = '/content/cctv.csv'
cctv = pd.read_csv(path , encoding='cp949')

cctv.head()
cctv.info()

cctv = cctv.dropna()
cctv.isna().sum()
```

## 2023-06-25

colab을 사용하여 막대 차트, 원그래프 분석 `Jupyter`, `colab`

``` python
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/content/data_science_job.csv',encoding='latin-1')
df.head()

df1 = df.dropna()

df1.head()

import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
df1['Experience level'].value_counts().plot.pie(autopct='%1.1f%%')
```

matplotlib

-원형 차트를 만들기 위해 필요한 시각화 라이브러리

## 2023-07-02
seaborn을 사용하여 막대 차트, 산점도 행렬, 상관관계 분석 시각화 `seaborn`, `colab`

``` python
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nRowsRead = 1000
df1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/50ulke.csv')
df1.dataframeName = '50ulke.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 18, 10)
```

plotPerColumnDistribution 
- 막대 차트를 만들때 사용하는 함수이다.

plotCorrelationMatrix
- 상관관계 분석을 시각화 할때 사용하는 함수이다.

plotScatterMatrix
- 산점도 행렬표를 만들때 사용하는 함수이다.

## 2023-07-09
