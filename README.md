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
colab을 사용해 상자 그림, 산점도 분석. `colab`

```py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pkmn = pd.read_csv('drive/MyDrive/Colab Notebooks/Pokemon.csv')

pkmn.head()

pkmn = pkmn.drop(['Generation', 'Legendary'],1)

sns.jointplot(x="HP", y="Attack", data=pkmn);

sns.boxplot(y="HP", data=pkmn);

sns.boxplot(data=pkmn);
```

## 2023-07-16

colab, Jupyter를 사용하여 점 도표, 3차원 산점도 분석 `Jupyter`, `colab`

``` python
import numpy as np
import pandas as pd
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.tools import FigureFactory as ff
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import os
import warnings
warnings.filterwarnings("ignore")

vgsales=pd.read_csv("C:\\Users\\user\\Documents\\dev\\vgsales.csv")

sales=vgsales.head(100)

traceNA = go.Scatter(
                    x = sales.Rank,
                    y = sales.NA_Sales,
                    mode = "markers",
                    name = "North America",
                    marker = dict(color = 'rgba(28, 149, 249, 0.8)',size=8),
                    text= sales.Name)

traceEU = go.Scatter(
                    x = sales.Rank,
                    y = sales.EU_Sales,
                    mode = "markers",
                    name = "Europe",
                    marker = dict(color = 'rgba(249, 94, 28, 0.8)',size=8),
                    text= sales.Name)
traceJP = go.Scatter(
                    x = sales.Rank,
                    y = sales.JP_Sales,
                    mode = "markers",
                    name = "Japan",
                    marker = dict(color = 'rgba(150, 26, 80, 0.8)',size=8),
                    text= sales.Name)
traceOS = go.Scatter(
                    x = sales.Rank,
                    y = sales.Other_Sales,
                    mode = "markers",
                    name = "Other",
                    marker = dict(color = 'lime',size=8),
                    text= sales.Name)

data = [traceNA, traceEU,traceJP,traceOS]
layout = dict(title = 'North America, Europe, Japan and Other Sales of Top 100 Video Games',
              xaxis= dict(title= 'Rank',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white"),
              yaxis= dict(title= 'Sales(In Millions)',ticklen= 5,zeroline= False,zerolinewidth=1,gridcolor="white",),
              paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)' )
fig = dict(data = data, layout = layout)
iplot(fig)


data1000=vgsales.iloc[:1000,:]

data1000["normsales"] = (data1000["Global_Sales"] - np.min(data1000["Global_Sales"]))/(np.max(data1000["Global_Sales"])-np.min(data1000["Global_Sales"]))

data1000.Rank=data1000.Rank.astype("str")
data1000.Global_Sales=data1000.Global_Sales.astype("str")
trace1 = go.Scatter3d(
    y=data1000["Publisher"],
    x=data1000["Year"],
    z=data1000["normsales"],
    text="Name:"+ data1000.Name +","+" Rank:" + data1000.Rank + " Global Sales: " + data1000["Global_Sales"] +" millions",
    mode='markers',
    marker=dict(
        size=data1000['NA_Sales'],
        color = data1000['normsales'],
        colorscale = "Rainbow",
        colorbar = dict(title = 'Global Sales'),ㄴ
        line=dict(color='rgb(140, 140, 170)'),
       
    )
)

data=[trace1]

layout=go.Layout(height=800, width=800, title='Top 1000 Video Games, Release Years, Publishers and Sales',
            titlefont=dict(color='rgb(20, 24, 54)'),
            scene = dict(xaxis=dict(title='Year',
                                    titlefont=dict(color='rgb(20, 24, 54)')),
                            yaxis=dict(title='Publisher',
                                       titlefont=dict(color='rgb(20, 24, 54)')),
                            zaxis=dict(title='Global Sales',
                                       titlefont=dict(color='rgb(20, 24, 54)')),
                            bgcolor = 'whitesmoke'
                           ))
 
graph=go.Figure(data=data, layout=layout)
iplot(graph)
```
go.Figure
- 이 클래스는 주어진 데이터와 레이아웃을 기반으로 그래프를 생성한다.