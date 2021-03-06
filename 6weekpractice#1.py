import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pandas, numpy, matplotlob.pyplot 라이브러리를 간결하게 사용하기위해 pd,np,plt로 선언해 주었다.

temp_data = pd.read_csv("C:\\Users\\GongHyeon\\lin_regression_data_01.csv")
#판다스 라이브러리의 읽기를 사용하여 내가 사용할 데이터(용수철의 길이)를 불러왔다.
#엑셀을 수정하여 25개의 데이터를 모두 받아왔다.
temp_data.info()
#데이터의 크기가 얼마나 되는지 확인했다.
#그 결과 2개의 열과 25개의 행임을 알 수 있었다.
#용수철길이 y=kx에서 각 칼럼이 x와y 값들이다.

data_numpy=temp_data.to_numpy(dtype='float32')
#내가 불러온 데이터를 사용하기 위해 to_numpy를 통해 배열로 불러와주었다.

column_data_x = data_numpy[:,0]
column_data_y = data_numpy[:,1]
#용수철 길이에 대한 식 y=kx에 대하여, 내가 불러온 데이터에서 y값과 x값을 구분해 주기위해
#각각 다른 변수로 x값과 y값으로 분류해 주었다. 용수철의 식 y=kx에서,
#x에 해당하는 부분을 column_data_x로 설정하고 y에 해당하는 부분을
#column_data_y로 설정해주었다.


x=column_data_x 
y=column_data_y
#식을 한눈에 보기 편하게 칼럼_x를 x로, 칼럼_y를 y로 지정해 놓고 이번 실습을 진행했다.

c=[]
d=[]
# c=[],d=[]는 append로 쌓은값들을 넣어주기위해 선언한 공간이다.

for j in np.arange(0,25):
    random_noise=(np.random.rand(11,1)-0.5)*2
    for h in np.arange(0,10):
        made_x=x[j]+random_noise[h]
        b=made_x.tolist()
        c.append(b)
new_x=np.array(c)
# x데이터 25개중 한 x의 값들에 대해 각각에 노이즈를 다르게 주기 위해 이중 포문을 사용하였다. 
# x값이 하나 정해졌을때 랜덤한 노이즈 10개를 더하거나 빼서, 새로운 x값들인 made_x를 만들었고,
# 그 값을 리스트로 만들어 쌓아올린 후 다시 배열로 만들어 주었다. 이 최종적으로 노이즈가 더해져 만들어진 x값(250개)들을 new_x라고 하였다.
        

for j in np.arange(0,25):
    random_noise=(np.random.rand(11,1)-0.5)*2
    for h in np.arange(0,10):
        made_y=y[j]+random_noise[h]
        b=made_y.tolist()
        d.append(b)
new_y=np.array(d)
# y데이터 25개중 한 y의 값들에 대해 각각에 노이즈를 다르게 주기 위해 x데이터 다룰 때와 마찬가지로 이중 포문을 사용하였다. 
# y값이 하나 정해졌을때 랜덤한 노이즈 10개를 더하거나 빼서, 새로운 y값들인 made_y를 만들었고,
# 그 값을 리스트로 만들어 쌓아올린 후 다시 배열로 만들어 주었다. 이 최종적으로 노이즈가 더해져 만들어진 y값(250개)들을 new_y라고 하였다.

plt.grid(True, alpha=0.5)
# plt.grid(True)와 같이 설정하면, 그래프의 x, y축에 대해 그리드가 표시된다. 
# axis로 축을 정할 수 있고, 디폴트는 x,y축 모두 그리는 both이다. 나는 x,y축 모두 그렸다.
# alpha는 그리드의 투명도이다.

plt.scatter(new_x,new_y,color='b')
plt.scatter(x,y,color='r')
plt.xlabel('Weight')
plt.ylabel('Length of Spring')
plt.title('6Week Practice Make Random Noise Data')
plt.legend(['Argument Set','Original Set'])

# scatter에 내가 원하는 x,y값을 넣고 색깔을 넣어 점들을 나타낼 수 있다.
# x축의 label은 추의무게인 Weight, y축의 label은 용수철의 길이인 Length of Spring을 넣었다.
# title은 6주차 과제 추가 데이터 만들기이다.
# Legend를 이용하여 scatter에 이름을 구분지어 주었다.