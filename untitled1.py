import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
# pandas, numpy, matplotlob.pyplot 라이브러리를 간결하게 사용하기위해 pd,np,plt로 선언해 주었다.
# 수의 올림(math.ceil)을 사용하기 위해 math 라이브러리도 m으로 import해주었다.

temp_data = pd.read_csv("C:\\Users\\GongHyeon\\chap2_data.csv")
#판다스 라이브러리의 읽기를 사용하여 내가 사용할 데이터 chap2_data를 불러왔다.
#500개의 x,y데이터를 불러왔다.
temp_data.info()
#데이터의 크기가 얼마나 되는지 확인했다.
#그 결과 2개의 열과 500개의 행임을 알 수 있었다.

data_numpy=temp_data.to_numpy(dtype='float32')
#내가 불러온 데이터를 사용하기 위해 to_numpy를 통해 배열로 불러와주었다.

column_data_x = data_numpy[:,0]
column_data_y = data_numpy[:,1]
x=column_data_x 
y=column_data_y
#내가 불러온 데이터에서 y값과 x값을 구분해 주기위해
#각각 다른 변수로 x값과 y값으로 분류해 주었다. 
#column_data_x에 해당하는 부분을 사용하기 쉽게 x로 설정하고 column_data_y에 해당하는 부분을
#y로 설정해주었다.


def X_Data_Sets(a,b):
    global X_Training_set
    X_Training_set=x[0:m.ceil((x.size/10)*a)]
    global X_Validation_set
    X_Validation_set=x[m.ceil(x.size/10*a):m.ceil((x.size/10)*a+(x.size/10)*b)]
    global X_Test_set
    X_Test_set=x[m.ceil((x.size/10)*a+(x.size/10)*b):x.size]
    return X_Training_set,X_Validation_set,X_Test_set

def Y_Data_Sets(a,b):
    global Y_Training_set
    Y_Training_set=y[0:m.ceil((y.size/10)*a)]
    global Y_Validation_set
    Y_Validation_set=y[m.ceil(y.size/10*a):m.ceil((y.size/10)*a+(y.size/10)*b)]
    global Y_Test_set
    Y_Test_set=y[m.ceil((y.size/10)*a+(y.size/10)*b):y.size]
    return Y_Training_set,Y_Validation_set,Y_Test_set

# 사용자 지정함수를 이용하여 데이터를 split해주었다. 데이터는 Training,Validation,Test로 분리시켰다.
# 내가 원하는 비율의 합이 10이였기 때문에 (예시, 5:2:3 > 5+2+3 = 10) 전체갯수를 10으로 나누고 비율의 값을
# 곱하도록 코드를 작성하였다. 이렇게 되면 순서대로 내가 원하는 비율만큼 데이터를 나눌 수 있기 때문이다.
# 또한 값이 겹쳐지거나 빠지지 않게 올림을 사용해서 데이터를 모두 사용할 수 있도록 하였다.
# global을 사용해서 전역변수로 사용할 수 있도록 했다.
# X와 Y를 구분해서 진행하였다.

X_Data_Sets(7,0)
Y_Data_Sets(7,0)

# 7:0:3으로 나누기 위해 사용자 지정함수를 위와같이 사용하였다.
ms = []
# ms = []는 뮤에 해당하는 m에 정보를 넣기 위해 만들어준 빈 배열이다.
temp_pi=[]
# temp_pi는 가우시안 기저함수값을 쌓기위한 빈 배열이다.
mu = []
# mu = []는 리스트의 뮤값인 ms를 배열로 사용하기 위해, array로 바꿔주려고 선언하였다.
main_stack= []
# main_stack 은 N개의 훈련 데이터 입력에 대한 행렬을 넣기위해 선언해준 배열이다.
global K
# K값은 입력값이지만 사용자 지정함수에 들어가서 다른곳에 쓰일때 전역변수로 인식이 안되어서 선언해주었다.

y_pred_result = []
result_stack = []
store = []
mse_stack = []
MSE=[]
K=10


delta=(max(X_Training_set)-min(X_Training_set))/(K-1)
print(delta)
for k in range(K):
    m=(min(X_Training_set))+(((max(X_Training_set)-min(X_Training_set))*k)/K-1) # m이 상수로 계산되고 그걸 배열로 저장해주었다.
    ms.append(m)
    mu=np.array(ms)   
    print(mu)
    
for j in np.arange(len(X_Training_set)):
    for k in np.arange(K):
        temp_value = np.exp((-0.5)*(((X_Training_set[j]-mu[k])/((delta)))**2))
        temp_pi.append(temp_value)
        main_stack=temp_pi
a=np.array(main_stack)
b=a.reshape([len(X_Training_set),K])
aa=np.linalg.inv(np.matmul(b.transpose(),b))
bb=b.transpose()
c = np.matmul(np.matmul(aa,bb),Y_Training_set)

o=b[:,0:K-1]
new=np.ones([350,1])
ap=np.hstack((o,new))

for j in np.arange(350):
    pred_y=np.matmul(c,ap[j])
    store.append(pred_y)
    print(store)
# =============================================================================
# for k in np.arange(K-1):
#     for j in np.arange(len(X_Training_set)):
#         result=(np.exp((-0.5)*((X_Training_set[j]-mu[k])/delta)**2))
#         result_stack.append(result)
#         result_stack_1=result_stack
#         matrix=np.array(result_stack_1)
#     matrix_plus=matrix.tolist()
#     matrix_plus.append(1)
#     matrix_plus1=np.array(matrix_plus)
#     y_pred=np.matmul(c,matrix_plus1)
#     y_pred_result.append(y_pred)
#     y_pred_result
# =============================================================================

mse = ((store-Y_Training_set)**2).mean()
    

    

# =============================================================================
# MSE = ((b-y_col)**2).mean()
# =============================================================================

plt.figure(1)
plt.grid(True, alpha=0.5)
plt.plot(X_Training_set,store,color='g')
# =============================================================================
# plt.plot(X_Training_set,MSE,color='r')
# =============================================================================
plt.scatter(X_Training_set,Y_Training_set,color='b')
plt.xlabel('weight')
plt.ylabel('length of spring')
plt.title('5week Practice #3')
plt.legend(['예측값 y'])

