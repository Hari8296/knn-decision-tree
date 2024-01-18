Name :- Hari singh r 
Batch id :-  DSWDMCOD 25082022 B


1
Business problems 
1.1 what is the Business odjectives 
Ans:-To classify the Glass type based on various material used in manufacturing the glass.

1.2 What are the constraints      
Ans:-Correctness of the amounts in the contents is necessary to avoid misclassification of the type of glass.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

glass=pd.read_csv("D:/assignments of data science/12 knn & decision tree/glass.csv")
glass

glass.info() # checking data details
glass.head() # checking top 5  
glass.duplicated().sum()
glass.describe()
glass.isna().sum()

glass=glass.iloc[:,:]

glass.mean()
glass.median()
list(glass.mode())
glass.skew()
glass.kurt()
glass.var()

for i in glass.columns:
    plt.hist(glass[i])
    plt.xlabel(i)
    plt.show()


# normalized functoion 
    
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

glass_n=norm_func(glass.iloc[:,:])
glass_n.describe()

x=np.array(glass_n.iloc[:,:])
y=np.array(glass['Type'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)  

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

lab_enc=preprocessing.LabelEncoder()
encoded=lab_enc.fit_transform(y)
encoded

knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)

pred=knn.predict(x_test)
pred

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
pd.crosstab(y_test,pred,rownames=['Actual'],colnames=['predications'])

pred_train = knn.predict(x_train)
print(accuracy_score(y_train,pred_train))
pd.crosstab(y_train,pred_train, rownames=['Actual'],colnames=['Predications'])

acc=[]

for i in range( 3,50,2):
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train,y_train)
    train_acc=np.mean(neigh.predict(x_train)==y_train)
    test_acc=np.mean(neigh.predict(x_test)==y_test)
    acc.append([train_acc,test_acc])

    
plt.plot(np.arange(3,50,2),[i[0]for i in acc],"ro-")

plt.plot(np.arange(3,50,2),[i[1]for i in acc],"bo-")

knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)

pred=knn.predict(x_test)
pred

accuracy_score(y_test,pred)


    



