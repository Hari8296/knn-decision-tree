Name :- Hari singh r 
Batch id :-  DSWDMCOD 25082022 B


1
Business problems 
1.1 what is the Business odjectives 
Ans:-To classify the animals in Zoo based on their physical appearance & features.

1.2 What are the constraints      
Ans:-There can be more features & factors to classify the animals.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

zoo=pd.read_csv("D:/assignments of data science/12 knn & decision tree/Zoo.csv")
zoo

zoo.info()
zoo.describe()
zoo.head()
zoo.duplicated().sum()
zoo.isna().sum()

zoo.mean()
zoo.median()
list(zoo.mode())
zoo.skew()
zoo.kurt()
zoo.var()


for i in zoo.columns:
    plt.hist(zoo[i])
    plt.xlabel(i)
    plt.show()

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

zoo_n=norm_fun(zoo.iloc[:,1:])
zoo_n.describe()

x=np.array(zoo_n.iloc[:,:])
y=np.array(zoo['type'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)  

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=21)   
knn.fit(x_train,y_train)

pred=knn.predict(x_test)
pred

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred))
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

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

pred=knn.predict(x_test)
pred

accuracy_score(y_test,pred)
    






















