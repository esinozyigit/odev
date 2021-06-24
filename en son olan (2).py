


import csv
import datetime
import numpy as np

import pandas as pd
import pandas_datareader.data as web

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

#benimmmmmm
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV,Lasso,Ridge,LassoCV,ElasticNet,ElasticNetCV
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR,SVC
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Get the stock data using yahoo API:
style.use('ggplot')

# get 2014-2018 data to train our model
start = datetime.datetime(2017,5,28)
end = datetime.datetime(2021,5,28)
df = web.DataReader("TSLA", 'yahoo', start, end) 
df

#hisse_veri1 = pd.read_csv("data.csv",sep=",")

df.isnull().sum()

df["Volume"][:2020].plot(figsize=(16,4),legend=True)
df["Volume"][:2021].plot(figsize=(16,4),legend=True)
plt.legend(["egitim 2020","Test 2020"])
plt.title("tesla")
plt.show()


print(df.describe())




df.head(10).plot(kind="bar",figsize=(16,8))

plt.grid(which="major",linestyle="-",linewidth=0.5,color="green")

plt.grid(which="minor",linestyle=":",linewidth=0.5,color="black")

plt.show()

df.plot()
plt.show()


df['Volume'].plot()
plt.show()

df['100ma'] = df['Volume'].rolling(window=100).mean()
print(df.head())

df['100ma'] = df['Volume'].rolling(window=100,min_periods=0).mean()
print(df.head())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()


# get 2019 data to test our model on 
#start = datetime.datetime(2019,1,1)
#end = datetime.date.today()
#test_df = web.DataReader("TSLA", 'yahoo', start, end) 



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Get the stock data using yahoo API:
style.use('ggplot')

# get 2014-2018 data to train our model
start = datetime.datetime(2017,5,28)
end = datetime.datetime(2021,5,28)
df = web.DataReader("TSLA", 'yahoo', start, end) 
df

"""

model ile tahmin yapma v1


"""

from sklearn.svm import SVR


df = df.dropna()
df

#VERİ AZALTMA

#df=df[0:200]
#df=df.drop(['SampleNo'], axis=1)



#df=df[0:1000]


# sort by date
df = df.sort_values('Date')

# fix the date 
df.reset_index(inplace=True)
uzunluk=len(df["Date"])

for tarih_icin in range(0,uzunluk):
    try:
            
        tarih=[]
        kkk=df["Date"][tarih_icin]
        kkk=str(kkk)
        kkk=kkk.split("-")
        yil=kkk[0]
        ay=kkk[1]
        gun=kkk[2]
        gun=gun.split(" ")
        gun=gun[0]
        tarih.append(yil)
        tarih.append(ay)
        tarih.append(gun)
        
        def listToString(s):    
            str1 = ""   
            for ele in s: 
                str1 += ele      
            return str1 
        
        tarih=(listToString(tarih)) 
        df["Date"][tarih_icin]=tarih
    except:
        pass

y = df["Volume"]
X = df.drop(["Volume"], axis = 1)





X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=42)



#loj model
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics

#linear

model = LinearRegression()

model.fit(X_train,y_train)

y_pred_linear=model.predict(X_test)



print(model.score(X_test,y_test))
prediction = model.predict(X_test)
prediction[0:10]


#y_test orijinal value değerleri 

#linear modele göre yazdır

#linear

model = LinearRegression()

model.fit(X_train,y_train)

y_pred_linear=model.predict(X_test)



print("linear model için doğruluk oranı",model.score(X_test,y_test))
prediction = model.predict(X_test)
prediction[0:10]

X_test.reset_index(inplace=True)
uzunluk=len(X_test)
for test_icin in range(0,uzunluk):
    try:
        
        X_test["index"][test_icin]=prediction[test_icin]
    except:
        pass



#ikinci index gerçek değerler
y_test_degerler=y_test.values
X_test.reset_index(inplace=True)
uzunluk=len(X_test)
for test_icin in range(0,uzunluk):
    try:
        
        X_test["level_0"][test_icin]=y_test_degerler[test_icin]
    except:
        pass

   
X_test
X_test=X_test.rename(columns={'index': 'Tahmin'})
X_test=X_test.rename(columns={'level_0': 'Gerçek sonuç'})
X_test

y_test_degerler=y_test.values


sum_column = ((X_test["Gerçek sonuç"] - X_test["Tahmin"]) /X_test["Gerçek sonuç"])*100
X_test["değişim oranı"] = sum_column

print(X_test)
print(X_test[0:20])
print("sonuçlar bu şekildedir")



"""

yapılacak şey şu tahmin ve bulunannları yazdır.

"""
print("svm için olan kısım")

y = df["Volume"]
X = df.drop(["Volume"], axis = 1)





X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=42)

#svm

svm_tuned = SVC(C = 8, kernel = "poly").fit(X_train, y_train)

y_pred = svm_tuned.predict(X_test)
y_pred[0:10]
(X_test.index.values)
list(X_test.index.values.tolist())


#svm için olan kısım
#ilk index tahminler
X_test.reset_index(inplace=True)
uzunluk=len(X_test)
for test_icin in range(0,uzunluk):
    try:
        
        X_test["index"][test_icin]=y_pred[test_icin]
    except:
        pass



#ikinci index gerçek değerler
y_test_degerler=y_test.values
X_test.reset_index(inplace=True)
uzunluk=len(X_test)
for test_icin in range(0,uzunluk):
    try:
        
        X_test["level_0"][test_icin]=y_test_degerler[test_icin]
    except:
        pass
    

X_test=X_test.rename(columns={'index': 'Tahmin'})
X_test=X_test.rename(columns={'level_0': 'Gerçek sonuç'})


y_test_degerler=y_test.values


sum_column = ((X_test["Gerçek sonuç"] - X_test["Tahmin"]) /X_test["Gerçek sonuç"])*100
X_test["değişim oranı"] = sum_column

print(X_test)
print(X_test[0:20])
print("sonuçlar bu şekildedir")




