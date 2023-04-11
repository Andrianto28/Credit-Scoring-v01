# -*- coding: utf-8 -*-
"""Fraud Detection (Business User).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SgjjpufV8aImcstG36tPhkOJVz8uO0WP
"""

pip install scale

pip install faker

"""## 1. Load Data dan Packaging"""

import numpy as np
import pandas as pd
import sklearn
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import tensorflow
from tensorflow import keras
from faker import Faker
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
fake = Faker()

import pandas as pd
from faker import Faker 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import csv
fake = Faker()

df = pd.read_csv('/content/data_dummy.csv')

def create_business_data(n):
    business = []
    for i in range(n):
        data = {
            'business_size': fake.random_element(elements=('0', '1', '2', '3', '4', '5')),
            'industry': fake.random_element(elements=('0','1','2','3','4','5','6')),
            'order_history': fake.random_element(elements=('0','1','2')),
            'payment_terms': fake.random_element(elements=('0','1','2',)),
            'authorized_user': fake.random_element(elements=('0','1','2')),
            'communication_history': fake.random_element(elements=('0','1','2')),
            'transaction_type': fake.random_element(elements=('0','1')),
            'geolocation': fake.random_element(elements=('0','1','2','3','4','5','6','7','8','9')),
            'user_behavior_history': fake.random_element(elements=('0','1')),
            'session_duration': fake.random_element(elements=('0','1','2','3')),
            'time_of_day': fake.random_element(elements=('0','1','2')),
            'payment_method': fake.random_element(elements=('0','1','2','3','4')),
            'velocity': fake.random_element(elements=('0','1','2')),
            'target' : fake.random_element(elements=('0','1'))
        }
        business.append(data)
    return business

df = pd.DataFrame(create_business_data(20000))
df.head(10)

df.info()

df.describe()

df.shape

"""## 2. Memahami Dataset

This dataset contains of 20000 rows and 14 columns with the variables below :
1. business_size = untuk mengidentifikasi kelas pada bisnis.
2. industry = industry merupakan pernyataan suatu perusahaan yang bergerak di bidang masing-masing.
3. order_history = untuk order history ini untuk menyatakan riwayat pembelian atau transaksi sebelumnya dari pelanggan. Dan dengan variabel yang digunakan ini adalah untuk menentukkan kemampuan para pebisnis dalam memenuhi kewajiban pembayaran.
4. payment terms = sebuah metode pembayaran yang harus dilakukan oleh user dalam suatu transaksi bisnis.
5. authorized_user = dalam suatu perusahaan, authorized user ini merupakan orang yang diberikan izin untuk mengurus permasalahan keuangan di suatu perusahaan.
6. communication_history = catatan atau log interaksi komunikasi antara dua belah pihak, baik secara pesan teks, email, panggilan telepon ataupun komunikasi lainnya.
7. transaction type = merupakan kategori atau jenis dari sebuah transaksi.
8. geolocation = geolocation sendiri merupakan sebuah lokasi geografis atau informasi mengenai posisi user pada saat proses transaksi.
9. user_behavior_history = menjelaskan mengenai patuh atau tidaknya user dalam proses pembayarannya.
10. session_duration = durasi dari lamanya transaksi.
11. time_of_day = waktu yang menunjukkan saat proses transaksi user.
12. payment_method = merupakan sebuah metode pembayaran yang dilakukan oleh user.
13. velocity = mengukur waktu kecepatan proses transaksi pembayaran dan pelaporan.
14. target = untuk menentukan apakah data fraud atau tidak fraud.

Pada dataset ini dijelaskan bahwa order_history merupakan menyatakan riwayat pembelian atau transaksi sebelumnya dari pelanggan. Namun untuk variabel payment_method merupakan tahapan dalam proses pembayaran baik menggunakan kartu atau tunai. Sehingga disini kami memutuskan untuk membuat variable total_history untuk menggabungkan order_history dengan payment_method untuk melihat antara riwayat pemesanan dan metode pembayaran yang digunakan oleh user dalam proses transaksi.
"""

df['total_history'] = df['order_history'] + df['payment_method']

df.shape

df.head()

df.nunique()

"""## 3. Pengolahan Data (Missing Value, Outlier, Duplicated Data)"""

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.shape

df['business_size'].value_counts()

df.isnull().sum()

#Menghapus missing value
df.dropna(inplace=True)

#Memeriksa kembali missing value
df.isnull().sum()

df.shape

#Memeriksa proporsi outliers
fig = plt.figure(figsize=(20, 15))
fig.subplots_adjust(hspace=.5, wspace=.5)

ax = fig.add_subplot(3,6,1)
sns.boxplot(x=df['business_size'])

ax = fig.add_subplot(3,6,2)
sns.boxplot(x=df['industry'])

ax = fig.add_subplot(3,6,3)
sns.boxplot(x=df['order_history'])

ax = fig.add_subplot(3,6,4)
sns.boxplot(x=df['payment_terms'])

ax = fig.add_subplot(3,6,5)
sns.boxplot(x=df['authorized_user'])

ax = fig.add_subplot(3,6,6)
sns.boxplot(x=df['communication_history'])

ax = fig.add_subplot(3,6,7)
sns.boxplot(x=df['transaction_type'])

ax = fig.add_subplot(3,6,8)
sns.boxplot(x=df['geolocation'])

ax = fig.add_subplot(3,6,9)
sns.boxplot(x=df['PAYMENTS'])

ax = fig.add_subplot(3,6,10)
sns.boxplot(x=df['user_behavior_history'])

ax = fig.add_subplot(3,6,11)
sns.boxplot(x=df['session_duration'])

ax = fig.add_subplot(3,6,12)
sns.boxplot(x=df['time_of_day'])

ax = fig.add_subplot(3,6,13)
sns.boxplot(x=df['payment_method'])

ax = fig.add_subplot(3,6,14)
sns.boxplot(x=df['velocity'])

ax = fig.add_subplot(3,6,15)
sns.boxplot(x=df['target'])

plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
print(Q1)
print(Q3)

IQR = Q3-Q1
print(IQR)

len((df < (Q1-1.5*IQR)) | (df > (Q3+1.5*IQR)))

"""## 4. Transformasi Data"""

df_vis = df.copy()

#transform data
df_vis['business_size'] = df_vis['business_size'].replace({0: 'Small', 1: 'Micro', 2: 'Super Micro', 3: 'Medium', 4: 'Big', 5: 'Super Big'})
df_vis['industry'] = df_vis['industry'].replace({0: 'Agriculture', 1: 'Trading', 2:'Service', 3: 'Manufacture', 4: 'Construction', 5: 'Finance', 6: 'Transportation', 7: 'Grosir Sembako', 8: 'Pertanian Palawijaya Sayur', 9: 'Pemborong Rumah', 10: 'Rental Mobil', 11: 'Service AC'})
df_vis['order_history'] = df_vis['order_history'].replace({0: 'Excellent', 1: 'Good', 2: 'Bad'})
df_vis['payment_terms'] = df_vis['payment_terms'].replace({0: 'Advance payment', 1: 'Installment payments', 2: 'Immediate payment'})
df_vis['authorized_user'] = df_vis['authorized_user'].replace({0: 'President director', 1: 'Director of finance', 2: 'General manager'}) 
df_vis['communication_history'] = df_vis['communication_history'].replace({0: 'Never communicated', 1: 'Have communicated via e-mail', 2:'Have communicated via telephone'})
df_vis['transaction_type'] = df_vis['transaction_type'].replace({0: 'Cash', 1: 'Non cash'})
df_vis['geolocation'] = df_vis['geolocation'].replace({0: 'Jakarta', 1: 'Surabaya', 2: 'Bandung', 3: 'Medan', 4: 'Makassar', 5: 'Kalimantan', 6: 'Bali', 7: 'Palembang', 8: 'Semarang', 9: 'Yogyakarta'})
df_vis['user_behavior_history'] = df_vis['user_behavior_history'].replace({0: 'Obedient in paying', 1: 'Disobedient in paying'})
df_vis['session_duration'] = df_vis['session_duration'].replace({0: '<5 hour', 1: '5-10 hour', 2: '10-15 hour', 3: '>15 hour'})
df_vis['time_of_day'] = df_vis['time_of_day'].replace({0: 'Morning', 1: 'Evening', 2: 'Night'})
df_vis['payment_method'] = df_vis['payment_method'].replace({0: 'Cash', 1: 'Debit card', 2: 'Credit card', 3: 'Bank transfers', 4: 'Mobile banking'})
df_vis['velocity'] = df_vis['velocity'].replace({0: 'high', 1: 'medium', 2: 'low'})
df_vis['target'] = df_vis['target'].replace({0: 'No Fraud', 1: 'Is Fraud'})

unique_values = {}
for col in df_vis.columns:
    unique_values[col] = df_vis[col].value_counts().shape[0]

pd.DataFrame(unique_values, index=['unique value count']).transpose()

df_vis['target'].value_counts()

df_vis.groupby('order_history')[['payment_method']].count().sort_values(by='payment_method')

df_vis.groupby('order_history')[['payment_method']].agg(['count','median','min','max'])

df_vis.groupby('authorized_user')[['payment_method']].count().sort_values(by='payment_method')

df_vis.groupby('authorized_user')[['payment_method']].agg(['count','median','min','max'])

df_vis.groupby('velocity')[['session_duration']].count().sort_values(by='session_duration')

df_vis.groupby('velocity')[['session_duration']].agg(['count','median','min','max'])

df_vis.groupby('velocity')[['payment_method']].count().sort_values(by='payment_method')

df_vis.groupby('velocity')[['payment_method']].agg(['count','median','min','max'])

df['business_size'] = df['business_size'].astype(int)
df['industry'] = df['industry'].astype(int)
df['order_history'] = df['order_history'].astype(int)
df['payment_terms'] = df['payment_terms'].astype(int)
df['authorized_user'] = df['authorized_user'].astype(int)
df['payment_terms'] = df['payment_terms'].astype(int)
df['communication_history'] = df['communication_history'].astype(int)
df['transaction_type'] = df['transaction_type'].astype(int)
df['geolocation'] = df['geolocation'].astype(int)
df['user_behavior_history'] = df['user_behavior_history'].astype(int)
df['session_duration'] = df['session_duration'].astype(int)
df['time_of_day'] = df['time_of_day'].astype(int)
df['payment_method'] = df['payment_method'].astype(int)
df['velocity'] = df['velocity'].astype(int)
df['target'] = df['target'].astype(int)

import matplotlib.pyplot as plt

df.plot(kind="box", subplots=True, layout=(7,4), figsize=(15,14));

df2 = pd.DataFrame({'order_history': df['order_history'], 'authorized_user': df['authorized_user'], 'payment_method': df['payment_method'], 'velocity': df['velocity'], 'session_duration': df['session_duration'], 'geolocation': df['geolocation'], 'transaction_type': df['transaction_type'], 'user_behavior_history': df['user_behavior_history'], 'target': df['target'], 'industry' : df['industry'], 'business_size' : df['business_size'], 'time_of_day' : df['time_of_day'], 'communication_history' : df['communication_history']})
df2

df3=df2.copy()

from sklearn.ensemble import IsolationForest
model = IsolationForest()
model.fit(df3)
#df['anomailes_scores']=model.decision_function(df)
df3['anomaly']= model.predict(df3)
df3

df3[df3['anomaly']==-1]

df3[df3['anomaly']==-1].shape

df3.drop(df3[df3['anomaly']==-1].index,inplace = True)

df3.shape

df3

"""## Statistika Deskriptif"""

#Distribusi Uniform
import statistics
import numpy as np

mean1 = df2['target'].mean()
mean1
round(mean1,1)

std = statistics.stdev(df2['target'])
std
round(std,3)

# Menghitung 2 * rata-rata
a_plus_b = 2*mean1
a_plus_b 
round(a_plus_b,2)

# Menghitung Akar Kuadrat Dari STD
b_min_a = std*np.sqrt(12)
b_min_a
round(b_min_a,3)

# nilai a
a = (a_plus_b - b_min_a)/2
a

# b
b = (a_plus_b + b_min_a)/2
b

def dis_uniform(batas,a,b):
    atas = batas - a
    bawah = b - a
    rumus = atas/bawah
    return rumus

dis_uniform(5,a,b)

import matplotlib.pyplot as plt

plt.hist(df['business_size'])

# Adding Title to the Plot
plt.title("Distribusi Uniform")
 
# Setting the X and Y labels

plt.show()

"""## Distribusi Normal"""

df2 = df.copy()

def dis_normal(data, miu, gamma):
    phi = 3.141592
    e = 2.7183
    atas = 1*e**(-0.5*((data-miu)/gamma)**2)
    bawah = gamma*np.sqrt(2*phi)
    rumus = atas/bawah
    return rumus

df2['f(x)'] = dis_normal(df['business_size'],0.2,0.528)
df2

import matplotlib.pyplot as plt

plt.plot(df2['business_size'],df2['f(x)'])

"""## 7. Distribusi Skewness"""

def skewness(data):
    mean = data.mean()
    median = data.median()
    atas = 3*(mean-median)
    bawah = data.std()
    rumus = atas/bawah
    return rumus

skewness(df2['business_size'])

import matplotlib.pyplot as plt

plt.plot(df2['business_size'],df2['f(x)'])

import matplotlib.pyplot as plt

plt.plot(df2['business_size'],df2['f(x)'])

#Uji normalitas data
import pandas as pd
from scipy.stats import kstest, shapiro

# dan Uji Shapiro-Wilk (shapiro)

swdata = shapiro(df2['business_size'])
print(swdata)

# Hipotesis dengan T-test dan P-value

from scipy.stats import ttest_1samp
import numpy as np


average = np.mean(df2["target"])
print("Average data is = {0:.3f}".format(average))

tset,pval = ttest_1samp(df2["business_size"], 0.180)

print("P-value = {}".format(pval))

if pval < 0.05:
    print("We are rejecting the null Hypotheis.")
else:
    print("We are accepting the null hypothesis")

"""## 8. Uji Korelasi"""

correlation = df2.corr(method='pearson')
correlation

import seaborn as sns
sns.heatmap(correlation,xticklabels=correlation.columns,
            yticklabels=correlation.columns)

"""## 9. Pemodelan Data"""

# Decision Tree
from sklearn.model_selection import train_test_split

y = df2['business_size']
x = df2.drop(['business_size'], axis=1)
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.35, random_state=44, shuffle =True)

from sklearn.tree import DecisionTreeClassifier

dt= DecisionTreeClassifier(max_features=12, max_depth=15)
dt.fit(x_train , y_train)

from sklearn.metrics import accuracy_score

y_pred_train_dt = dt.predict(x_train)
acc_train_dt = accuracy_score(y_train, y_pred_train_dt)

y_pred_test_dt = dt.predict(x_test)
acc_test_dt = accuracy_score(y_test, y_pred_test_dt)
print("Hasil Akurasi dari pengujian data training adalah",acc_train_dt)
print("Hasil Akurasi dari pengujian data test adalah",acc_test_dt)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test_dt))

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(penalty='l2',solver='sag',C=1.0,random_state=100)
lg.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

y_pred_train_lg = lg.predict(x_train)
acc_train_lg = accuracy_score(y_train, y_pred_train_lg)

y_pred_test_lg = lg.predict(x_test)
acc_test_lg = accuracy_score(y_test, y_pred_test_lg)
print("Hasil Akurasi dari pengujian data training adalah",acc_train_lg)
print("Hasil Akurasi dari pengujian data test adalah",acc_test_lg)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test_lg))

from sklearn.metrics import f1_score,  recall_score, precision_score, plot_roc_curve, roc_curve, roc_auc_score

print('Precision: %.3f' % precision_score(y_test, y_pred_test_lg,average="micro"))
print('Recall: %.3f' % recall_score(y_test, y_pred_test_lg,average="micro"))
print('F-measure: %.3f' % f1_score(y_test, y_pred_test_lg,average="micro"))

# Scoring the Model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

# R2 Score
print(f"R2 score: {r2_score(y_test, y_pred_test_lg)}")

# Mean Absolute Error (MAE)
print(f"MSE score: {mean_absolute_error(y_test, y_pred_test_lg)}")

# Mean Squared Error (MSE)
print(f"MSE score: {mean_squared_error(y_test, y_pred_test_lg)}")

"""## Visualisasi Interaktif"""

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='order_history')
plt.xticks(rotation=90)
plt.title('Target dari Order History',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='payment_method')
plt.xticks(rotation=90)
plt.title('Target dari Payment Method',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='communication_history')
plt.xticks(rotation=90)
plt.title('Target dari Communication History',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='geolocation')
plt.xticks(rotation=90)
plt.title('Target by Geolocation',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='business_size')
plt.xticks(rotation=90)
plt.title('Target dari Business Size',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='payment_terms')
plt.xticks(rotation=90)
plt.title('Target dari Payment Terms',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='industry')
plt.xticks(rotation=90)
plt.title('Target dari Industry',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='user_behavior_history')
plt.xticks(rotation=90)
plt.title('Target dari User Behavior History',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='velocity')
plt.xticks(rotation=90)
plt.title('Target dari Velocity',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='authorized_user')
plt.xticks(rotation=90)
plt.title('Target dari Authorized User',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='session_duration')
plt.xticks(rotation=90)
plt.title('Target dari Session Duration',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='time_of_day')
plt.xticks(rotation=90)
plt.title('Target dari Time of Day',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Visualisasi Interaktif
plt.figure(figsize=(12,4))
x= sns.countplot(x='target',data=df_vis,hue='transaction_type')
plt.xticks(rotation=90)
plt.title('Target dari Transaction Type',fontdict={'fontsize':20})
for i in x.patches:
    x.annotate('{:.2f}'.format((i.get_height()/df_vis.shape[0])*100)+'%',(i.get_x()+0.25, i.get_height()+0.01))
plt.show()

#Relationship Visualization
df_vis.plot(kind='scatter',x='target', y='payment_method')

#Relationship Visualization
df_vis.plot(kind='scatter',x='order_history', y='payment_method')

df_vis.plot(kind='scatter',x='authorized_user', y='payment_method')

"""## 11. PCA Method"""

from sklearn.model_selection import train_test_split

y = df2['business_size']
x = df2.drop(['business_size'], axis=1)
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.95, random_state=44, shuffle =True)

# performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Applying PCA function on training
# and testing set of x component
from sklearn.decomposition import PCA

pca = PCA(n_components = 0.95)

x_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_
n_components = pca.n_components_

explained_variance, n_components

hasil_pca = pd.DataFrame(pca_test)
hasil_pca

"""#### Hasil PCA adalah memakai 11 principal components"""

from sklearn.decomposition import PCA
pca = PCA()
x_train = pca.fit_transform(x_train)
pca.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def fit_evaluate(x_train, y_train, x_test, y_test, return_pred = False):
    model = LogisticRegression()
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()

    duration = end-start
    accuracy = accuracy_score(y_test, model.predict(x_test))

    if return_pred == False:
        return duration, accuracy
    else:
        return duration, accuracy, model.predict(x_test)

duration_raw, accuracy_raw = fit_evaluate(x_train, y_train, x_test, y_test)

print('Business Size :', duration_raw, 'seconds.')
print('Model accuracy :', accuracy_raw, 'percent.')

from sklearn.decomposition import PCA
pca_test = PCA(0.95)

start_time = time.time()
pca_test.fit(x_train)
x_train_pca_test = pca_test.transform(x_train)
x_test_pca_test = pca_test.transform(x_test)
finish_time = time.time()

print('PCA Fit and Transform finished in', finish_time - start_time, 'seconds.')
print('Hanya tersisa:', pca_test.n_components_, 'feature columns setelah dlakukan PCA.')
print('PCA kali ini berhasil mereduksi dataset sebanyak', (1-(pca_test.n_components_/784))*100, 'persen')

pca_test.n_components_

plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.show()

duration_pca, accuracy_pca = fit_evaluate(x_train_pca_test, y_train, x_test_pca_test, y_test)

print('Business Size :', duration_pca, 'seconds.')
print('Model accuracy :', accuracy_pca, 'percent.')

x_train_pca_test.shape

df.columns

df = pd.read_csv('D:/cooliyeah/semester 6/msib/faker/data/data_dummyy.csv')
df

import pandas as pd

# Tentukan variabel mana yang merupakan features dan target
features = ['order_history', 'payment_method', 'business_size', 'industry', 'authorized_user', 'communication_history', 'transaction_type', 'geolocation', 'user_behavior_history', 'session_duration', 'time_of_day', 'velocity', 'payment_terms']
target = ['target']

# Pisahkan features dan target dari dataset
x = df[features]
y = df[target]
print(x, y)

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# Mendefinisikan features dan target
x = df['order_history'].values.reshape(-1,1)
y = df['target'].values.reshape(-1,1)

# Membuat objek model Linear Regression
model = LinearRegression()

# Mendefinisikan jumlah fold untuk cross validation
n_folds = 5

# Membuat objek k-fold cross validation
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Melakukan cross validation dengan scoring=r2 dan menampilkan hasil
scores = cross_val_score(model, x, y, scoring='r2', cv=kfold)
print("R-squared scores: ", scores)
print("R-squared mean: ", np.mean(scores))
print("R-squared std dev: ", np.std(scores))

"""## 12. K-Means and Hierarchical Clustering"""

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('D:/cooliyeah/semester 6/msib/faker/data/data_dummyy.csv')
df

# Visualizing the data
x1 = df['order_history']
x2 = df['payment_method']

plt.plot()
plt.xlim([0, 15])
plt.ylim([0, 15])
plt.title('Data Dummy')
plt.scatter(x1, x2)
plt.show()

sns.set(style='whitegrid')
sns.displot(data=df2, x='payment_method', kde=True)
plt.title('Distribution of Payment Method', fontsize=20)
plt.xlabel('Range of Payment Method')
plt.ylabel('Count')
plt.show()

df_vis['payment_method'].value_counts()

sns.pairplot(df2)

plt.matshow(df2.corr())

import pandas as pd
import matplotlib.pyplot as plt

pd.plotting.scatter_matrix(df2, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
plt.show()

fig, axis = plt.subplots(figsize=(10, 8))
corr = df2.corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), cmap = sns.diverging_palette(220, 10, as_cmap = True),
            square = True, ax = axis)

x = df2.iloc[:,0:].values

print(x.shape)

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(df)
    kmeanModel.fit(df)
    
    distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0]
    mapping2[k] = kmeanModel.inertia_

for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

for key, val in mapping2.items():
    print(f'{key} : {val}')

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'blue', label = 'No Fraud')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'Is Fraud')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

plt.title('K Means Clustering')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=100, c='blue', label='No Fraud')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=100, c='green', label='Is Fraud')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c='yellow', label='centroid')

plt.title('K Means Clustering')
plt.legend()

# set range of y-axis
plt.ylim(bottom=-5, top=5)

plt.show()

#if the scale of the variables is not the same, the model might become biased towards the variables with a higher magnitude like Fresh or Milk

from sklearn.preprocessing import normalize
data_scaled = normalize(df)
data_scaled = pd.DataFrame(data_scaled, columns=df.columns)
data_scaled.head()

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogam')
plt.xlabel('User')
plt.ylabel('Ecuclidean Distance')
plt.show()

dendrogram = sch.dendrogram(sch.linkage(x, method='ward'), orientation='left')
plt.title('Dendrogram')
plt.xlabel('User')
plt.ylabel('Euclidean Distance')
plt.show()

"""## 13. Faktor Analisis"""

pip install factor factor analyzer



"""## 14. K-Means Evaluation Method"""



"""## 15. Deployment Method"""

import pickle



"""### BERHENTI SAMPAI DISINI"""

from sklearn.decomposition import PCA
pca = PCA()
x_train = pca.fit_transform(x_train)
pca.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def fit_evaluate(x_train, y_train, x_test, y_test, return_pred = False):
    model = LogisticRegression()
    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()

    duration = end-start
    accuracy = accuracy_score(y_test, model.predict(x_test))

    if return_pred == False:
        return duration, accuracy
    else:
        return duration, accuracy, model.predict(x_test)

duration_raw, accuracy_raw = fit_evaluate(x_train, y_train, x_test, y_test)

print('Business Size :', duration_raw, 'seconds.')
print('Model accuracy :', accuracy_raw, 'percent.')

from sklearn.decomposition import PCA
pca_test = PCA(0.95)

start_time = time.time()
pca_test.fit(x_train)
x_train_pca_test = pca_test.transform(x_train)
x_test_pca_test = pca_test.transform(x_test)
finish_time = time.time()

print('PCA Fit and Transform finished in', finish_time - start_time, 'seconds.')
print('Hanya tersisa:', pca_test.n_components_, 'feature columns setelah dlakukan PCA.')
print('PCA kali ini berhasil mereduksi dataset sebanyak', (1-(pca_test.n_components_/784))*100, 'persen')

pca_test.n_components_

plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.show()

duration_pca, accuracy_pca = fit_evaluate(x_train_pca_test, y_train, x_test_pca_test, y_test)

print('Business Size :', duration_pca, 'seconds.')
print('Model accuracy :', accuracy_pca, 'percent.')

x_train_pca_test.shape

df.columns

"""## Faktor Analisis"""

pip install factor factor analyzer

df = pd.read_csv('C:/Users/Lenovo/Downloads/data_dummy.csv')
df.head()

df.columns

import pandas as pd

# Baca dataset
df = pd.read_csv('C:/Users/Lenovo/Downloads/data_dummy.csv')

# Tentukan variabel mana yang merupakan features dan target
features = ['order_history', 'payment_method', 'business_size', 'industry', 'authorized_user', 'communication_history', 'transaction_type', 'geolocation', 'user_behavior_history', 'session_duration', 'time_of_day', 'velocity', 'payment_terms']
target = ['target']

# Pisahkan features dan target dari dataset
x = df[features]
y = df[target]
print(x, y)

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# Mendefinisikan features dan target
x = df['order_history'].values.reshape(-1,1)
y = df['target'].values.reshape(-1,1)

# Membuat objek model Linear Regression
model = LinearRegression()

# Mendefinisikan jumlah fold untuk cross validation
n_folds = 3

# Membuat objek k-fold cross validation
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Melakukan cross validation dengan scoring=r2 dan menampilkan hasil
scores = cross_val_score(model, x, y, scoring='r2', cv=kfold)
print("R-squared scores: ", scores)
print("R-squared mean: ", np.mean(scores))
print("R-squared std dev: ", np.std(scores))

