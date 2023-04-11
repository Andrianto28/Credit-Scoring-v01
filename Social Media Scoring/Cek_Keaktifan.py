#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import for read the data
import pandas as pd

#import for model and data test
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#evaluation model
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report    
from sklearn.model_selection import StratifiedKFold


# In[2]:


#read the data
data_frame = pd.read_csv('Data_Profil_IG.csv')
data_frame


# In[3]:


#data feature filters
data = data_frame[['id_user','username', 'follower_count', 'media_count', 'like_count(last10post)', 'comment_count(last10post)', 'last_post_interval(days)', 'last_story_interval(hours)', 'Keaktifan']]
data


# In[4]:


#cek NaN
data.isnull().sum()


# In[5]:


#replace with 0 for NaN
data['last_post_interval(days)'] = data['last_post_interval(days)'].replace(np.nan, 0)
data['last_story_interval(hours)'] = data['last_story_interval(hours)'].replace(np.nan, 0)


# In[6]:


#cek again
data.isnull().sum()


# In[7]:


#check label distribution
data['Keaktifan'].value_counts()


# In[8]:


#separate labels and data features
X = data[['follower_count', 'media_count', 'like_count(last10post)', 'comment_count(last10post)', 'last_post_interval(days)', 'last_story_interval(hours)']]
y = data['Keaktifan']


# In[9]:


#Handling imbalance data
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=4).fit_resample(X, y)


# In[10]:


#check label distribution (again)
y_resampled.value_counts()


# In[12]:


# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# #### Lore

# In[29]:


# Create logistic regression model
model_lore = LogisticRegression()

# Train model on training set
model_lore.fit(X_train, y_train)

# Make predictions on testing set
y_pred = model_lore.predict(X_test)

# Calculate accuracy
accuracy = model_lore.score(X_test, y_test)
print('Accuracy:', accuracy)


# #### Evaluasi

# In[30]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(model_lore, X_resampled, y_resampled, cv=5)
scores


# In[31]:


print(classification_report(y_test, y_pred, zero_division=0))
print(f'confusion matrix:\n {confusion_matrix(y_test,  y_pred)}')


# #### Rafe

# In[13]:


from sklearn.ensemble import RandomForestClassifier

# Create random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model on training set
model.fit(X_train, y_train)

# Make predictions on testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)


# In[14]:


print(X_test)


# #### Evaluasi

# In[15]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
scores


# In[16]:


print(classification_report(y_test, y_pred, zero_division=0))
print(f'confusion matrix:\n {confusion_matrix(y_test,  y_pred)}')


# In[17]:


d = {'id_user': [24218829, 24239210, 21346929], 'username': ['ilexuz', 'chana', 'AriaSalt'], 'follower_count': [200,300, 800], 'media_count': [10,20, 0], 'like_count(last10post)': [80, 200, 0], 'comment_count(last10post)': [10, 90, 0], 'last_post_interval(days)': [20, 10, 0], 'last_story_interval(hours)': [0,2,1],}
df = pd.DataFrame(data=d)
df


# In[18]:


u = 24218829

index = np.where(df['id_user'] == u)[0][0]
user_key = df.iloc[[index]]
user_key


# #### Use pickle

# In[25]:


import pickle

output = 'Model_Cek_keaktifanIG_Rafo'
with open(output, 'wb') as f:
    pickle.dump(model, f)


# In[26]:


#cara make model 
#import pickle #(jika di file line)

with open(output, 'rb') as f:
    obj_rafo = pickle.load(f)


# In[27]:


user = []
id_user = []
score = []

for users in range(df.shape[0]):
    user_key = df.iloc[[users]]
    x_coba= user_key[['follower_count', 'media_count', 'like_count(last10post)', 'comment_count(last10post)', 'last_post_interval(days)', 'last_story_interval(hours)']]
    output = obj_rafo.predict(x_coba)
    user.append(df['username'].iloc[users])
    id_user.append(df['id_user'].iloc[users])
    score.append(output)


# In[28]:


hasil = pd.DataFrame({'id_users':id_user, 'username':user,'scoring':score})
hasil


# #### Menggunakan Joblib

# In[29]:


import joblib as job

job.dump(model, 'Rafo_AktifIG_cek')


# In[30]:


Rafo_test= job.load('./Rafo_AktifIG_cek')


# In[ ]:




