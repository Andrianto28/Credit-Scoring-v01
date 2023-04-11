#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install np_utils')

import pandas as pd
import numpy as np
import re
import nltk
import string

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk import word_tokenize

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import OrderedDict

from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dropout


# In[2]:


df = pd.read_csv('Data_Posting_IG_Bersih.csv')
df


# In[3]:


df.dropna(axis=0, subset=['bersih1'], inplace=True)
df


# In[4]:


data = df[['bersih1', 'Label']]
data


# In[5]:


def example_data (index):
    exam = data[data.index == index][['bersih1', 'Label']].values[0]
    if len(exam) > 0:
        print(exam[0])
        print()
        print('Category:', exam[1])


# In[6]:


example_data(4)


# In[7]:


#check label distribution
data['Label'].value_counts()


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()


# In[9]:


from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences



MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

# Tokenize input texts
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(data['bersih1'].values)
word_index = tokenizer.word_index

# Convert input texts to sequences and pad them
X = tokenizer.texts_to_sequences(data['bersih1'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor : ', X.shape)


# In[10]:


X


# In[11]:


Y = pd.get_dummies(data['Label']).values
print('Shape of label tensors: ', Y.shape)


# In[12]:


#Handling imbalance data
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(k_neighbors=3, random_state=4).fit_resample(X, Y)


# In[13]:


len(y_resampled)


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.2, random_state=42)

print(x_train.shape, y_train.shape)
print(x_train.shape, y_train.shape)


# In[30]:


import tensorflow as tf

class EarlyStopping(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    '''
    Stops training when 95% accuracy is reached
    '''
    # Get the current accuracy and check if it is above 95% 
    #if(logs.get('accuracy') > 0.95):
    if(logs.get('val_accuracy') > 0.77):

      # Stop training if condition is met
      print("\nThreshold reached. Stopping training...")
      self.model.stop_training = True

early_stopping = EarlyStopping()


# In[31]:


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1])) 
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, input_shape=(X.shape[1], EMBEDDING_DIM))) 
model.add(Dense(3, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt , metrics=['accuracy'])

model.summary()


# In[32]:


'''import tensorflow as tf
tf.config.run_functions_eagerly(True)'''

#history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

history = model.fit(x_train, y_train, epochs=40, batch_size=64, validation_split=0.1)


# In[33]:


import matplotlib.pyplot as plt
from plotly.offline import iplot
import seaborn as sns


# In[34]:


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();


# In[35]:


plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();


# In[36]:


teks = ['Nikmat kopi disenja']
seq = tokenizer.texts_to_sequences(teks)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
label = ['Kehidupan Pribadi', 'Promosi', 'Sarkas/Sindiran']
print(pred, label[np.argmax(pred)])


# In[37]:


d = {'id_user': [24218829, 24239210, 21346929], 'username': ['ilexuz', 'chana', 'AriaSalt'], 'Caption':[ '[kembali] Rabu kemarin, 22 Februari 2023 saya berkesempatan tampil dalam acara Sastra Reboan. Saya menampilkan dramatisasi puisi berjudul Wisuda Putri karya A. Slamet Widodo. Puisi tersebut saya bacakan di depan penyairnya langsung dan puluhan sastrawan terkemuka. Acara ini selain diskusi sastra, ternyata juga peluncuran buku antologi puisi karya penyair yang sama. Puisi-puisinya jujur, Pak Slamet menggunakan puisi sebagai media bercerita. Mulai dari politik, isu pendidikan, hingga nikmatnya malam pertama (katanya).Momen ini mengingatkan saya pada 4 tahun lalu saat tampil di Teater Kecil Taman Ismail Marzuki. Saya kira, kesempatan besar itu hanya sekali dan terakhir tampil di panggung yang saya idamkan, panggung Ismail Marzuki. Alhamdulillah, menginjak tahun ketiga perkuliahan, saya kembali berperan di sana, dengan perasaan yang masih sama, beribu bahagia.','Pekerjaan di dunia data yang namanya mirip mirip emang suka bikin bingung!🥴😥', 'Love kamu sayang']}
df = pd.DataFrame(data=d)
df


# In[38]:


def clear(caption):
    #remove angka
    caption = re.sub('[0-9]+', '', str(caption))

    # remove hyperlinks
    caption = re.sub(r'https?:\/\/\S+', '', str(caption))
    
    # remove hashtags
    caption = re.sub(r'#[A-Za-z0-9_]+', '', str(caption))

    # remove symbol
    caption = re.sub(r'[^\x00-\x7f]', '', str(caption))

    #remove coma
    caption = re.sub(r',',' ', str(caption))
    
    #remove double huruf
    captions = ''.join(OrderedDict.fromkeys(caption))
    
    #remove punctuation
    caption = re.sub(r'[^\w\s]', '', str(caption))
    
    return caption


# In[39]:


df['bersih1'] = df['Caption'].apply(lambda x: clear(x))
df


# In[40]:


# oleh Tala, F.z (2003)
stopwords_buatan=['ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri', 'akhirnya',
     'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara', 'antaranya', 'apa', 'apaan', 'apabila',
     'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya',
     'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya',
     'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini', 'beginian', 'beginikah',
     'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun', 'bekerja', 'belakang', 'belakangan', 'belum', 'belumlah',
     'benar', 'benarkah', 'benarlah', 'berada', 'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah',
     'berapalah', 'berapapun', 'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya',
     'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu',
     'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam', 'bermaksud', 'bermula', 'bersama', 'bersama-sama',
     'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut', 'berturut-turut', 'bertutur', 'berujar', 'berupa',
     'besar', 'betul', 'betulkah', 'biasa', 'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah',
     'bolehlah', 'buat', 'bukan', 'bukankah', 'bukanlah', 'bukannya', 'bulan', 'bung', 'cara', 'caranya', 'cukup',
     'cukupkah', 'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang', 'dekat', 'demi',
     'demikian', 'demikianlah', 'dengan', 'dng', 'dg', 'depan', 'di', 'dia', 'diakhiri', 'diakhirinya', 'dialah', 'diantara',
     'diantaranya', 'diberi', 'diberikan', 'diberikannya', 'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan',
     'diibaratkan', 'diibaratkannya', 'diingat', 'diingatkan', 'diinginkan', 'dijawab', 'dijelaskan', 'dijelaskannya',
     'dikarenakan', 'dikatakan', 'dikatakannya', 'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan',
     'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan', 'dimaksudkannya', 'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan',
     'dimulai', 'dimulailah', 'dimulainya', 'dimungkinkan', 'dini', 'dipastikan', 'diperbuat', 'diperbuatnya',
     'dipergunakan', 'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 'dipersoalkan', 'dipertanyakan',
     'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan', 'disebutkannya', 'disini', 'disinilah',
     'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan', 'ditunjuk', 'ditunjuki',
     'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 'dituturkan', 'dituturkannya', 'diucapkan', 'diucapkannya',
     'diungkapkan', 'dong', 'dua', 'dulu', 'empat', 'enggak', 'enggaknya', 'entah', 'entahlah', 'guna', 'gunakan', 'hal',
     'hampir', 'hanya', 'hanyalah', 'hari', 'harus', 'haruslah', 'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga',
     'ia', 'ialah', 'ibarat', 'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin', 'inginkah',
     'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah', 'jadinya', 'jangan', 'jngn', 'jng', 'jgn', 'jangankan',
     'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas', 'jelaskan', 'jelaslah', 'jelasnya', 'jika', 'jikalau',
     'juga', 'jumlah', 'jumlahnya', 'justru', 'kala', 'kalau', 'kalaulah', 'kalaupun', 'kalian', 'kami', 'kamilah', 'kamu',
     'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 'karena', 'krn', 'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah',
     'katanya', 'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan', 'kelihatan',
     'kelihatannya', 'kelima', 'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya', 'kenapa', 'kepada',
     'kepadanya', 'kesampaian', 'keseluruhan', 'keseluruhannya', 'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah',
     'kira', 'kira-kira', 'kiranya', 'kita', 'kitalah', 'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain', 'lainnya', 'lalu',
     'lama', 'lamanya', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lima', 'luar', 'macam', 'maka', 'makanya', 'makin',
     'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi', 'masa', 'masalah', 'masalahnya', 'masih',
     'masihkah', 'masing', 'masing-masing', 'mau', 'maupun', 'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya',
     'memang', 'memastikan', 'memberi', 'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan',
     'memisalkan', 'memperbuat', 'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan',
     'mempertanyakan', 'mempunyai', 'memulai', 'memungkinkan', 'menaiki', 'menambahkan', 'menandaskan', 'menanti',
     'menanti-nanti', 'menantikan', 'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan', 'mendatang',
     'mendatangi', 'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya', 'mengenai',
     'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan', 'mengibaratkannya', 'mengingat',
     'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya', 'mengungkapkan', 'menjadi', 'menjawab',
     'menjelaskan', 'menuju', 'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya', 'menurut', 'menuturkan',
     'menyampaikan', 'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh', 'menyiapkan', 'merasa', 'mereka', 'merekalah',
     'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan', 'minta', 'mirip', 'misal', 'misalkan', 'misalnya', 'mula',
     'mulai', 'mulailah', 'mulanya', 'mungkin', 'mungkinkah', 'nah', 'naik', 'namun', 'nanti', 'nantinya', 'nyaris',
     'nyatanya', 'oleh', 'olehnya', 'pada', 'padahal', 'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'pasti',
     'pastilah', 'penting', 'pentingnya', 'per', 'percuma', 'perlu', 'perlukah', 'perlunya', 'pernah', 'persoalan',
     'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan', 'pihak', 'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa',
     'rasanya', 'rata', 'rupanya', 'saat', 'saatnya', 'saja', 'sajalah', 'saling', 'sama', 'sama-sama', 'sambil', 'sampai',
     'sampai-sampai', 'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah', 'se', 'sebab', 'sebabnya',
     'sebagai', 'sbg', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik', 'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak',
     'sebegini', 'sebegitu', 'sebelum', 'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisanya',
     'sebuah', 'sebut', 'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 'sedikit',
     'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh',
     'sejenak', 'sejumlah', 'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 'sekalian', 'sekaligus', 'sekalipun',
     'sekarang', 'sekarang', 'sekecil', 'seketika', 'sekiranya', 'sekitar', 'sekitarnya', 'sekurang-kurangnya',
     'sekurangnya', 'sela', 'selain', 'selaku', 'selalu', 'selama', 'selama-lamanya', 'selamanya', 'selanjutnya', 'seluruh',
     'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya',
     'sementara', 'semisal', 'semisalnya', 'sempat', 'semua', 'semuanya', 'semula', 'sendiri', 'sendirian', 'sendirinya',
     'seolah', 'seolah-olah', 'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti',
     'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera',
     'sesekali', 'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah',
     'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi', 'seusai', 'sewaktu',
     'siap', 'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu', 'sudah', 'sudahkah',
     'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya',
     'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya', 'tapi', 'tegas', 'tegasnya', 'telah', 'tempat',
     'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu',
     'terdapat', 'terdiri', 'terhadap', 'terhadapnya', 'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'terjadinya',
     'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata', 'tersampaikan', 'tersebut', 'tersebutlah',
     'tertentu', 'tertuju', 'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'tiba-tiba', 'tidak', 'tidakkah',
     'tidaklah', 'tiga', 'tinggi', 'toh', 'tunjuk', 'turut', 'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya',
     'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai', 'waduh', 'wah', 'wahai', 'waktu', 'waktunya',
     'walau', 'walaupun', 'wong', 'yaitu', 'yakin', 'yakni', 'yang','yg']


# In[41]:


#Remove Mention
df['bersih1'] = df['bersih1'].str.replace('@\S+', ' ', case=False)

#Remove extra whitespace
df['bersih1'] = df['bersih1'].str.replace("\s(2)", ' ', case=False)

df['bersih1'] = df['bersih1'].str.replace('[^a-zA-Z \n\.]'," ")

df['bersih1'] = df['bersih1'].str.replace('\n'," ")


# In[42]:


#=========================================================================#
#import stopword
from nltk.corpus import stopwords 
stopwords_indonesia = stopwords.words('indonesian')

df['bersih1'] = df['bersih1'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_buatan)]))
#=========================================================================#
df


# In[43]:


user = []
id_user = []
score = []

for users in range(df.shape[0]):
    user_key = df.iloc[[users]]
    x_coba= tokenizer.texts_to_sequences(user_key['bersih1'].values)
    X = pad_sequences(x_coba, maxlen=MAX_SEQUENCE_LENGTH)
    output = model.predict(X)
    label = ['Kehidupan Pribadi', 'Promosi', 'Sarkas/Sindiran']
    y_pred_label = label[np.argmax(output)]
    user.append(df['username'].iloc[users])
    id_user.append(df['id_user'].iloc[users])
    score.append(y_pred_label)


# In[44]:


hasil = pd.DataFrame({'id_users':id_user, 'username':user,'scoring':score})
hasil


# In[45]:


import joblib as job

job.dump(model, 'LSTM_Model_Caption')


# In[46]:


test = job.load('LSTM_Model_Caption')


# In[47]:


user = []
id_user = []
score = []

for users in range(df.shape[0]):
    user_key = df.iloc[[users]]
    x_coba= tokenizer.texts_to_sequences(user_key['bersih1'].values)
    X = pad_sequences(x_coba, maxlen=MAX_SEQUENCE_LENGTH)
    output = test.predict(X)
    label = ['Kehidupan Pribadi', 'Promosi', 'Sarkas/Sindiran']
    y_pred_label = label[np.argmax(output)]
    user.append(df['username'].iloc[users])
    id_user.append(df['id_user'].iloc[users])
    score.append(y_pred_label)


# In[48]:


hasil = pd.DataFrame({'id_users':id_user, 'username':user,'scoring':score})
hasil


# In[ ]:




