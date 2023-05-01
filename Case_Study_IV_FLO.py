###############################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu (Customer Segmentation with Unsupervised Learning)
###############################################################

###############################################################
# İş Problemi (Business Problem): Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering) müşterilerin
# kümelere ayrılması istenmektedir. Ayrıca kümelere ayrılan müşterilerin davranışları gözlemlenmek istenmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline) olarak yapan müşterilerin
# geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# Veri seti 20.000 gözlem değerinden ve 13 değişkenden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channe: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

###############################################################
# GÖREV 1: Veri setini okutunuz ve müşterileri segmentlerken kullanıcağınız değişkenleri seçiniz.
###############################################################

# Gerekli kütüphaneler
import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

# Veri setini okutuyoruz.
df_ = pd.read_csv("Machine_Learning/flo_data_20k.csv")
df = df_.copy()

df.head()
# Tarih değişkeninin veri tipini tarih olarak değiştiriyoruz.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Son alışveriş tarihi
df["last_order_date"].max() # 2021-05-30

# Analiz tarihi son alışveriş tarihinin bir sonraki günü olarak seçildi.
analysis_date = dt.datetime(2021, 6, 1)

# Son satın alımdan bugüne kadar geçen süre bize recency değerini verecek
df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')

# Son alışveriş tarihi ile ilk alışveriş tarihi arasındaki farkı alıyoruz.
df["tenure"] = (df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')

# Modelde kullanacağımız olan değişkenleri model_df'te belirtiyoruz.
model_df = df[["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
               "customer_value_total_ever_online", "recency", "tenure"]]
model_df.head()

###############################################################
# GÖREV 2: K-Means ile Müşteri Segmentasyonu
###############################################################

# 1. Değişkenleri standartlaştırınız. (Değişkenlerimizin çarpıklık durumuna bakıyoruz.)
# SKEWNESS
"""
Skewness, bir olasılık dağılımının çarpıklığını ölçen bir istatistiksel terimdir. Bir dağılımın çarpıklığı, dağılımın
simetrik olmayan kısımlarını ifade eder. Eğer bir dağılım sağa doğru çarpıksa (veri kümesi sağa çarpık olarak 
adlandırılır), yani ortalamadan daha uzun kuyruğa sahipse ve medyan ortalamadan daha düşükse, pozitif çarpıklık olarak 
adlandırılır. Eğer dağılım sola doğru çarpıksa, yani ortalamadan daha kısa kuyruğa sahipse ve medyan ortalamadan daha 
yüksekse, negatif çarpıklık olarak adlandırılır.
"""
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

# Değişkenler özelinde çarpıklığın görselleştirilmesi
plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df, 'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df, 'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df, 'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df, 'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df, 'recency')
plt.subplot(6, 1, 6)
check_skew(model_df, 'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)

# OLasılık dağılımı sağa doğru çarpık olduğundan dolayı Log Transformation işlemini uyguluyoruz.
"""
Log transformation (logaritma dönüşümü), verilerin doğal logaritmasını alarak veri dağılımının düzeltilmesi veya 
normalleştirilmesi için kullanılan bir yöntemdir.
Log transformation, özellikle verilerin sağa çarpık (positively skewed) olduğu durumlarda kullanılır. Sağa çarpık bir 
veri dağılımı, verilerin çoğunun düşük değerlerde oluşu ve yüksek değerlerin ise daha seyrek oluşu anlamına gelir. Bu 
durumda, verilerin doğal logaritması alınarak, yüksek değerlerin etkisi azaltılır ve verilerin dağılımı normalleştirilebilir.
"""
model_df['order_num_total_ever_online'] = np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline'] = np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline'] = np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online'] = np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency'] = np.log1p(model_df['recency'])
model_df['tenure'] = np.log1p(model_df['tenure'])
model_df.head()

# Scaling
"""
Scaling (ölçeklendirme), veri özelliklerinin farklı değer aralıklarına sahip olması durumunda, verilerin aynı ölçekte
olacak şekilde yeniden boyutlandırılması işlemidir. 
"""
sc = MinMaxScaler((0, 1))   # MinMaxScaler metodunu kullanıyoruz.
model_scaling = sc.fit_transform(model_df)
model_df = pd.DataFrame(model_scaling, columns=model_df.columns)
model_df.head()

# 2. Optimum küme sayısını belirleyiniz. Optimum küme ssayısını belirlerken Elbow Yönteminden yararlanacağız.
"""
Elbow yöntemi, kümeleme analizinde kullanılan bir yöntemdir. Bu yöntem, farklı küme sayılarına göre kümeleme 
sonuçlarının incelenmesi ve optimal küme sayısının belirlenmesi için kullanılır. 
"""
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show(block=True)

# 3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
k_means = KMeans(n_clusters=7, random_state=42).fit(model_df)
segments = k_means.labels_
segments

final_df = df[["master_id",
               "order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]

final_df["segment"] = segments
final_df.head(15)

# 4. Her bir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segment").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                 "order_num_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_online": ["mean", "min", "max"],
                                 "recency": ["mean", "min", "max"],
                                 "tenure": ["mean", "min", "max", "count"]})


###############################################################
# GÖREV 3: Hierarchical Clustering (Hiyerarşik Kümeleme) ile Müşteri Segmentasyonu
###############################################################

# 1. Görev 2'de standarlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
# Hiyerarşik kümeleme yönteminde optimal kümeye ulaşmak için Dendrogram yöntemini kullanacağız.
"""
Dendrogram, farklı küme sayılarına göre kümeleme sonuçlarını görselleştirmek ve optimal küme sayısını belirlemek için 
kullanılabilir. Dendrogramda, dikey eksen genellikle benzerlik ölçüleri veya uzaklık değerleri ile ölçeklenir. Optimal 
küme sayısı, dendrogramdaki kesim noktalarına veya yatay eksendeki mesafelere göre belirlenebilir.
"""
hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=10)

plt.axhline(y=1.2, color='r', linestyle='--')
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)

# 2. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)

final_df = df[["master_id",
               "order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]

final_df["segment"] = segments
final_df.head()

# 3. Her bir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segment").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                 "order_num_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_online": ["mean", "min", "max"],
                                 "recency": ["mean", "min", "max"],
                                 "tenure": ["mean", "min", "max", "count"]})