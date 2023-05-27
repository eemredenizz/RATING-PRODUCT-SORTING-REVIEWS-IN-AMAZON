######################################################################
# INDUSTRIAL PROJECT // RATING PRODUCT & SORTING REVIEWS IN AMAZON
######################################################################
#12 Değişken 4915 Gözlem 71.9 MB
#reviewerID: Kullanıcı ID’si
#asin: Ürün ID’si
#reviewerName: Kullanıcı Adı
#helpful: Faydalı değerlendirme derecesi
#reviewText: Değerlendirme
#overall: Ürün rating’i
#summary: Değerlendirme özeti
#unixReviewTime: Değerlendirme zamanı
#reviewTime: Değerlendirme zamanı Raw
#day_diff: Değerlendirmeden itibaren geçen gün sayısı
#helpful_yes: Değerlendirmenin faydalı bulunma sayısı
#total_vote: Değerlendirmeye verilen oy sayısı

#Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.

import pandas as pd
import numpy as np
import datetime as dt
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

df = pd.read_csv("amazon_review.csv")
df.head()
df.info()

df.groupby("asin").agg({"overall": "mean"})

df["reviewTime"] = pd.to_datetime(df["reviewTime"])

current_date = df["reviewTime"].max()

df["days"] = (current_date - df["reviewTime"]).dt.days

ceyrek1 = np.quantile(df["days"], 0.25)
ceyrek2 = np.quantile(df["days"], 0.5)
ceyrek3 = np.quantile(df["days"], 0.75)

df.groupby("asin").agg({"overall": "mean"})
df.loc[df["days"] <=ceyrek1, "overall"].mean()
df.loc[df["days"] <=ceyrek2, "overall"].mean()
df.loc[df["days"] <=ceyrek3, "overall"].mean()
df.loc[df["days"] >ceyrek3, "overall"].mean()

###################
#time_based_sorting
###################
df.loc[df["days"] <=ceyrek1, "overall"].mean() * 27/100 + \
df.loc[df["days"] <=ceyrek2, "overall"].mean() * 26/100 + \
df.loc[df["days"] <=ceyrek3, "overall"].mean() * 24/100 + \
df.loc[df["days"] >ceyrek3, "overall"].mean() * 23/100

##################
#sürecin fonksiyonlaştırılması
##################

def time_based_weighted_average(dataframe, w1=27, w2=26, w3=24, w4=23):
    current_date = dataframe["reviewTime"].max()
    dataframe["review_Time"] = pd.to_datetime(dataframe["reviewTime"])
    dataframe["days"] = (current_date - dataframe["reviewTime"]).dt.days
    ceyrek1 = np.quantile(df["days"], 0.25)
    ceyrek2 = np.quantile(df["days"], 0.5)
    ceyrek3 = np.quantile(df["days"], 0.75)
    return dataframe.loc[dataframe["days"] <= ceyrek1, "overall"].mean() * 27 / 100 + \
        dataframe.loc[dataframe["days"] <= ceyrek2, "overall"].mean() * 26 / 100 + \
        dataframe.loc[dataframe["days"] <= ceyrek3, "overall"].mean() * 24 / 100 + \
        dataframe.loc[dataframe["days"] > ceyrek3, "overall"].mean() * 23 / 100

time_based_weighted_average(df)

#Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.

import pandas as pd
import numpy as np
import datetime as dt
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

df = pd.read_csv("amazon_review.csv")
df.head()
df.info()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

def score_up_down_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no

def score_average_rating(up, down):
    if up + down ==0:
        return 0
    return up/(up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

#score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)