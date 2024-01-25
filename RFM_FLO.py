# Gunda Esra Altınışık Karaca

###############################################################
# PROJECT TASKS
###############################################################

# TASK 1: Data Understanding and Preparation
# 1. Read the flo_data_20K.csv data.
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)   # "%.3f" specifies how many digits of the decimal number we do not want to see.
pd.set_option('display.width', 1000)

df_ = pd.read_csv("WLast_git_projects/rfm_flo/flo_data_20k.csv")
df = df_.copy()

# 2. In the dataset:
# a. First 10 observations,
df.head(10)

# b. variable names,
df.columns

# c. descriptive statistics,
df.describe().T

# d. null values,
df.isnull().sum()

# e. examine variable types.
df.dtypes

# 3. Omnichannel means that customers shop both online and offline platforms. Create new variables for each customer's
# total number of purchases and spending.
df["omnichannel_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["Omnichannel_cv"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head()

# 4. Examine the variable types. Change the type of variables expressing date to date.
date = [col for col in df.columns if "date" in col]
df[date] = df[date].apply(pd.to_datetime)
df.dtypes

# 5. Look at the distribution of the number of customers, average number of products purchased and average expenditures
# across shopping channels.
df.groupby(["order_channel"]).agg({"master_id": "count",
                                   "omnichannel_order": "mean",
                                   "Omnichannel_cv": "mean"})

# 6. List the top 10 customers that bring the most profit.
df.groupby("master_id").agg({"Omnichannel_cv": "mean"}).sort_values("Omnichannel_cv", ascending=False).head(10)

#7. List the top 10 customers who place the most orders.
df.groupby("master_id").agg({"omnichannel_order": "mean"}).sort_values("omnichannel_order", ascending=False).head(10)

# 8. Functionalize the data preparation process.
def data_prep(path, csv=False):
    df_ = pd.read_csv(path)
    df = df_.copy()
    # 2. Veri setinde
    print("###################### Head ######################")
    print(df.head(10))
    print("###################### Features ######################")
    print(df.columns)
    print("###################### Describe ######################")
    print(df.describe().T)
    print("###################### NA ######################")
    print(df.isnull().sum())
    print("###################### Types ######################")
    print(df.dtypes)
    df["omnichannel_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["Omnichannel_cv"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    date = [col for col in df.columns if "date" in col]
    df[date] = df[date].apply(pd.to_datetime)
    print("###################### Groupby Order Channel ######################")
    print(df.groupby(["order_channel"]).agg({"master_id": "count",
                                       "omnichannel_order": "mean",
                                       "Omnichannel_cv": "mean"}))
    print("###################### Top10 by Order ######################")
    print(df.groupby("master_id").agg({"Omnichannel_cv": "mean"}).sort_values("Omnichannel_cv", ascending=False).head(10))
    print("###################### Top10 by Price ######################")
    print(df.groupby("master_id").agg({"omnichannel_order": "mean"}).sort_values("omnichannel_order", ascending=False).head(10))

data_prep("/Users/esraaltinisik/Desktop/PycharmProjects/Miuul-Dönem11/W3/FLOMusteriSegmentasyonu/flo_data_20k.csv", csv=True)

# TASK 2: Calculating RFM Metrics
df.head()
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 2)
type(today_date)
rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'omnichannel_order': lambda omnichannel_order: omnichannel_order,
                                     'Omnichannel_cv': lambda Omnichannel_cv: Omnichannel_cv})
rfm.head(15)
rfm.columns = ['recency', 'frequency', 'monetary']
rfm.describe().T
rfm["monetary"].min()

# TASK 3: Calculating RF and RFM Scores
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))
rfm.describe().T
rfm[rfm["RFM_SCORE"] == "55"]

# TASK 4: Defining RF Scores as Segments
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm.reset_index(inplace=True)

# TASK 5: Action time!
# 1. Examine the recency, frequency and monetary averages of the segments.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").mean()

# 2. With the help of RFM analysis, find the customers in the relevant profile for the following two cases and save
# the customer IDs to CSV.
# a. FLO is adding a new women's shoe brand. The product prices of the included brand are above general customer
# preferences. For this reason, it is desired to communicate specifically with customers who will be interested in the
# promotion of the brand and product sales. Loyal customers (champions, loyal_customers), people who make purchases over
# 250 TL on average and from the women's category, are the customers who will be contacted specifically. Save the ID
# numbers of these customers in the csv file as new_brand_target_customer_id.cvs.
new_df = pd.DataFrame()
new_df = rfm.loc[(rfm["monetary"] > 250)
       & ((rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")),
       ["master_id"]]
new_df.to_csv("new_customers.csv")

# b. Nearly 40% discount is planned for Men's and Children's products. It is intended to specifically target customers
# who are interested in the categories related to this discount, customers who have been good customers in the past but
# have not been shopping for a long time, those who are asleep and new customers. Save the IDs of the customers in the
# appropriate profile in the csv file as discount_target_customer_ids.csv.
df_new = pd.DataFrame()
df_new = rfm.loc[((rfm["segment"] == "new_customers") | (rfm["segment"] == "about_to_sleep")),
       ["master_id"]]
df_new.to_csv("new_customers.csv")
rfm.to_csv("rfm.csv")

# TASK 6: Functionalize the entire process.

def rfm(path, csv=False):
    df_ = pd.read_csv(path)
    df = df_.copy()
    # 2. Calculation of RFM Metrics
    df["omnichannel_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["Omnichannel_cv"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    date = [col for col in df.columns if "date" in col]
    df[date] = df[date].apply(pd.to_datetime)
    df["last_order_date"].max()
    today_date = dt.datetime(2021, 6, 2)
    rfm = df.groupby('master_id').agg(
        {'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
         'omnichannel_order': lambda omnichannel_order: omnichannel_order,
         'Omnichannel_cv': lambda Omnichannel_cv: Omnichannel_cv})
    rfm.columns = ['recency', 'frequency', 'monetary']

    # TASK 3: Calculating RF and RFM Scores
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

    # TASK 4: Defining RF Scores as Segments
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm.reset_index(inplace=True)
    if csv:
        rfm.to_csv("rfm.csv")
    return rfm

result = rfm("WLast_git_projects/rfm_flo/flo_data_20k.csv", csv=True)
print(result)



