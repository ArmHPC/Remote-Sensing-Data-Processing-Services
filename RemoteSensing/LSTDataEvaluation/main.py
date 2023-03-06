import pandas as pd
import numpy as np

from utils import *
from datetime import datetime



allDF = pd.read_csv('combined_diff_clean_removed.csv')
allDF['datefull'] = allDF['date'] + " " + allDF['time']
allDF['month'] = allDF['datefull'].map(lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S").month)
# # 21---06-night
# # 09--18--day
# allDF['daytime'] = allDF['hour'].map(lambda x: 0 if (x >= 21 or x <= 6) \
#                                                  else 1)
jiminezDF = allDF[["LST.jiminez-munoz.veg", "month","daytime","temp"]][(allDF["LST.jiminez-munoz.veg"].notna())]
priceDF = allDF[["LST.price.veg", "month","daytime","temp"]][(allDF["LST.price.veg"].notna())]
kerrDF = allDF[["LST.kerr.veg", "month","daytime","temp"]][(allDF["LST.kerr.veg"].notna())]
lst10DF = allDF[["LST_10", "month","temp"]][(allDF["LST_10"].notna())]
lst11DF = allDF[["LST_11", "month","temp"]][(allDF["LST_11"].notna())]

jiminezDFGroup = jiminezDF.groupby(["month","daytime"],as_index =False).mean()
priceDFGroup = priceDF.groupby(["month", "daytime"],as_index =False).mean()
kerrDFGroup = kerrDF.groupby(["month", "daytime"],as_index =False).mean()
lst10DFGroup = lst10DF.groupby(["month"],as_index =False).mean()
lst11DFGroup = lst11DF.groupby(["month"],as_index =False).mean()

months = jiminezDFGroup['month'].unique()
daytimes = jiminezDFGroup['daytime'].unique()

measdfDF = pd.DataFrame(columns=["channel", "month", "daytime", "mean", "sdf"])

for m in months:
    for dt in daytimes:
        row = jiminezDF[(jiminezDF["month"] == m) & (jiminezDF["daytime"] == dt)]["LST.jiminez-munoz.veg"].values.tolist()
        mean = calcMean(row)
        sdf = calcSDF(row)
        measdfDF = measdfDF.append({
            "channel": "LST.jiminez-munoz.veg",
            "month": m,
            "daytime": dt,
            "mean": round(mean,2),
            "sdf": round(sdf,2)
        },ignore_index=True)
        
for m in months:
    for dt in daytimes:
        row = priceDF[(priceDF["month"] == m) & (priceDF["daytime"] == dt)]["LST.price.veg"].values.tolist()
        mean = calcMean(row)
        sdf = calcSDF(row)
        measdfDF = measdfDF.append({
            "channel": "LST.price.veg",
            "month": m,
            "daytime": dt,
            "mean": round(mean,2),
            "sdf": round(sdf,2)
        },ignore_index=True)
        
for m in months:
    for dt in daytimes:
        row = kerrDF[(kerrDF["month"] == m) & (kerrDF["daytime"] == dt)]["LST.kerr.veg"].values.tolist()
        
        mean = calcMean(row)
        sdf = calcSDF(row)
        measdfDF = measdfDF.append({
            "channel": "LST.kerr.veg",
            "month": m,
            "daytime": dt,
            "mean": round(mean,2),
            "sdf": round(sdf,2)
        },ignore_index=True)
        
for m in months:
    row = lst10DF[(lst10DF["month"] == m)]["LST_10"].values.tolist()
    mean = calcMean(row)
    sdf = calcSDF(row)
    measdfDF = measdfDF.append({
        "channel": "LST_10",
        "month": m,
        "mean": round(mean,2),
        "sdf": round(sdf,2)
    },ignore_index=True)
        
        
for m in months:
    row = lst11DF[(lst11DF["month"] == m)]["LST_11"].values.tolist()
    mean = calcMean(row)
    sdf = calcSDF(row)
    measdfDF = measdfDF.append({
        "channel": "LST_11",
        "month": m,
        "mean": round(mean,2),
        "sdf": round(sdf,2)
    },ignore_index=True)
        

print(measdfDF)

measdfDF.to_csv('stats_mean_sdf.csv',index=False)