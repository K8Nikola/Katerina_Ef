#Import Libraries
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

Script=sys.argv[0]
DataSet=sys.argv[-1]

#Importing Data
print("Importing Orders from January 2021")

ordersJan = pd.read_csv(DataSet)


############################
#----USER SEGMETATION-------
############################    
ordersJan['SubmitDt']=pd.to_datetime(ordersJan['submit_dt'].str[0:10])

print('Creating Frequency and Monetary Indices')
SegmentDf = ordersJan.groupby(['user_id'])\
                       .agg({'order_id':'count',
                             'basket':'mean'}).reset_index()
SegmentDf.columns=['User','Frequency','Monetary']


#plt.figure(figsize=(12,10))
#plt.subplot(2,1,1)
#sns.distplot(SegmentDf['Frequency'])
#plt.subplot(2,1,2)
#sns.distplot(SegmentDf['Monetary'])
#plt.tight_layout()
#plt.show()


print("Fixing Positive Skewness by applying log transformation")
All_Attr_log =np.log(SegmentDf[['Frequency','Monetary']])
All_Attr_log = All_Attr_log[np.isfinite(All_Attr_log).all(1)]


#Scaling Data
print("Scaling the Data")
scaler=StandardScaler()
scaler.fit(All_Attr_log)
All_Attr_log_normalized=scaler.transform(All_Attr_log)
All_Attr_log_normalized=pd.DataFrame(All_Attr_log_normalized,
                                          index=All_Attr_log.index,
                                          columns=All_Attr_log.columns)


#Elbow Method
print("Calculating Elbow Criterion")
sse={}
for k in range(1,10):
    kmeans=KMeans(n_clusters=k,random_state=1)
    kmeans.fit(All_Attr_log_normalized)
    sse[k]=kmeans.inertia_

#plt.title("Elbow Criterion")
#plt.xlabel('k');plt.ylabel('SSE')
#sns.pointplot(x=list(sse.keys()),y=list(sse.values()))
#plt.show()

    
print("Applying K-Means Algorithm")    
kmeans=KMeans(n_clusters=3,random_state=1)
kmeans.fit(All_Attr_log_normalized)
labels_cl=kmeans.labels_

SegmentDf_b=SegmentDf.merge(All_Attr_log_normalized,
                                      left_index=True,
                                      right_index=True,
                                      how='inner')

SegmentDf_b=SegmentDf_b.assign(Cluster=labels_cl)

SegmentDf_b.groupby(['Cluster']).agg({
        'Frequency_x':'mean',
        'Monetary_x':'mean'})

    
All_Attr_log_normalized=All_Attr_log_normalized.assign(Cluster=labels_cl)

All_Attr_log_normalized_mlt=pd.melt(All_Attr_log_normalized.reset_index(),
        id_vars=['Cluster'],
        value_vars=['Frequency', 'Monetary'],
        var_name='Characteristic',
        value_name='Value')   

plt.title('Cluster Interpretation')    
sns.lineplot(x='Characteristic',y='Value',hue='Cluster',data=All_Attr_log_normalized_mlt)

#Classify Users ordering Breakfast
ordersJan_b=ordersJan.merge(SegmentDf_b[['User','Cluster']],
                left_on='user_id',
                right_on='User',
                how='inner')

print("Classifying Cuisine based on customer segment")
ordersJan_b.groupby(['cuisine_parent','Cluster'])\
           .agg({"user_id" : "count"})\
           .groupby(level=0).apply(lambda x: 100*x/x.sum())\
           .sort_values(by=['cuisine_parent','Cluster',"user_id"], 
                        ascending=[True, True, False])
