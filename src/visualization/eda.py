import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sp

df=pd.read_csv(r"E:\Python Projects\Churn Modelling using Deep Learning\data\raw\Churn_Modelling.csv")
display(df.shape,df.head())

#functions
def bivariatePercentPlot(f1,f2):
    pd.crosstab(f1,f2,normalize='index').plot(kind='bar',stacked=True)
    plt.ylabel('percentage',fontsize=14,fontweight='bold')
    plt.xlabel(f1.name,fontsize=14,fontweight='bold')
    

df.columns=df.columns.str.lower()

display(df.exited.value_counts(),df.exited.value_counts(normalize=True))

df.drop(['customerid','surname','rownumber'],axis=1,inplace=True)

#numOfProducts
display(df.numofproducts.value_counts(normalize=True),
        df.numofproducts.value_counts())
df['numofproducts'] = df.numofproducts.apply(lambda x: '2+' if x==3 or x==4 else x)
df['numofproducts']=df['numofproducts'].replace({1:'one',2:'two','2+':'twoPlus'})
df['numofproducts']=df['numofproducts'].astype('object')
bivariatePercentPlot(df.numofproducts,df.exited)

#geography
display(df.geography.value_counts(),df.geography.value_counts(normalize=True))
bivariatePercentPlot(df.geography,df.exited)

#gender
display(df.gender.value_counts(),df.gender.value_counts(normalize=True))
bivariatePercentPlot(df.gender,df.exited)
#hasCard
display(df.hascrcard.value_counts(),df.hascrcard.value_counts(normalize=True))
bivariatePercentPlot(df.hascrcard,df.exited)

#isactivemember
display(df.isactivemember.value_counts(),df.isactivemember.value_counts(normalize=True))
bivariatePercentPlot(df.isactivemember,df.exited)

#corr heatmap
numCols=df.select_dtypes('number').columns.to_list()
plt.figure(figsize=(10,10))
sb.heatmap(df[numCols].corr(),cbar=False,cmap='RdYlGn',annot=True)
plt.show()

for col in numCols:
    hist=df[[col,'exited']].hist(by='exited',bins=30)
    plt.show()
pd.plotting.scatter_matrix(df[numCols],figsize=(10,10))

#pairplot
plt.figure(figsize=(12,12))
sb.pairplot(df[numCols],hue='exited')
plt.show()

########### train test split #########################

############## building pipeline #########
#numerical columns
numCols=df.select_dtypes('number').columns.to_list()
numCols.remove('exited')
display(numCols)

#categorical columns-ordinal columns
# ordCols=['numofproducts']

# #categorical columns-nominal columns
# nomCols=['geography', 'gender']

#catCols
catCols=df.select_dtypes('object').columns.to_list()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

# numPip=Pipeline([('stdScaler',StandardScaler())])
# ordPip=Pipeline([('ordinalEncoder',OrdinalEncoder(categories=['1','2','2+']))])
# labPip=Pipeline([('labEncoder',LabelEncoder())])  


numPip=Pipeline([('stdScaler',StandardScaler())])
ohPip=Pipeline([('onehot',OneHotEncoder(drop='first'))])

feaPip=ColumnTransformer([('numPipeline',numPip,numCols),
                          ('ohPipeline',ohPip,catCols)])


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df,df.exited,test_size=0.2,random_state=100,stratify=df.exited)
display(x_train.shape,y_train.shape,x_test.shape,y_test.shape,
        y_train.value_counts(normalize=True),y_test.value_counts(normalize=True))




