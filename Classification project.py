### Classification project
### Edited by : Mahmoud sabry sallam
#1) importing Laibraris
#%%
import numpy as np
import pandas as pd
pd.set_option("display.max_columns",None)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile , f_classif , SelectKBest
import os
os.chdir("I:\machine-learning2\projects\classification example 1")



### 2) Reading Data

# In[2]:


data = pd.read_csv("pd_speech_features.csv")
data.shape


# In[3]:


data.head()


# ### 3) Data Analysis

# In[4]:


data.info()


# In[5]:


data[data.columns[0:20]].info()


# In[6]:


for i in range(0,755,20):
    print(data[data.columns[i:i+20]].info())


# In[7]:


with open("data analysis1",'w') as f :
    for i in range(0,755,20):
        data[data.columns[i:i+20]].info(buf = f)


# In[8]:




# ### 4) Data Cleaning

# In[9]:


DataNulls = dict(data.isna().sum())
DataNulls


# In[10]:


ExistedNulls = {k:v for k,v in zip(DataNulls.keys() , DataNulls.values()) if v!=0}
ExistedNulls


# In[11]:


Sorted_list = sorted(ExistedNulls.items() , key = lambda x:x[1] , reverse = True)
Sorted_list = {i[0]:i[1] for i in Sorted_list}
Sorted_list


# In[12]:


data.drop(["locShimmer"] ,axis = 1 , inplace = True)


# In[13]:


NullsInRows = dict(data.isna().sum(axis = 1))

NullsInRows = sorted(NullsInRows.items() , key = lambda x:x[1] , reverse = True)
NullsInRows = {i[0]:i[1] for i in NullsInRows}
NullsInRows


# In[14]:


data.drop([13,34] , axis = 0 , inplace = True)


# In[15]:


DataNulls = dict(data.isna().sum())
DataNulls


# In[16]:


ExistedNulls = {k:v for k,v in zip(DataNulls.keys() , DataNulls.values()) if v!=0}
ExistedNulls


# In[17]:


data.drop(["stdDevPeriodPulses"] ,axis = 1 , inplace = True)


# In[18]:


ExistedNulls = {k:v for k,v in zip(DataNulls.keys() , DataNulls.values()) if v!=0}
ExistedNulls


# In[ ]:


data


# In[ ]:


data.reset_index(inplace = True)
data.drop(['index' , 'id'] , axis= 1 , inplace = True)
data


# In[ ]:


ColumnsTypes = dict(data.dtypes)
ColumnsTypes = {k:str(v) for k,v in zip(ColumnsTypes.keys() , ColumnsTypes.values())}
ColumnsTypes


# In[ ]:


WrongColumns= {k:v for k,v in zip(ColumnsTypes.keys(),ColumnsTypes.values()) if v == 'object'}
WrongColumns


# ### 5) Outliers

# In[ ]:


data.boxplot('DFA')


# In[ ]:


data.boxplot('numPulses')


# In[ ]:


data.boxplot('RPDE')


# In[ ]:


for x in ('RPDE' , 'DFA' , 'numPulses'):
    q75,q25 = np.percentile(data.loc[:,x] , [75,25])
    intr_qr = q75-q25
    max = q75+(1*intr_qr)
    min = q25-(1*intr_qr)
    print(f"for {x} has min outliers {data.loc[data[x]< min , x].shape[0]} rows and max has {data.loc[data[x]> max , x].shape[0]} rows ")
    data.loc[data[x]< min , x] = np.nan
    data.loc[data[x]> max , x] = np.nan

    
data = data.dropna(axis = 0)
data.reset_index(inplace = True)
data.drop(['index'] ,axis = 1 , inplace = True)


# ### 6) Feature Extraction

# In[ ]:


data


# In[ ]:


for x in ('RPDE' , 'DFA' , 'numPulses'):
    q75,q25 = np.percentile(data.loc[:,x] , [75,25])
    intr_qr = q75-q25
    max = q75+(1*intr_qr)
    min = q25-(1*intr_qr)
    print(f"for {x} has min outliers {data.loc[data[x]< min , x].shape[0]} rows and max has {data.loc[data[x]> max , x].shape[0]} rows ")


# In[ ]:


int(0.5)


# In[ ]:


def CalculatingSignalType(Mean,Std,Pct,Abs):
    Result = ((float(Mean)*5)+(float(Std)*17)-(float(Pct)*3)) /(float(Abs)+0.6)
    if Result <0.05 :
        return "weak"
    elif Result < 0.1 :
        return "medium"
    else :
        return "strong"
data["SignalType"] = data.apply(lambda x:CalculatingSignalType (x["meanPeriodPulses"],
                                                               x["locPctJitter"],
                                                               x['locAbsJitter'],
                                                               x['rapJitter']) , axis = 1)


# In[ ]:


data['SignalType'].value_counts()


# In[ ]:


data['SignalType'] = data["SignalType"].replace({"weak" : 0 , "medium" : 1 , "strong" : 2})
data['SignalType'].value_counts()


# ### 7) Feature Selection

# In[ ]:


data.head()


# In[ ]:


data['class'].value_counts()


# In[ ]:


x = data.drop(["class"] , axis = 1)
y = data['class']


# In[ ]:


FeatureSelection = SelectPercentile(score_func = f_classif , percentile = 13.2)
X_Selected = FeatureSelection.fit_transform(x,y)
NewData = pd.DataFrame(X_Selected , columns = [ i for i,j in zip(x.columns , FeatureSelection.get_support()) if j])
NewData


# In[ ]:


FeatureSelection1 = SelectKBest(score_func = f_classif , k =100)
x_Selected = FeatureSelection1.fit_transform(x,y)
NewData = pd.DataFrame(x_Selected , columns = [ i for i,j in zip(x.columns , FeatureSelection1.get_support()) if j])
NewData


# In[ ]:


NewData['class'] = y
NewData


# In[ ]:


NewData.to_csv('data after select best feature1')


# ### 8) Visualiztion

# In[ ]:


data.head()


# In[ ]:


data['class'].value_counts()


# In[ ]:


data["gender"].value_counts()


# In[ ]:


def kplot(feature) :
    global data
    fig , ax = plt.subplots(figsize = (10,6))
    sns.kdeplot(data[feature] ,shade = True) 


# In[ ]:


kplot("PPE")


# In[ ]:


kplot("DFA")


# In[ ]:


kplot("numPulses")


# In[ ]:


sns.countplot(x = 'gender' , data = data)


# In[ ]:





# In[ ]:


def Keplot(feature , FirstName , SecondName , SelectedFeature , FirstValue , SecondValue):
    global data
    fig,ax = plt.subplots(figsize = (30,6))
    
    plt.subplot(1,3,1)
    plt.title('Total')
    Data = data
    sns.kdeplot(Data[feature] , shade = True)
    
    plt.subplot(1,3,2)
    plt.title(FirstName)
    Data = data[data[SelectedFeature] == FirstValue]
    sns.kdeplot(Data[feature] , shade = True)
    
    plt.subplot(1,3,3)
    plt.title(SecondValue)
    Data = data[data[SelectedFeature] == SecondValue]
    sns.kdeplot(Data[feature] , shade = True)


# In[ ]:


Keplot('PPE','Male','Female','gender',1,0) 


# In[ ]:


Keplot('numPulses','High','Low','class',1,0) 


# In[ ]:


def Bplot(feature1,feature2 = None , hue = None):
    global data
    fig ,ax = plt.subplots(figsize=(10,6))
    if feature2 == None and hue == None :
        sns.boxplot(data[feature1] , width = 0.3 , color = 'r')
    elif feature2 != None and hue == None :
        sns.boxplot(x = data[feature1], y = data[feature2] , width = 0.3 , color = 'r')
    else :
        sns.boxplot(x = data[feature1], y = data[feature2], width = 0.3 , color = 'r')


# In[ ]:


Bplot('PPE')


# In[ ]:


def CPlot(feature) : 
    global data
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(x = feature , data = data , facecolor = (0,0,0,0) , linewidth = 3 , edgecolor = sns.color_palette('dark' , 3))


# In[ ]:


CPlot('gender')


# In[ ]:


CPlot('class')


# In[ ]:


def RELKCPlot (feature , FirstName , SecondName , SelectedFeature , FirstValue , SecondValue):
    global data
    fig , ax = plt.subplots(figsize = (30,6))
    
    plt.subplot(1,3,1)
    plt.title('total')
    Data = data
    sns.countplot(x = feature , data = Data , facecolor = (0,0,0,0) , linewidth = 3 , edgecolor = sns.color_palette('dark' , 3))
    
    plt.subplot(1,3,2)
    plt.title(FirstName)
    Data = data[data[SelectedFeature] == FirstValue]
    sns.countplot(x = feature , data = Data , facecolor = (0,0,0,0) , linewidth = 3 , edgecolor = sns.color_palette('dark' , 3))
    
    plt.subplot(1,3,3)
    plt.title(SecondName)
    Data = data[data[SelectedFeature] == SecondValue]
    sns.countplot(x = feature , data = Data , facecolor = (0,0,0,0) , linewidth = 3 , edgecolor = sns.color_palette('dark' , 3))
    


# In[ ]:


RELKCPlot('class','Male','Female','gender',1,0)


# In[ ]:


def pie(feature , limit = 10):
    global data
    fig , ax = plt.subplots(figsize = (10,6))
    plt.pie(data[feature].value_counts()[:limit],
           labels = list(data[feature].value_counts()[:limit].index),
           autopct = "%1.2f%%", labeldistance = 1.1, 
           explode = [0.05 for i in range(len(data[feature].value_counts()[:limit]))])
plt.show()


# In[ ]:


pie('class')


# ### 9) Building Model

# In[ ]:


data.head()


# In[ ]:


x = data.drop(['class'] , axis = 1 )
y = data['class'] 


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25 , random_state = 44 , shuffle = True)

print("x_train sahpe is : " , x_train.shape)
print("x_test sahpe is : " , x_test.shape)
print("y_train sahpe is : " , y_train.shape)
print("x_test sahpe is : " , y_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=300,max_depth=7,random_state=33) 
RandomForestClassifierModel.fit(x_train, y_train)
y_pred = RandomForestClassifierModel.predict(x_test)
y_pred


# In[ ]:


y_test.values


# In[ ]:


Results = []
for pred,actual in zip(y_pred,y_test.values):
    if pred == actual:
        Results.append('correct')
    else :
        Results.append('incorrect')
pd.Series(Results).value_counts()
    


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
print("confusion_matrix is : \n" , cm)


# In[ ]:


sns.heatmap(cm , center = True , cmap = 'Blues_r')
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
classif_rep = classification_report(y_test , y_pred)
print("classification report is : \n" , classif_rep)


# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier 
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB , BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis


# In[ ]:


GaussianNBmodel = GaussianNB()
BernoulliNBmodel =BernoulliNB(alpha = 1.0 , binarize = 1)
LogisticRegressionmodel = LogisticRegression(penalty = 12.2 , solver = 'sag' , C =1.0 , random_state = 33)
SGDClassifiermodel = SGDClassifier(penalty = 12.2 , loss = 'squared_loss' , learning_rate = "optimal" , random_state = 33)
RandomForestClassifiermodel = RandomForestClassifier(criterion = 'gini' , n_estimators = 300 , max_depth = 7 , random_state = 33)
GDCmodel = GradientBoostingClassifier(n_estimators = 300 , max_depth = 7 , random_state = 33)
SVCmodel = SVC(kernel = 'rbf' , max_iter = 100 , C = 1.0 , gamma = 'auto')
QDAmodel = QuadraticDiscriminantAnalysis(tol = 0.0001)
DecisionTreeClassifiermodel = DecisionTreeClassifier(criterion = 'gini' , max_depth = 7 , random_state = 33)
KNNClassifiermodel = KNeighborsClassifier(n_neighbors = 5 , weights = 'uniform' , algorithm = 'auto')
Models = [ RandomForestClassifier , GradientBoostingClassifier,LogisticRegression ,SGDClassifier,SVC,DecisionTreeClassifier
       ,KNeighborsClassifier , GaussianNB , BernoulliNB ,LinearDiscriminantAnalysis  , QuadraticDiscriminantAnalysis]


# In[ ]:


ModelsScoreAllData = {}
for Model in Models : 
    print(f'for Model {str(Model).split("(")[0]}')
    m = Model()
    m.fit(x_train, y_train)
    print(f"Training Score is :{m.score(x_train , y_train)}")
    print(f"Testing Score is : {m.score(x_test , y_test)}")
    y_pred = m.predict(x_test)
    ClassificationReport = classification_report(y_test , y_pred)
    print(" Classification Report is : \n" , ClassificationReport)
    print(f" Precision value is : {ClassificationReport.split()[19]}")
    print(f" Recall value is : {ClassificationReport.split()[20]}")
    print(f" f1 Score value is : {ClassificationReport.split()[21]}")
    ModelsScoreAllData[str(Model).split("(")[0]] = [ClassidicationReport.split()[19],
                                                   ClassificationReport.split()[20], Classification.split()[21]]
    
    


# In[ ]:


ModelsScoreAllData


# In[ ]:




