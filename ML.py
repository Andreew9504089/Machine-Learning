# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
import eli5

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from eli5.sklearn import PermutationImportance



# %%
test = pd.read_csv('./test.csv')
train = pd.read_csv('./train.csv')


# %%
train.info(verbose = True)
test.info(verbose=True)


# %%
train['set'] = 1
test['set'] = 0
train_test=train.append(other = test)
train_test.reset_index(drop = False, inplace = True)

# %%
train_test.isnull().sum()


# %%

# Dealing with missing value in CryoSleep with billing feature vice versa
non_sleeping_features = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
train_test.loc[:,non_sleeping_features]=train_test.apply(lambda x: 0 if x.CryoSleep == True else x,axis =1)
train_test['Expenses'] = train_test.loc[:,non_sleeping_features].sum(axis=1)
train_test.loc[:,['CryoSleep']]=train_test.apply(lambda x: True if x.Expenses == 0 and pd.isna(x.CryoSleep) else x,axis =1)
train_test.loc[:,['CryoSleep']]=train_test.apply(lambda x: False if x.Expenses != 0 and pd.isna(x.CryoSleep) else x,axis =1)

# %%
train_test.loc[:,['FirstName']] = train_test.Name.str.split(" ",expand=True).iloc[:,0]
train_test.loc[:,['LastName']] = train_test.Name.str.split(" ",expand=True).iloc[:,1]

# %%
# Generate new feature from existing features
train_test.loc[:,['group']] = train_test.PassengerId.apply(lambda x: x[0:4] ).astype('int')
train_test.loc[:,['id']] = train_test.PassengerId.apply(lambda x: x.split("_")[1]).astype('int')

train_test['group_size']=train_test['group'].map(lambda x: train_test['group'].value_counts()[x])
train_test["Financial"] = train_test["Expenses"].apply(lambda x: "poor" if x < 5000 else ("middle" if x>5000 and x<20000 else "rich"))

# %%


# %%
def add_traveller_type(df):

    checkdict ={

    }
    newColumnTravellerType = []

    for index in range(len(df)):
        gggg =df.iloc[index]['PassengerId'].split('_')[0]
        if df.iloc[index]['Name'] != df.iloc[index]['Name'] :
            lastname = 'NA'
        else :
            lastname =df.iloc[index]['Name'].split(' ')[1]
        if gggg in checkdict:
            checkdict[gggg].append(lastname)
        else :
            checkdict[gggg] = [lastname]

    for index in range(len(df)):
        gggg =df.iloc[index]['PassengerId'].split('_')[0]
        lastnames = checkdict[gggg]
        lastname = 'NA'
        if df.iloc[index]['Name'] == df.iloc[index]['Name'] :
            lastname =df.iloc[index]['Name'].split(' ')[1]

        if len(lastnames) == 1:
            newColumnTravellerType.append('INDIVIDUAL')
        elif len(lastnames) > 1:
            if lastname != 'NA' and lastnames.count(lastname) > 1:
                newColumnTravellerType.append('FAMILY')
            else :
                newColumnTravellerType.append('GROUP')

    df['TravellerType'] = newColumnTravellerType
    return df

train_test=  add_traveller_type(train_test)

# %%
def cabin_regions(df):
    df["Cabin_Region1"] = (df["Cabin_Number"]<300)
    df["Cabin_Region2"] = (df["Cabin_Number"]>=300) & (df["Cabin_Number"]<600)
    df["Cabin_Region3"] = (df["Cabin_Number"]>=600) & (df["Cabin_Number"]<900)
    df["Cabin_Region4"] = (df["Cabin_Number"]>=900) & (df["Cabin_Number"]<1200)
    df["Cabin_Region5"] = (df["Cabin_Number"]>=1200) & (df["Cabin_Number"]<1500)
    df["Cabin_Region6"] = (df["Cabin_Number"]>=1500)

train_test = cabin_regions(train_test)

# %%
# From inspecting impact of same group's member, we can found out that same group's members come from 
# same planet and has high probability is from same family (i.e same last name), same cabin

group_Cabin     = train_test.loc[:,['group','Cabin']].dropna().drop_duplicates('group')
group_LastName = train_test.loc[:,['group','LastName']].dropna().drop_duplicates('group')
group_HomePlanet  = train_test.loc[:,['group','HomePlanet']].dropna().drop_duplicates('group')
train_test      = pd.merge(train_test,group_Cabin,how="left",on='group',suffixes=('','_y'))
train_test      = pd.merge(train_test,group_LastName,how="left",on='group',suffixes=('','_y'))
train_test      = pd.merge(train_test,group_HomePlanet,how="left",on='group',suffixes=('','_y'))

# Fill in the missing value related to group
train_test.loc[:,['Cabin']]=train_test.apply(lambda x:  x.Cabin_y if pd.isna(x.Cabin) else x,axis=1)
train_test.loc[:,['HomePlanet']]=train_test.apply(lambda x:  x.HomePlanet_y if pd.isna(x.HomePlanet) else x,axis=1)
train_test.loc[:,['LastName']]=train_test.apply(lambda x:  x.LastName_y if pd.isna(x.LastName) else x,axis=1)

# %%

train_test.loc[:,['Cabin_deck']] = train_test.Cabin.str.split("/",expand=True).iloc[:,0]
train_test.loc[:,['Cabin_num']] = train_test.Cabin.str.split("/",expand=True).iloc[:,1]
train_test.loc[:,['Cabin_side']] = train_test.Cabin.str.split("/",expand=True).iloc[:,2]

train_test['Cabin_num' ].fillna(value = -1, inplace = True)
train_test['Cabin_num'] = train_test['Cabin_num'].astype(dtype = 'int')


# %%
df = train_test[train_test.duplicated('group', keep=False)].sort_values('group')
df.to_csv('group_inspect.csv', index=False)

# %%
df = train_test.sort_values('group')
df.to_csv('group_inspect.csv', index=False)

# %%
num_cols = ['ShoppingMall','FoodCourt','RoomService','Spa','VRDeck','Age']
cat_cols = ['CryoSleep','Cabin_deck', 'Cabin_side','VIP','HomePlanet','Destination', 'TravellerType', 'Financial']
transported=['Transported']
train_test_1 = train_test[num_cols+cat_cols+transported+['group_size','Expenses', 'Cabin_num','set']].copy()

num_imp = SimpleImputer(strategy='mean')
cat_imp = SimpleImputer(strategy='most_frequent')
ohe = OneHotEncoder (handle_unknown='ignore',sparse = False)
le = LabelEncoder()

# Filling other missing value unrelated to group
train_test[num_cols] = pd.DataFrame(num_imp.fit_transform(train_test_1[num_cols]),columns=num_cols)
train_test[cat_cols] = pd.DataFrame(cat_imp.fit_transform(train_test_1[cat_cols]),columns=cat_cols)
temp_train = pd.DataFrame(ohe.fit_transform(train_test_1[cat_cols]),columns= ohe.get_feature_names_out())
train_test_1 = train_test_1.drop(cat_cols,axis=1)
train_test_1 = pd.concat([train_test_1,temp_train],axis=1)
train_test['Expenses'] = train_test.loc[:,non_sleeping_features].sum(axis=1)


# %%
def get_score(model,X,y):
    n = cross_val_score(model,X,y,scoring ='accuracy',cv=20)
    return n

# %%
train_test_1

# %%
train = train_test_1[train_test['set']== 1].copy()
train.Transported =train.Transported.astype('int')
test = train_test_1[train_test['set'] == 0].drop("Transported",axis=1)
test = test.drop('set',axis=1)

# %%
X = train.drop(['set','Transported'],axis=1)
y = train.Transported

# %%
X,y = shuffle(X,y)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# %%
params_XGB_best= {  'tree_method': "hist",
                    'lambda': 3, 
                    'alpha': 4.58, 
                    'colsample_bytree': 0.9, 
                    'subsample': 0.95, 
                    'learning_rate': 0.07, 
                    'n_estimators': 800, 
                    'max_depth': 4, 
                    'min_child_weight': 1, 
                    'num_parallel_tree': 1}
print(get_score(xgb.XGBClassifier(**params_XGB_best),X,y).mean())

# %%
model = xgb.XGBClassifier()
model.fit(X, y)
xgb.plot_importance(model)

# %%
good_feature = ['group', 'Cabin_num', 'Age', 'Expenses', 'VRDeck', 'FoodCourt', 'ShoppingMall',
                'Spa', 'RoomService', 'group_size','Cabin_side_P']
train = train[good_feature]
test = test[good_feature]

X = train

print(get_score(xgb.XGBClassifier(**params_XGB_best),X,y).mean())

# %%
perm = PermutationImportance(xgb.XGBClassifier(**params_XGB_best), random_state=1,n_iter =10,cv=5).fit(X, y)
eli5.show_weights(perm, feature_names = X.columns.tolist(),top=50)

# %%
# drop_list=['VIP_True', 'Destination_nan', 'Destination_55 Cancri e', 'HomePlanet_nan', 'Destination_PSO J318.5-22']
# X=X.drop(drop_list,axis=1)
# test=test.drop(drop_list,axis=1)
# print(get_score(xgb.XGBClassifier(**params_XGB_best),X,y).mean())

# %%
test

# %%
pred_XGB_best = (xgb.XGBClassifier(**params_XGB_best).fit(X,y)).predict(test)
sample = pd.read_csv('./sample_submission.csv')
sample['Transported'] = pred_XGB_best
sample['Transported']=sample['Transported']>0.5
sample.to_csv('311605002.csv', index=False)


