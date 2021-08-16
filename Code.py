# Importing the necessary libraries.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, f_regression
import pickle

# Reading the data.
data = pd.read_csv("C:\H_D_P_project/framingham.csv")

# Using a creative way to distinguish and treat null values.
# creating a "Unknown - 5" because number of null values are high.
data["education_nan"] = np.where(data.education.isnull(),1,0)
data["education"].fillna(5, inplace=True)
data["cigsPerDay"].fillna(9, inplace=True)
data["BPMeds_nan"] = np.where(data.BPMeds.isnull(),1,0)
data["BPMeds"].fillna(0, inplace=True)
data["totChol_nan"] = np.where(data.totChol.isnull(),1,0)
data["totChol"].fillna(data.totChol.median(), inplace=True)
data["BMI"].fillna(data.BMI.mean(), inplace=True)

# Has high number of null values, so using a creative way to distinguish and treat null values specially.
# Using median as big outliers are present.
data["glucose_nan"] = np.where(data.glucose.isnull(),1,0)
data["glucose"].fillna(data.glucose.median(), inplace=True)
data["heartRate"].fillna(data.heartRate.median(), inplace=True)

# dividing the data into features and target variable.
x = data.drop(columns="TenYearCHD")
y = data.TenYearCHD

# Fixing the imbalanced data by random oversampling, as our dataset is small.
#ros = RandomOverSampler(random_state=42)
#X_resampled, y_resampled = ros.fit_resample(x, y)
# this step was causing data leakage.

# Splitting the data into features and target variable.
train_x, test_x, train_y, test_y = train_test_split(X_resampled, y_resampled, test_size=0.25)

# Creating pipelies.
pipe4 = Pipeline([("robust_scalar", RobustScaler()),("std_scalar", StandardScaler()), ("XGboost", XGBClassifier())])

# Fitting the pipelines

pipe4.fit(train_x, train_y)
    
# Predicting
pred4 = pipe4.predict(test_x)

# Trying to reduce the unnecessary features
obj = SelectKBest(f_regression, k=4)
new_data = obj.fit_transform(x,y)

filter = obj.get_support()
feature = x.columns
final_f = feature[filter]
print(final_f)

# Using Grid search cv to get the best parameter values

params = {
    'XGboost__random_state'           : [42],
    'XGboost__max_depth'               : [37,38,39,40],
    'XGboost__min_child_weight'        : [1,2,3]
    }

from sklearn.model_selection import GridSearchCV

final_model = GridSearchCV(pipe4, param_grid=params, cv=3)

final_model.fit(train_x[final_f], train_y)

# prediction with test data
prediction = final_model.predict(test_x[final_f])

#Finally saving our model as a pickel file. (For deployment)
pickle.dump(final_model, open('model.pkl','wb'))
