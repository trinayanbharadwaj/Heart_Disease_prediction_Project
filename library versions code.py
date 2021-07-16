import pandas as pd
import numpy as np
import sklearn
import flask
import gunicorn
import jinja2
import pickle
import xgboost

print("Version of pandas is ", pd.__version__)
print("Version of numpy is ", np.__version__)
print("Version of sklearn is ", sklearn.__version__)
print("Version of flask is ", flask.__version__)
print("Version of gunicorn is ", gunicorn.__version__)
print("Version of jinja2 is ", jinja2.__version__)
print("Version of pickle is ", pickle.format_version)
print("Version of xgboost is ", xgboost.__version__)



#pandas==1.0.1
#numpy==1.18.1
#scikit-learn==0.24.2
#Flask==1.1.1
#gunicorn==20.1.0
#Jinja2==2.11.1
#pickle==4.0
#xgboost==0.90