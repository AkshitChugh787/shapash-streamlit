import pickle
import pandas as pd
df = pd.read_csv('./sample_250_rec.csv')
df.head()

with open('cat_model.pkl', 'rb') as f:
    pred_model = pickle.load(f)
    
test_data = df[['orderdate','filldate','miss_flag', 'dispensedqty1', 'rxqty', 'daysupp', 'refills', 'rxstatus', 'deliverystatus', 'drug_form', 'routeofadmin', 'sex', 'age', 'drug_name', 'tier', 'Disease', 'THERAPYCODE', 'HouseholdIncome', 'PLANID', 'PLANNAME', 'PAYORID', 'PROVIDERTYPE', 'Cashpaid', 'monthlyCoPay', 'UnitsPerMOnth', 'AffordabilityIndex', 'OrderMonth', 'orderweek','DaysWithNoMedicine', 'totalSupply', 'totalRefills', 'abandonDays']]

test_data['orderdate'] =  pd.to_datetime(test_data['orderdate'], format='%d-%m-%Y')
test_data['filldate'] =  pd.to_datetime(test_data['filldate'], format='%d-%m-%Y')
test_data['MedArrDur']= (test_data.filldate - test_data.orderdate).astype('timedelta64[D]').astype('int')

import numpy as np
object_col=test_data.select_dtypes(include=['object']).columns
for i in object_col:
  test_data[i].replace(np.nan,'NA',inplace=True)
  
drop_col = ['orderdate','filldate']
test_data.drop(drop_col,axis=1,inplace=True)

columns = ['HouseholdIncome']
test_data[columns] = test_data[columns].replace(' N ',0)
test_data[columns] = test_data[columns].replace('-',0)
test_data[columns] = test_data[columns].replace(' - ',0)
test_data[columns] = test_data[columns].replace('NA',0)
test_data[columns] = test_data[columns].astype(float)

X = test_data.drop("miss_flag", axis=1)
y_orig=test_data['miss_flag']

from shapash import SmartExplainer

xpl = SmartExplainer(
    model=pred_model,

)  

xpl.compile(x=X,
             y_target=y_orig # Optional: allows to display True Values vs Predicted Values
           )

app = xpl.run_app(title_story='Dashboard', port=8020)
      