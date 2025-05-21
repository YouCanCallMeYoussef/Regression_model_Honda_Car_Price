import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

pd.set_option('display.max_columns', None)
dataFrame = pd.read_csv('honda_car_selling.csv')
print(dataFrame.info())
#--------------------shape-------------------:
print("shape: \n",dataFrame.shape)
print("\nthe data frame: \n",dataFrame)
print(dataFrame.describe(include='all'))
            #checking for missing values
print(dataFrame.isnull().sum()) #no null values

#----------------histogram ---------------------------:

plt.title('Fuel type of honda cars')

Fuel_types=dataFrame["Fuel Type"]
counts = Counter(Fuel_types)

# Extract fuel types and their counts
labels, values = zip(*counts.items())

# Plotting
plt.bar(labels, values, color='skyblue')#log to notice that CNG is not null
# Customize labels and title
plt.xlabel('Fuel Type')
plt.ylabel('Number of cars')
plt.title('Frequency of cars per fuel type')
#plt.show()

#-----------Deleting the single CNG row----------------

dataFrame = dataFrame[dataFrame['Fuel Type'] != ' CNG ']






print("\n\n\n\n\n--------------------------------------------------")
#---------------------convertion-----------------------
def to_dollar(x):
    return str(float(x[:-4])*1196.73)

dataFrame_kakh=dataFrame
dataFrame['Price']=dataFrame['Price'].apply(to_dollar)


# removing 'kms' from kms driven

def shorten(x,a,b):
    ch= x.split()
    res=""
    for i in range (a,b):
        res=res+ch[i]+" "
    return res

dataFrame['kms Driven']=dataFrame['kms Driven'].apply(shorten,args=(0,1))
dataFrame['Car Model']=dataFrame['Car Model'].apply(shorten,args=(0,2))
print(dataFrame['Car Model'].unique())
#-----------Deleting the 4 least frequent car models ----------------

dataFrame = dataFrame[dataFrame['Car Model'] != 'Honda Accord ']
dataFrame = dataFrame[dataFrame['Car Model'] != 'Honda BR-V ']
dataFrame = dataFrame[dataFrame['Car Model'] != 'Honda Mobilio ']
dataFrame = dataFrame[dataFrame['Car Model'] != 'Honda CR-V ']

print(dataFrame['Car Model'].unique())
#--------------encoding_Fuel Type and Suspension------------------

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first').set_output(transform='pandas')
ohetransformFuel=ohe.fit_transform(dataFrame[['Fuel Type']])
ohetransformSusp=ohe.fit_transform(dataFrame[['Suspension']])
ohetransformCarM=ohe.fit_transform(dataFrame[['Car Model']])
dataFrame=pd.concat([dataFrame,ohetransformFuel,ohetransformSusp,ohetransformCarM],axis=1)
counts = dataFrame['Fuel Type'].value_counts()
print(counts)

dataFrame=dataFrame.drop(columns=['Fuel Type'],axis=1)
dataFrame=dataFrame.drop(columns=['Suspension'],axis=1)
dataFrame=dataFrame.drop(columns=['Car Model'],axis=1)



def round_str(x):
    return round(float(x),2)

dataFrame['Price']=dataFrame['Price'].apply(round_str)

print(dataFrame.head())



#splitting X and Y:
X=dataFrame.drop('Price',axis=1)
Y=dataFrame['Price']



from sklearn.model_selection import train_test_split

# Split data into train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
#print("splitted data\n\n\n",X_train, X_test, Y_train, Y_test)


from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.2, random_state=42)
elastic_net.fit(X_train, Y_train)

y_pred = elastic_net.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score


mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


prediction = elastic_net.predict([[2019,19000,1.0,1.0,1.0,0,0,0,0]])
print(prediction)
import joblib

joblib.dump(elastic_net , 'Honda.pkl')









