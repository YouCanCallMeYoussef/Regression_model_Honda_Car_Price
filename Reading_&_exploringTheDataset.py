import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

dataFrame = pd.read_csv('honda_car_selling.csv')
print(dataFrame.info())
#shape:
print("shape: \n",dataFrame.shape)
print("\nthe data frame: \n",dataFrame.head())
print(dataFrame.describe(include='all'))
#checking for missing values
print(dataFrame.isnull().sum()) #no null values


#histogram :
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



# Display the plot
plt.show()