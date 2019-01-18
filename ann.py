# Artificial Neural Network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset
df=pd.read_csv("data.csv")
df.keys()
df.shape
df.head()
X=df.iloc[:,2:-1]
y=df.iloc[:,1]

#categorial of data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#visualizing the data
sns.pairplot(df,hue="diagnosis",vars=["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"])
sns.countplot(df.diagnosis)

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

#feature scaling Normalization
'''
min_train=X_train.min()
range_train=(X_train-min_train).max()
X_train=(X_train-min_train)/range_train
min_test=X_test.min()
range_test=(X_test-min_test).max()
X_test=(X_test-min_test)/range_test

'''# Feature Scaling standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing the keras libraries and package
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the ANN
classifier=Sequential()
#Adding the input layer and first hidden layer
classifier.add(Dense(output_dim=15,init="uniform",activation="relu",input_dim=30))
#Adding the second hidden layer
classifier.add(Dense(output_dim=15,init="uniform",activation="relu"))
#Adding the output layer
classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))
#compiling the ANN
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

# fitting ANN to the training dataset
classifier.fit(X_train,y_train,batch_size=1,nb_epoch=200)

#predicting the test set result
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
#evaluating the model by confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
# to check accuracy
print(classification_report(y_test,y_pred))

#predicting a single new observation
new_pred=classifier.predict(sc.transform(np.array([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]])))
new_pred=(new_pred>0.5)


#Evaluating the ANN
#Improving the ANN
#Tunning the ANN
