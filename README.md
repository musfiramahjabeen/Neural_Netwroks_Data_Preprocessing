<H3>MUSFIRA MAHJABEEN M</H3>
<H3>212223230130</H3>
<H3>EX. NO.1</H3>
<H3>26.04.2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Print the dataframe correctly (without quotes around 'df')
print(df)

# Separate features (X) and target (y)
x = df.iloc[:, :-1].values
print(x)

y = df.iloc[:, -1].values
print(y)

# Check for missing values
print(df.isnull().sum())

# Fill missing values with the column mean (rounded to 1 decimal place)
df.fillna(df.mean(numeric_only=True).round(1), inplace=True)

# Check again for missing values
print(df.isnull().sum())

# Update y again after filling missing values (optional, only needed if you expect y to change, but it's fine)
y = df.iloc[:, -1].values
print(y)

df.duplicated()
print(df['Balance'].describe())
scaler = MinMaxScaler()
numeric_features=df.select_dtypes(include=['number'])
df1 = pd.DataFrame(scaler.fit_transform(numeric_features))
print(df1)

X_train , X_test , y_train , y_test = train_test_split(x,y,test_size =0.2)
print(X_test)
print(len(X_train))
print(X_test)
print(len(X_test))
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/26d5ce55-672f-44b7-8367-d959c7c27fd4)
![image](https://github.com/user-attachments/assets/594b9472-8f2e-4d4f-896f-516a8d680701)
![image](https://github.com/user-attachments/assets/f9b654f9-054e-40ec-966b-49094276c511)
![image](https://github.com/user-attachments/assets/f2338fa0-7919-4010-87dd-d7f3cec42d6e)
## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


