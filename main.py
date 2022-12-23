"""

arr = np.array([1, 2, 3, 4, 5])

print(arr)
print(type(arr))
arr = np.array(10)
print(arr)
arr = np.array([[1, 2], [3, 4]])
print(arr)
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
arr1 = np.array(10)
arr2 = np.array([1, 2, 3, 4, 5])
arr3 = np.array([[1, 2], [3, 4]])
print(arr[0])
print(arr1.ndim, arr2.ndim, arr3.ndim)
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1, 1])
print(arr[0, -1])

print(arr[1:4])
print(arr[2:])
print(arr[:4])
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(arr[0: 7: 2])
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1, 0:2])
copy = arr.copy()

copy[0] = 24

print(arr)
print(copy)
view = arr.view()
view[0] = 24
arr[2] = 22
print(arr)
print(view)
print(arr.shape)
print(arr.reshape(2, 3))
arrList = np.array_split(arr, 4)
print(np.where(arr == 2))
print(np.where(arr % 2 == 1))
print(np.sort(arr))
gia arithmous false true kai str
arr1 = np.array([11, 2, 3, 43])
arr2 = np.array([1, 22, 3, 4])
print(np.add(arr1, arr2))
print(np.subtract(arr1, arr2))
print(np.multiply(arr1, arr2))
print(np.divide(arr1, arr2))
print(np.power(arr1, arr2))
print(np.mod(arr1, arr2))
print(np.absolute(arr3))
print(np.trunc(arr))
print(np.fix(arr))
print(np.around(arr, 1))
print(np.floor(arr))
print(np.ceil(arr))
print(np.log(arr))
print(np.log2(arr))
print(np.log10(arr))
print(np.sum([arr1, arr2]))
print(np.sum([arr1, arr2], axis=1))
print(np.cumsum(arr))
print(np.prod([arr1, arr2],axis = 1))
print(np.lcm.reduce(arr))
print(np.gcd.reduce(arr))
arr = np.array([np.pi / 2, np.pi / 3, np.pi / 4])

print(np.around(np.sin(arr), 8))
print(np.around(np.cos(arr), 8))
arr = np.deg2rad(arr)
arr = np.rad2deg(arr)
print(np.hypot(3, 4))
x = [23, 48, 19]
my_first_series = pd.Series(x)
print(my_first_series)
data = {
    "students": ['Emma', 'John', 'Bob'],
    "grades": [12, 18, 17]
}
my_first_dataframe =  pd.DataFrame(data)
print(my_first_dataframe)
data = {
    "students": ['Emma', 'John', 'Bob'],
    "grades": [12, 18, 17]
}
my_first_df=  pd.DataFrame(data, index=["a", "b", "c"])
print(my_first_df['students'])
first_row = my_first_df.loc["a"]
print(first_row)

second_row = my_first_df.iloc[1]
print(second_row)
import numpy as np
data = {
    "students": ['Emma', 'John', np.nan, 'Bob'],
    "grades": [12, np.nan, 18, 17]
}
my_first_df = pd.DataFrame(data, index=["a", "b", "c","d"])
print(my_first_df.isnull())
my_first_df["students"].fillna("No Name", inplace=True)
my_first_df["grades"].fillna("No Grade", inplace=True)
my_first_df=  pd.DataFrame(data, index=["a", "b", "c", "d"])
df2 = my_first_df.replace(to_replace="Bob", value="Alice")
my_first_df=  pd.DataFrame(data, index=["a", "b", "c", "d"])
df = my_first_df.interpolate(method='linear', limit_direction='forward')
my_first_df=  pd.DataFrame(data, index=["a", "b", "c", "d"])
my_first_df.dropna(inplace=True)
s = pd.Series(['workearly', 'e-learning', 'python'])
for index, value in s.items():
     print(f"Index : {index}, Value : {value}")
my_first_df = pd.DataFrame(data, index=["a", "b"])
for i, j in my_first_df.iterrows():
    print(i, j)
columns = list(my_first_df)
for i in columns:
      print(my_first_df[i][1])
df = pd.read_csv("finance_liquor_sales.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.shape)
mean = df.mean(numeric_only=True)
median = df.median(numeric_only=True)
max_v = df.max(numeric_only=True)
summary = df.describe()
cn = df.groupby('category_name')
print(cn.first())
cn2 = df.groupby(['category_name', 'city'])
print(cn2.first())
cn = df.groupby('category_name')
print(cn.aggregate(np.sum))
cn2 = df.groupby(['category_name', 'city'])
print(cn2.agg({'bottles_sold': 'sum', 'sale_dollars': 'mean'}))
ng = df.groupby('vendor_name')
print(ng.filter(lambda x : len(x) >= 20))
d1 = {'Name': ['Mary', 'John', 'Alice', 'Bob'],
         'Age': [27, 24, 22, 32],
         'Position': ['Data Analyst', 'Trainee', 'QA Tester', 'IT']}
d2 = {'Name': ['Steve', 'Tom', 'Jenny', 'Nick'],
          'Age': [37, 25, 24, 52],
          'Position': ['IT', 'Data Analyst', 'Consultant', 'IT']}
df1 = pd.DataFrame(d1, index=[0, 1, 2, 3])
df2 = pd.DataFrame(d2, index=[4, 5, 6, 7])
result = pd.concat([df1, df2])
d1 = {'key': ['a', 'b', 'c', 'd'],
         'Name': ['Mary', 'John', 'Alice', 'Bob']}
d2 = {'key': ['a', 'b', 'c', 'd'],
          'Age': [27, 24, 22, 32]}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
result = pd.merge(df1, df2, on='key')
d1 = {'Name': ['Mary', 'John', 'Alice', 'Bob'],
         'Age': [27, 24, 42, 32]}
d2 = {'Position': ['Data Analyst', 'Trainee', 'QA Tester', 'IT'],
          'Years_of_experience':[5, 1, 10, 3] }
df1 = pd.DataFrame(d1, index=[0, 1, 2, 3])
df2 = pd.DataFrame(d2, index=[0, 2, 3, 4])
result = df1.join(df2, how='inner')
L = [5, 10, 15, 20, 25]
ds = pd.Series(L)
d = {'col1': [1, 2, 3, 4, 7, 11],
       'col2': [4, 5, 6, 9, 5, 0],
       'col3': [7, 5, 8, 12, 1,11]}
df = pd.DataFrame(d)
s1 = df.iloc[:, 0]
print("1st column as a Series:")
print(s1)
print(type(s1))
df = pd.read_csv('data.csv')
print(df.head(20))
for i, j in df.iterrows():
   print(i, j)

import pandas as pd
import numpy as np
data = pd.read_csv('1.supermarket.csv')

print(data.head())
print('\nShape of dataset:', data.shape)
print()
print(data.info())
x = data.groupby('item_name')
x = x.sum()
print(x)
import matplotlib.pyplot as plt
plt.plot([0, 10], [0, 300] ,marker = 'o')
plt.show()
plt.plot([0, 2, 4], [3, 8, 1], ls='dotted')
plt.title("Title")
plt.xlabel("X - Axis")
plt.ylabel("y - Axis")
plt.grid()
plt.subplot(3, 1, 1)
plt.plot([0, 2, 4, 6, 8, 10], [3, 8, 1, 10, 5, 12])
plt.subplot(3, 1, 2)
plt.plot([0, 10], [0, 300])
plt.subplot(3, 1, 3)
plt.plot([0, 10], [0, 100])
x = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
y = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
plt.scatter(x, y)
import numpy as np
x = np.array([99, 86, 87, 88, 111, 86,
              103, 87, 94, 78, 77, 85, 86])
y = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
plt.scatter(x, y)
x = np.array([100, 105, 84, 105, 90, 99,
              90, 95, 94, 100, 79, 112, 91, 80, 85])
y = np.array([2, 2, 8, 1, 15, 8, 12, 9,
              7, 3, 11, 4, 7, 14, 12])
plt.scatter(x, y)
plt.bar(x, y)
mylabels = np.array(["Potatoes","Bacon", "Tomatoes", "Sausages"])
x = np.array([25, 35, 15, 25])
plt.pie(x, labels=mylabels)
plt.legend()
age = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
cardiac_cases = [5, 15, 20, 40, 55, 55, 70, 80, 90, 95]
survival_chances = [99, 99, 90, 90, 80, 75, 60, 50, 30, 25]
plt.xlabel("Age")
plt.ylabel("Percentage")
plt.plot(age, cardiac_cases, color='black', linewidth=1, label="Cardiac Cases", marker='v',markerfacecolor='red',markersize=12)
plt.plot(age, survival_chances, color='yellow', linewidth=1, label="survival changes", marker='o',markerfacecolor='green',markersize=12)
plt.legend(loc='lower right', ncol=1)
products = np.array([
    ["Apple", "Orange"],
    ["Beef", "Chicken"],
    ["Candy", "Chocolate"],
    ["Fish", "Bread"],
    ["Eggs", "Bacon"]])
random = np.random.randint(2, size=5)
choices = []
counter = 0
for product in products:
    choices.append(product[random[counter]])
    counter += 1
print(choices)
percentages = []
for i in range(4):
    percentages.append(np.random.randint(25))
percentages.append(100 - np.sum(percentages))
print(percentages)
plt.pie(percentages, labels=choices)
plt.legend(loc='lower right', ncol=1)
plt.show()
data = pd.read_csv('1.supermarket.csv')
q = data.groupby('item_name').quantity.sum()
plt.bar(q.index, q, color =['orange', 'purple', 'yellow', 'red', 'green', 'blue', 'cyan'])
plt.xlabel('Items')
plt.xticks(rotation=0)
import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, "html.parser")
print(s.prettify())
print(s.title)
print(s.title.string)
s = BeautifulSoup(url_txt,'lxml')
tag = s.find_all('a')
print(tag)
import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')
my_table = s.find('table', class_='wikitable sortable plainrowheaders')
table_links = my_table.find_all('a', href= True)
actors = []
for links in table_links:
      actors.append(links.get('title'))
print(actors)
"""
print("Hello World")










































