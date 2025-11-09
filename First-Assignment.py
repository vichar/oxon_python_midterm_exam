# %% [markdown]
# # Formative Assignment
# 
# **Deadline: Tuesday 28 October 2025.**
# 
# Your first assignment is submitted for feedback only, and does not form part of the assessment of passing or failing this course.
# 
# For this assignment, you must submit a completed workbook. There are five components to it.
# 
# Please email your assignment to onlinemarking@conted.ox.ac.uk along with a completed Declaration of Authorship Form. The Online Courses Office will confirm receipt of your assignment.
# 
# Name your file based on the naming convention: CourseTitle_Assignment1_Surname.pdf
# 

# %%
## Let's import the libraries we are going to use later on
import numpy as np
import pandas as pd

# Let's set the precision for NumPy and Pandas
np.set_printoptions(precision=3)
pd.options.display.float_format = '{:,.2f}'.format

# %% [markdown]
# ## Exercises on Data Structures

# %% [markdown]
# ### 1. Compute statistics on a Python list
# Given the following list
# ```
# l = [12, 45, 12, 999, 10, 8, 76, 20, 10, 10, 7, 70, 17]
# ```
# Compute the mean, the median and the standard deviation of the values in the list, using only built-in functionalities of python (no imported libraries, no NumPy or Pandas)

# %%
l = [12, 45, 12, 999, 10, 8, 76, 20, 10, 10, 7, 70, 17]

# mean

def mean_function(x):
    total_sum = 0
    for current_item in x:
      total_sum = total_sum + current_item
    number_of_terms = len(x)
    mean_value = total_sum/number_of_terms
    return mean_value

result = mean_function(l)

print(f"Mean of {l} is : {result:.2f}")


# %%
l = [12, 45, 12, 999, 10, 8, 76, 20, 10, 10, 7, 70, 17]

def bubble_sort(x):
    number_array = x.copy()
    number_of_term = len(number_array)
    for i in range(number_of_term):
        swapped = False
        for j in range(0, len(number_array) -1 -i):
            if number_array[j] > number_array[j + 1]:
                number_array[j], number_array[j + 1] = number_array[j + 1], number_array[j]
                swapped = True
        if not swapped: 
            break
    return(number_array)

def median_function(values_list): 
    values_list = bubble_sort(values_list.copy())
    number_of_terms = len(values_list)
    if number_of_terms == 0:
        return None
    
    if number_of_terms % 2 == 0:
       median_value = (values_list[number_of_terms // 2 - 1] + values_list[number_of_terms // 2])/2
    else: 
       median_value = values_list[number_of_terms // 2]
    
    return median_value

#median

result = median_function(l)

print(f"Median of {l} is : {result:.0f}")


# %%
l = [12, 45, 12, 999, 10, 8, 76, 20, 10, 10, 7, 70, 17]

# standard deviation

def standard_dev_function(x):
    intermediate_value = 0
    count = 0
    average_value = 0
    for _, current_value in enumerate(x):
        count = count + 1
        delta_1 = current_value - average_value
        average_value = average_value + (delta_1 / count)
        delta_2 = current_value - average_value
        intermediate_value = intermediate_value + (delta_1 * delta_2)
    
    if count < 2: 
        return 0
    
    variance = intermediate_value / count
    
    return (variance ** 0.5)

result = standard_dev_function(l)

print(f"Standard Deviation of {l} is : {result:.2f}")


# %% [markdown]
# ### 2A. Create a dictionary from a list of tuples (records)
# 
# Given the following list of tuples, each one holding a record
# ```
# records = [
#     ("author", 'title', 'publication_year', 'page_count'), 
#     ('J. R. R. Tolkien', 'The Fellowship of the Ring', 1954, 398),
#     ('J. K. Rowling', 'Harry Potter and the Philosopher\'s stone', 1996, 223),
#     ('Evelyn Waugh', 'Brideshead Revisited', 1945, 402),
#     ('Philip K. Dick', 'Ubik', 1969, 202),
#     ('Thomas Pynchon', "Gravity's Rainbow", 1973, 760),
#     ('Stephen King', 'The Stand', 1978, 829)
# ]
# ```
# Convert them into a list of dictionaries, using the first tuple as keys for all the dictionaries and all the other tuples as values, one per dictionary. 
# Expected result:
# 
# ```
# books = [{'author': 'J. R. R. Tolkien',
#   'title': 'The Fellowship of the Ring',
#   'publication_year': 1954,
#   'page_count': 398},
#  {'author': 'J. K. Rowling',
#   'title': "Harry Potter and the Philosopher's stone",
#   'publication_year': 1996,
#   'page_count': 223},
#  {'author': 'Evelyn Waugh',
#   'title': 'Brideshead Revisited',
#   'publication_year': 1945,
#   'page_count': 402},
#  {'author': 'Philip K. Dick',
#   'title': 'Ubik',
#   'publication_year': 1969,
#   'page_count': 202},
#  {'author': 'Thomas Pynchon',
#   'title': "Gravity's Rainbow",
#   'publication_year': 1973,
#   'page_count': 760},
#  {'author': 'Stephen King',
#   'title': 'The Stand',
#   'publication_year': 1978,
#   'page_count': 829}]
# ```

# %%
records = [
    ('author', 'title', 'publication_year', 'page_count'), 
    ('J. R. R. Tolkien', 'The Fellowship of the Ring', 1954, 398),
    ('J. K. Rowling', 'Harry Potter and the Philosopher\'s stone', 1996, 223),
    ('Evelyn Waugh', 'Brideshead Revisited', 1945, 402),
    ('Philip K. Dick', 'Ubik', 1969, 202),
    ('Thomas Pynchon', "Gravity's Rainbow", 1973, 760),
    ('Stephen King', 'The Stand', 1978, 829)
]

## Write your solution here

def populate_dict_function(r): 
    columns = r[0]
    rows = r[1:]
    books = []
    for row in rows :
        book = {}
        for index, column in enumerate(columns):
           book[column] = row[index]
        books.append(book)
    return books

def print_dict_function(x):
    for item in x:
        for key, value in item.items():
            print(f"{key}: {value}")
        print("\n")

books = populate_dict_function(records)
print(f"Transform {records} to : \n")
print(books)
print("\n")
print_dict_function(books)

# %% [markdown]
# ### 2B. Dictionary manipulation
# Write a function that takes the dictionary like `books` and return the author whose book has the highest page count. 
# 
# Apply the function to `books` to verify that it works correctly

# %%
## Write your solution here

sorted_by_page_count = sorted(books, key=lambda x:["page_count"])

print(f"Longest Book by Page Count: {list(sorted_by_page_count)[-1]}")

# %% [markdown]
# ## Exercises on Numpy

# %% [markdown]
# For these exercises we will work on the IRIS dataset, a very popular multivariate dataset for data anaylisis, first introduced by Ronald Fisher in 1936.
# 
# See more about the IRIS dataset heare: https://archive.ics.uci.edu/ml/datasets/Iris
# 
# The data source is the file `iris.data` retrieved from the Web. It contains the data for this example in comma separated values (CSV) format. The number of columns is 5 and the number of rows is 150. The data set is composed of 50 samples from each of three species of Iris Flower: Iris Setosa, Iris Virginica and Iris Versicolor. Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species one from each other.
# 
# The variables are:
# 
#     sepal_length: Sepal length, in centimeters.
#     sepal_width: Sepal width, in centimeters.
#     petal_length: Petal length, in centimeters.
#     petal_width: Petal width, in centimeters.
# 
# The three categories of Irises are: 'Iris Setosa', 'Iris Versicolor', and 'Iris Virginica'. 
# 
# Let's first import the `iris` dataset as a 1-dimensional array of tuples.

# %%
# Input:
# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(
    url, delimiter=',', 
    names=['sepal length', 'sepal width', 'petal length', 'petal width', 'species'],
    dtype=[np.float64, np.float64, np.float64, np.float64, 'U15']
)
iris

# %% [markdown]
# ### 3A. How to convert a 1d array of tuples to a 2d numpy array
# 
# Exercise: convert the `iris` 1D array to a numeric-only 2D array `iris_data` by omitting the species text field. Create a `iris_label` 1D array containing only the species text field. Keep the same indexing/order as in the original array.

# %%
import numbers
import numpy as np

## Write your solution here

def numerical_data_function(r): 
   rows = []
   labels = []
   for item in r:
      row = [column for column in item if isinstance(column, numbers.Number)] 
      rows.append(row)
      label = [column for column in item if isinstance(column, str)]
      labels.append(label[0]) 
   return (np.array(rows, dtype=float), np.array(labels, dtype=str)) 

def print_geometry_function(x):
    for item in x:
        print(f"[{', '.join(map(str,item))}]")

def print_labels_function(x):
    print(f"[{', '.join(map(str,x))}]")

(geometric_values, label_values) = numerical_data_function(iris)
print_geometry_function(geometric_values)
print_labels_function(label_values)

# %% [markdown]
# ### 3B. Split the IRIS dataset by Label
# 
# 
# Split the dataset in `iris_data` according to the labels `iris_labels`. 

# %%
# Write your solution here
def create_dictionary_function(r): 
   results = []
   labels = []
   for index, item in enumerate(r):
      row = [column for column in item if isinstance(column, numbers.Number)] 
      label = [column for column in item if isinstance(column, str)]
      iris_dictionary = { "index" : index, "label" : label[0] , "values" : row }
      results.append(iris_dictionary)
   return np.array(results, dtype=dict)

def create_iris_label_set_function(r): 
   rows = []
   labels = []
   for item in r:
      label = [column for column in item if isinstance(column, str)]
      labels.append(label[0]) 
   labels_array = np.array(labels, dtype=str)
   return list(set(labels_array))

def initialise_iris_dict_function(x):
   value_dict = {}
   for value in x:
      value_dict.update({ value : []})
   return value_dict


labels = create_iris_label_set_function(iris)

iris_dict = initialise_iris_dict_function(labels)

values_dict_list = create_dictionary_function(iris)

def loading_value(iris_dict, values_dict_array): 
   for row in values_dict_array:
      iris_dict[row["label"]].append({"sepal_length": row['values'][0], "sepal_width" : row['values'][1], "petal_length" : row['values'][2], "petal_width" : row['values'][3]})

loading_value(iris_dict, values_dict_list)

def print_dictionary_function(x):
   for key, values in x.items():
      print(f"{key}: ")
      for value in values:
         print(f" {value} ")

print_dictionary_function(iris_dict)

# %% [markdown]
# ### 4. Compute statistics with Numpy
# 
# 1) For each flower species compute the key statistics for `sepal width`:
# - mean
# - median
# - standard deviation
# 
# #### Answer the following questions:
# 
#   a) Which is the flower type with the largest mean value for `sepal_width`?
# 
#   b) Which is the flower type with the smallest median value for `sepal_width`?

# %%
### Compute mean, median, and standard deviation for `sepal width` here:

def extract_sepal_width_function (iris_dict):
    iris_sepal_width_rows = []
    for species, irises in iris_dict.items():
        values = [iris['sepal_width'] for iris in irises]
        iris_sepal_width_rows.append({species: values})
    return iris_sepal_width_rows

def compute_sepal_width_stats_function(iris_dict):
    iris_stats = []
    for species, irises in iris_dict.items():
        values = [row['sepal_width'] for row in irises]
        iris_stat = {"species" : species, "mean": np.mean(values), "median" : np.median(values), "std" : np.std(values)}
        iris_stats.append(iris_stat)
    return iris_stats

iris_sepal_width_rows = extract_sepal_width_function(iris_dict)
iris_stats = compute_sepal_width_stats_function(iris_dict)

sorted_by_mean_desc = sorted(iris_stats, key=lambda iris_stat: iris_stat['mean'], reverse=True)
print(f"The species with the largest mean value for sepal width is {sorted_by_mean_desc[0]['species']}")

sorted_by_median_asc = sorted(iris_stats, key=lambda iris_stat: iris_stat['median'], reverse=False)
print(f"The species with the smallest median value for sepal width is {sorted_by_median_asc[0]['species']}")


# %% [markdown]
# 2) Compute the correlation matrix between `sepal_width` and `petal_length` for the three scpecies. Which is the species that shows the highest correlation among the two parameters?

# %%
### Compute the correlation coefficients here

def extract_petal_length_function (iris_dict):
    iris_petal_rows = []
    for species, irises in iris_dict.items():
        values = [iris['petal_length'] for iris in irises]
        iris_petal_rows.append({species: values})
    return iris_petal_rows

iris_petal_length_rows = extract_petal_length_function(iris_dict)


def compute_correlation_matrix_function(iris_sepal_width_rows, iris_petal_length_rows):
    correlations = {}
    for iris_sepal_row in iris_sepal_width_rows:
        current_key = list(iris_sepal_row.keys())[0]
        iris_petal_length_row = list(filter(lambda d: current_key in d, iris_petal_length_rows))
 
        width_values = [list(d.values())[0] for d in iris_sepal_width_rows]
        length_values = [list(d.values())[0] for d in iris_petal_length_rows]
        correlations[current_key] = np.corrcoef(width_values, length_values)[0, 1]
    return correlations

correlations_matrix = compute_correlation_matrix_function(iris_sepal_width_rows, iris_petal_length_rows)
sorted_correlations_matrix = dict(sorted(correlations_matrix.items(), key=lambda item: item[1], reverse=True))
highest_correlation_species = next(iter(sorted_correlations_matrix))

print(f"{highest_correlation_species} is the species that shows the highest correlation among sepal width and petal length.")

# %% [markdown]
# ### 5. How to create a new column from existing columns of a numpy array
# 
# Create a new column for the flower volume V in `iris`, where volume is computing according to this formula (the assumption here is that the shape of the iris flower is approximately conical):
# 
# $$ Volume = \frac{\pi \times Sepal Length^2 \times Petal Length}{3}$$

# %%
import math

pi_value = math.pi

# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

# Write your solution here

headers = ['volume', 'sepal length', 'sepal width', 'petal length', 'petal width', 'species']
iris_2d_with_volume_rows = []
for value in iris_2d:
    sepal_length = float(value[0].decode())
    petal_length = float(value[2].decode())
    volume = (pi_value * (sepal_length ** 2) * petal_length) / 3 
    current_row = new_arr = np.insert(value, 0, volume)
    iris_2d_with_volume_rows.append(current_row)
# Print header
print(f"{headers[0]:<15}{headers[1]:<15}{headers[2]:<15}{headers[3]:<15}{headers[4]:<15}{headers[5]:<15}")
print("-" * 90)

for current_row in iris_2d_with_volume_rows:
    decoded = [x.decode() if isinstance(x, bytes) else x for x in current_row]
    print(f"{float(decoded[0]):<15.4f}{decoded[1]:<15}{decoded[2]:<15}{decoded[3]:<15}{decoded[4]:<15}{decoded[5]:<15}")      


