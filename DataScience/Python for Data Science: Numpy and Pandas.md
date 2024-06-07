1. What are NumPy and Pandas?
Tags: #dataframe #data #array 

NumPy is a Python library used for mathematical and scientific computing and can deal with multidimensional data. NumPy allows us to create ndarrays, which are n-dimensional array objects optimized for computational efficiency, and provides a variety of functions for accessing, manipulating, performing different operations, and exporting ndarrays.

Pandas is a Python library that provides fast, flexible, and expressive data structures designed to make working with structured (tabular, multidimensional, potentially heterogeneous) data both easy and intuitive. It is built on top of NumPy and provides a variety of functions for accessing, manipulating, analyzing, and exporting structured data.

 

2. What are the differences between a Python list and a NumPy array?
Tags: #datatypes #difference 

List

NumPy Array (ndarray)

Can have elements of different datatypes

All elements are of the same datatype

Included in of core Python 

Not included in core Python and a part of the NumPy library

Do not support element-wise operations

Support element-wise operations

Require more memory for storage as they have to store the datatype of each element separately

Require less memory for storage as they do not store the datatype of each element separately

Computationally less efficient

Computationally more efficient

 

3. How to load a CSV file in Google Colab?

The following steps can be followed to load a CSV file into Google Colab (refer to the screenshots attached with each step):

Step 1: Upload the csv file in the google drive (driveimage.PNG)

Step 2: Create a new notebook/ open an existing notebook (newnotebook.png)

Step 3: Import numpy and pandas to import use the following code. After that, click on the Files Option on the left (Files.png)

import pandas as pd
import numpy as np
Step 4: Select the Mount Drive option and select the connect to google drive option (mount_drive.png)

Step 5: Expand the Drive option, and browse to your working directory: for this example, we have saved the file in the content folder in MyDrive. After that, copy the file path, by right-clicking on the option (File_location.png)

Step 6: Paste the path in the variable path and run the code (path.png)

Step 7: Call the path variable in read_csv() function to load the file and then df.head() for checking if the data is imported correctly (head-1.png)

 

4. How to load a CSV file in Jupyter Notebook?
The following steps can be followed to load a CSV file into Jupyter Notebook (kindly refer to the screenshots attached with each step):

 

Step 1: Download the CSV file you want to work with (downloadfile.png)

Step 2: Locate the file in the Local Drive (locatefile.png)

Step 3: Right-click on the file and click on Properties (rightclick.png)

Step 4: Copy the file location (copypath.png)

Step 5: Import numpy and pandas using the following code: (importlibraries.png)

import pandas as pd
import numpy as np
Step 6: Paste the path in the variable path and add the filename at the end, as shown below (pathvariable.png)

Note: It is important to replace the single slash (i.e., \) in the file path with a double slash (i.e., \\)

For example, if the filename is StockData.csv and the file path is C:\Users\User\Downloads, then the path variable should be defined as follows:

path = 'C:\\Users\\User\\Downloads\\StockData.csv'
Step 7: Call the path variable in the read_csv() function of pandas to load the file into a pandas dataframe, and store it in a variable, as shown below (readcsv.png)

df = pd.read_csv(path)
Step 8: Call the head() function of the dataframe to check if the data is imported correctly, as shown below (head-2.png)

df.head()
 

5. I used the drop() function to drop a column from the dataframe but the changes are not reflected in the original data. How can I resolve this?
Tags: #dataframe #function #column

The drop() function has a parameter 'inplace' which is set to False by default. This ensures that the function does not make changes to the original dataframe and returns a copy of the dataframe with specified changes. The inplace parameter can be set to True to makes changes in the original data.

 

6. What is Data Aggregation?
Tags: #addition #maximum 

Aggregation refers to performing an operation on multiple rows corresponding to a single column. Some of the aggregation examples are as follows:

sum: It is used to return the sum of the values for the requested axis.
min: It is used to return a minimum of the values for the requested axis.
max: It is used to return maximum values for the requested axis.
 

7. I am trying to read a CSV file but am getting this error:
FileNotFoundError: [Errno 2] No such file or directory:
How to resolve?
Tags: #error #import #df 

The FileNotFound Error is a very common error caused by a mismatch in the directory that the code searches for and the directory of the actual file location.

The error is usually resolved by ensuring the following:

The data to be loaded and the code notebook on which the data needs to be loaded are stored in the same folder.
Ensure that the name of the dataset is correct (check for lowercase and uppercase, check for spaces, etc.)
 

8. What is the difference between .iloc and .loc commands in pandas?
Tags: #pandas #select #filter #access 

loc is a label-based indexing method to access rows and columns of pandas objects. When using the loc method on a dataframe, we specify which rows and columns to access by using the following format:

dataframe.loc[row labels, column labels]
iloc is an integer-based indexing method to access rows and columns of pandas objects. When using the loc method on a dataframe, we specify which rows and which columns we want by using the following format:

dataframe.iloc[row indices, column indices]
 

9. Why does `head' need parenthesis `( )` but `shape` does not?
Tags: #attribute #error #rows #columns

'shape' is an attribute of a pandas object, which means that every time you create a dataframe, pandas would have to precompute the shape of the data and store it in the 'shape' attribute.

'head' is a function provided by pandas and is computed only when we call it. Whenever called, it will return the first 5 rows of the data by default.