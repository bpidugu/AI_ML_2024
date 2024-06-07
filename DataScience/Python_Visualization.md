FAQ - Python for Visualization
1. What is the seaborn library?
Tags: #seaborn #plot #visualisation 

Seaborn is a Python data visualization library built on top of another Python visualization library, matplotlib. It provides a high-level interface for drawing attractive and informative plots. Visualization is the central part of seaborn which helps in the exploration and understanding of data.

 




2. How to order categorical variables in plots in seaborn?
Tags: #plots #hue #order


The order parameter of the seaborn plot function can be used to specify the order of categorical variables. The order parameter can take a list of items or categories to determine the order. For example, if there are two categories category_1 and category_2, and category_2 is to be displayed before category_1, then category_2 can be mentioned first and category_1 second in the order parameter.


sns.catplot(x='variable_to_plot', y='variable_to_plot', hue='hue_variable’, 
            data=df, order = ['category_2','category_1'] ,kind="bar")

The hue_order parameter of the seaborn plot function can be used to specify the 'hue' order of categorical variables. The order parameter can take a list of items or categories to determine the order. For example, if there are two hue categories  hue_1 and hue_2, and hue_2 is to be displayed before hue_1, then hue_2 can be mentioned first and hue_1 second in the order parameter.


hue_order = ['hue_2', 'hue_1']
sns.catplot(x='variable_to_plot', y='variable_to_plot', hue='hue_variable’,
            data=df, hue_order = hue_order, kind="bar")

 

3. How to print labels on the x-axis vertically?
Tags: #print #labels #plt


The xticks() function can be used to rotate the labels on the x-axis. Sample code:


plt.xticks(rotation=90)

 

4. I am getting this error while running the code:
ModuleNotFoundError: No module name 'seaborn'
How to resolve this?
Tags: #error #install #seaborn


The error suggests that the seaborn library is not present in the system. It can be installed by using the following command in a notebook:


!pip install seaborn --user
If the library is already installed and still showing the error message, then the steps mentioned below can be followed:


Step 1: Uninstall the seaborn library.


!pip uninstall seaborn
When the above code is run, it will pop with the command Proceed (y/n)?


Write y in front of '?' symbol.


Step 2: Install the seaborn library.


!pip install seaborn --user

 

5. How do I change the color of the different levels of the hue variable in a countplot to suit my preference?
Tags:#countplot #color #hue


The palette parameter of the countplot function (and many other seaborn plot functions) can be used to change the hue colors of the plot. The palette parameter can only take values that can be interpreted by Seaborn’s color_palette(). Sample Code:


sns.countplot(data=df, x='variable_to_plot', hue='hue_variable’, palette='Spectral')


6 - I am getting this error while running the code.
UsageError: unrecognized arguments: #this code is to plot inline the notebook
How to resolve this?
Tags:#error,#matplotlib


The magic function %matplotlib inline can be used to enable inline plotting, where the plots/graphs will be displayed just below the cell where your plotting commands are written.


Adding a comment in the same line as a magic function will raise an error as Python perceives those comments to be part of the magic command specified in the line. So, it is important to ensure that no comment or other operation is specified in the same line as a magic command. The best practice, in this case, is to pass the comment in the previous or next line as shown below:


%matplotlib inline
# tells Python to actually display the graphs


7. How to set the size of a seaborn plot?
Tags: #height #width #size #matplotlib


The figure() function of matplotlib.pyplot can be used to adjust the figure size of the seaborn plot. The figure() function creates a figure. It has a parameter called figsize which takes a tuple as an argument that contains the height and the width of the plot.

import matplotlib.pyplot as plt
# setting the dimensions of the plot
plt.figure(figsize=(20,7))
 

8. How do I label my histogram plot in Seaborn?
Tags : #axis #label


The xlabel() and ylabel() function of matplotlib.pyplot can be used to set the labels for the histogram.The labels to be assigned can be set as an argument to function. For example :

plt.title('Histogram: Price')
plt.xlabel('Price of cars')
plt.ylabel('Frequency')
sns.histplot(data=df, x='price', color='orange')