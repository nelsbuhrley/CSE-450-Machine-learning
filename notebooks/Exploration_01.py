# <a href="https://colab.research.google.com/github/nelsbuhrley/CSE-450-Machine-learning/blob/main/notebooks/Exploration_01.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Data Exploration 01
# 
# A consumer watchdog group wants to see if Netflix has more movies for adults or children.
# 
# Using a dataset containing metadata for all of the movies Netflix had available on their platform in 2019, we'll use the MPAA movie rating system to determine if they are correct.
# 
# ## MPAA Movie Ratings:
# * G: All ages admitted.
# * PG: Some material may not be suitable for children.
# * PG-13: Some material may be inappropriate for children under 13.
# * R: Under 17 requires accompanying parent or adult guardian
# * NC-17: No One 17 and Under Admitted
# 
# Most people would consider G and PG as ratings suitable for children. However, not everyone would agree that a PG-13 movie is necssarily a children's movie. It is up to you to decide how to handle this.

# ## Part 1: Import Pandas
# 
# The `pandas library` is a python library used for data analysis and manipulation. It will provide the core functionality for most of what you do in the data exploration and preprocessing stages of most machine learning projects.
# 
# Please see [this Getting Started Guide](https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html) for information on the conventional way to import Pandas into your project, as well as other helpful tips for common Pandas tasks.

import polars as pl
import pandas as pd
# Import both libraries

# ## Part 2: Load the data
# 
# The dataset for this exploration is stored at the following url:
# 
# [https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/netflix_titles.csv](https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/netflix_titles.csv)
# 
# There are lots of ways to load data into your workspace. The easiest way in this case is to [ask Pandas to do it for you](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html).
# 
# ### Initial Data Analysis
# Once you've loaded the data, it's a good idea to poke around a little bit to find out what you're dealing with.
# 
# Some questions you might ask include:
# 
# * What does the data look like?
# * What kind of data is in each column?
# * Do any of the columns have missing values?

# Part 2: Load the dataset into a Pandas dataframe.
raw_data = pl.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/netflix_titles.csv')

# Then, explore the data by seeing what the first few rows look like.
print(raw_data.head(10))

# Next, display a technical summary of the data to determine the data types of each column, and which columns have missing data.

raw_data.describe()

# ## Part 3: Filter the Data
# 
# Since we're just interested in movies, we'll need to [filter out anything](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html) that isn't a movie for our analysis. The `type` feature contains this information.
# 
# Once we have the subset, we should see how many rows it contains. There are a [variety of ways to get the length of a data frame](https://stackoverflow.com/a/38025280/28106).

# Use pandas's filtering abilitites to select the subset of data
# that represents movies, then calculate how many rows are in the filtered data.
data = raw_data.filter(pl.col('type') == 'Movie')
length = data.shape[0]
print(length)

# ### MPAA Ratings
# Now that we have only movies, let's get a [quick count of the values being used](https://pandas.pydata.org/docs/getting_started/intro_tutorials/06_calculate_statistics.html)  in the `rating` feature.

# Determine the number of records for each value of the "rating" feature.
# Remember to count the values in your subset only, not in the original dataframe.
rating_counts = data['rating'].value_counts()
print(rating_counts)

# ### More Filtering
# 
# There are apparently some "made for TV" movies in the list that don't fit the
# MPAA rating scheme.
# 
# Let's [filter some more](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html) to just see movies rated with the standard MPAA ratings of G, PG, PG-13, R, and NC-17.

# Filter the list of movies to select a new subset containing only movies with
# a standard MPAA rating. Calculate how many rows are in this new set, and
# then see which ratings appear most often.
rating_counts = rating_counts.filter(pl.col('rating').is_in(['G', 'PG', 'PG-13', 'R', 'NC-17']))
print(rating_counts)

# ## Part 4: Visualization
# Now that we have explored and preprocessed our data, let's create a visualization to summarize our findings.
# 
# ### Exploration vs Presentation
# Broadly speaking, there are two types of visualizations:
# * Barebones visualizations you might use to get a quick, visual understanding of the data while you're trying to decide how it all fits together.
# * Presentation-quality visualizations that you would include in a report or presentation for management or other stakeholders.
# 
# ### Visualization Tools
# There are many different visualization tools availble. In the sections below,
# we'll explore the three most common. Each of these libraries has strengths and weaknesses.
# 
# It is probably a good idea for you to become familiar with each one, and then become proficient at whichever one you like the best.

# ### Altair
# The Altair visualization library provides a large variety of very easy to use statistical charting tools.
# 
# Altair uses a declarative language to build up charts piece by piece.
# 
# Assume we have a pandas dataframe called `employees`, with three columns: `name`, `job`, `salary`.
# 
#     # Make a box plot style categorical plot showing the distribution of salaries for each job:
#     alt.Chart(employees).mark_boxplot().encode(
#         x='job',
#         y='salary'
#     )
# 
#     # Make a box plot style categorical plot, and customize the results
#     alt.Chart(employees).mark_boxplot().encode(
#         alt.X('job', title='Job title'),
#         alt.Y('salary', title='Annual salary in thousands of $USD')
#     ).properties(
#       title='Salaries by Job Title'
#     )

# Like with Pandas, there is a [conventional way to import Altair](https://altair-viz.github.io/getting_started/starting.html#the-chart-object) into your projects.

# Import the Altair library the conventional way.
# Skiped altair because i want to use seaborn
# import altair as alt

# Let's create a barchart showing the count of each movie rating by using Altair's aggregation capabilities.
# 
# In [this example](https://altair-viz.github.io/getting_started/starting.html#data-transformation-aggregation), we see the x axis being set to a feature called `a`, and the y axis set to the `average()` of a feature called `b`.
# 
# In our case, we want the x axis to be set to `rating` and the y axis to be the `count()` of `rating`.

# Use Altair to create a bar chart comparing the count of each movie rating
# alt.Chart(rating_counts).mark_bar().encode(
#     x=alt.X('rating', sort=['G', 'PG', 'PG-13', 'R', 'NC-17']),
#     y='count'
# )

# ### Seaborn
# While Altair uses a "declarative" syntax for building charts piece by piece, the Seaborn library provides a large variety of pre-made charts for common statistical needs.
# 
# These charts are divided into different categories. Each category has a high-level interface you can use for simplicity, and then a specific function for each chart that you can use if you need more control over how the chart looks.
# 
# Seaborn uses matplotlib for its drawing, and the chart-specific functions each return a matplitlib axes object if you need additional customization.
# 
# For example, there are several different types of categorical plots in seaborn: bar plots, box plots, point plots, count plots, swarm plots, etc...
# 
# Each of these plots can be accessed using the `catplot` function.
# 
# Assume we have a pandas dataframe called `employees`, with three columns: `name`, `job`, `salary`.
# 
#     # Make a box plot style categorical plot showing the distribution of salaries for each job:
#     sns.catplot(data=employees, x='job', y='salary', kind='box')
# 
#     # Make a swarm plot style categorical plot
#     sns.catplot(data=employees, x='job', y='salary', kind='swarm')
# 
# Alternatively, you can use the plot specific functions to give yourself more control over the output by using matplotlib functions:
# 
#     # Make a box plot style categorical plot, and customize the results
#     import matplotlib.pyplot as plt
# 
#     plt.figure(figsize=(12, 9))
#     ax = sns.boxplot(data=employees, x='job', y='salary')
#     ax.set_title("Salaries by Job Title")
#     ax.set_ylabel("Annual salary in thousands of $USD")
#     ax.set_xlabel("Job title")

# Like with Pandas, there is a [conventional way to import Seaborn](https://seaborn.pydata.org/tutorial/relational.html) into your projects.
# 
# Optionally, you may wish to set some default chart aesthetics by [setting the chart style](https://seaborn.pydata.org/tutorial/aesthetics.html).

# Import the seaborn library the conventional way. Then optionally configure
# the default chart style.
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# Since the `rating` column uses categorical data, we need to use Seaborn's [categorical visualizations](https://seaborn.pydata.org/tutorial/categorical.html).
# 
# In particular, we want a "count plot" that will display a count of movie ratings.

ax = sns.barplot(data=rating_counts, x='rating', y='count', order=['G', 'PG', 'PG-13', 'R', 'NC-17'])
ax.set_title('Count of Movie Ratings')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')

# ### Pandas built-in plotting
# In addition to libraries like Altair and Seaborn, Pandas has some [built in charting functionality](https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html).
# 
# While not as sophisticated as some of the other options, it is often good enough for quick visualizations.
# 
# Just like with seaborn's plotting functions, the pandas plotting functions return matplotlib axes objects, which can be further customized.
# 
# Assume we have a pandas dataframe called `employees`, with three columns: `name`, `job`, `salary`.
# 
#     # Make a box plot style categorical plot showing the distribution of salaries for each job:
#     employees[ ['job','salary'] ].plot.box()
# 
#     # Make a box plot style categorical plot, and customize the results
#     import matplotlib.pyplot as plt
# 
#     plt.figure(figsize=(12, 9))
#     ax = employees[ ['job','salary'] ].plot().box()
#     ax.set_title("Salaries by Job Title")
#     ax.set_ylabel("Annual salary in thousands of $USD")
#     ax.set_xlabel("Job title")

# Use pandas' built in plotting functions to create a count plot comparing the count of each movie rating
# This will be a little trickier than the other libraries, but one hint is that the pandas value_counts() function
# actually returns a dataframe.
import matplotlib.pyplot as plt

ax = rating_counts.to_pandas().plot(kind='bar', x='rating', y='count')
ax.set_title('Count of Movie Ratings')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')

summed_cats_pg13_adult = rating_counts.with_columns(
    pl.when(pl.col('rating').is_in(['G', 'PG']))
    .then(pl.lit('Child Movies'))
    .when(pl.col('rating').is_in(['PG-13', 'R', 'NC-17']))
    .then(pl.lit('Adult Movies'))
    .alias('category_group')
).group_by('category_group').agg(pl.col('count').sum())

summed_cats_pg13_child = rating_counts.with_columns(
    pl.when(pl.col('rating').is_in(['PG-13', 'G', 'PG']))
    .then(pl.lit('Child Movies'))
    .when(pl.col('rating').is_in(['R', 'NC-17']))
    .then(pl.lit('Adult Movies'))
    .alias('category_group')
).group_by('category_group').agg(pl.col('count').sum())

total_movies = rating_counts['count'].sum()

display(summed_cats_pg13_adult)
display(summed_cats_pg13_adult.with_columns(pl.col('count')/total_movies))
display(summed_cats_pg13_child)
display(summed_cats_pg13_child.with_columns(pl.col('count')/total_movies))

# # Part 5: Interpretation and Conclusions
# 
# Now that you've seen the data, what is your conclusion? Is the watchdog group
# correct that Netflix has more movies for adults than for children? Are there any
# caveats you'd include in your analysis?

# Write your summary here:

# It depends on where you draw the line, if a PG-13 movie is considerd a child movie there is nearly a even amount of adult and children movies, if it is for adults than there almost 4x as many adult movies as there are children movire

# ## 🌟 Above and Beyond 🌟
# 
# After reviewing your findings, the watchdog group would like some additional questions answered:
# 
# 1. How are things affected if you include the "made for TV movies" that have been assigned [TV ratings](https://en.wikipedia.org/wiki/Television_content_rating_system#United_States) in your analysis, but still exclude unrated movies?
# 
# 2. They would also like to see a separate report that includes only TV shows.
# 
# 3. For an upcoming community meeting, the group would like to present a simple
# chart showing "For Kids" and "For Adults" categories. The easiest way to accomplish this would be to [create a new column in your data frame that maps](https://pandas.pydata.org/docs/getting_started/intro_tutorials/10_text_data.html) each rating to the appropriate "For Kids" or "For Adults" label, then create a new visualization based on that column.

data = raw_data.filter(pl.col('type') == 'TV Show')
length = data.shape[0]
print(length)

rating_counts = data['rating'].value_counts()
print(rating_counts)

rating_counts = rating_counts.filter(pl.col('rating').is_in(['TV-Y', 'TV-PG', 'TV-Y7-FV', 'TV-MA', 'TV-Y7', 'TV-G', 'TV-14']))
print(rating_counts)

ax = sns.barplot(data=rating_counts, x='rating', y='count', order=['TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA'])
ax.set_title('Count of TV Show Ratings')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')

summed_cats_tv14_below = rating_counts.with_columns(
    pl.when(pl.col('rating').is_in(['TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'TV-PG']))
    .then(pl.lit('Child Shows'))
    .when(pl.col('rating').is_in(['TV-14', 'TV-MA']))
    .then(pl.lit('Adult Shows'))
    .alias('category_group')
).group_by('category_group').agg(pl.col('count').sum())

summed_cats_tv14_above = rating_counts.with_columns(
    pl.when(pl.col('rating').is_in(['TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'TV-PG', 'TV-14']))
    .then(pl.lit('Child Shows'))
    .when(pl.col('rating').is_in(['TV-MA']))
    .then(pl.lit('Adult Shows'))
    .alias('category_group')
).group_by('category_group').agg(pl.col('count').sum())

total_shows = rating_counts['count'].sum()

display(summed_cats_tv14_below)
display(summed_cats_tv14_below.with_columns(pl.col('count')/total_shows))
display(summed_cats_tv14_above)
display(summed_cats_tv14_above.with_columns(pl.col('count')/total_shows))
