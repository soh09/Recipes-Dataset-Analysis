# Recipes Dataset Analysis: Beat the Calories!
by So Hirota (hirotasp92602@gmail.com)

------

## Introduction

### Understanding the Datasets
The recipes dataset contains two .csv files: the RAW_recpies and the RAW_interactions dataset.
RAW_recipes.csv contains `83782 rows` and `12 columns`. The rows represent the recipes, and the columns contain `name`, `id`, `minutes`, `contributor_id`, `submitted`, `tags`, `nutrition`, `n_steps`, `steps`, `description`, `ingredients`, `n_ingredients`.

* | column name | meaning ||-----|-----|
| `name` | the name of the recipe |
| `id` | the id of the recipe |
| `minutes` | the time it takes to make the recipe |
| `contributor_id` | the id of the recipe contributor |
| `submitted` | the date the recipe was submitted, in YY-MM-DD format |
| `tags` | the tags associated with the recipe |
| `nutrition` | nutritional information, in order of calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV) |
| `n_steps` | the number of steps the recipe requires |
| `steps` | the descriptions of each step |
| `description` | the description of the recipe |
| `ingredients` | the ingredients of the recipe | 
| `n_ingredients` | the number of ingredients required to make the recipe | 


RAW_interactions.csv contains `731927 rows` and `5 columns`. The rows represent an individual review of a recipe, and the columns contain `user_id`, `recipe_id`, `date`, `rating`, `review`.

* | columns name | meaning ||-----|-----|
| `user_id` | user id of the user who posted a review |
| `recipe_id` | recipe id for the review, same as the ones in RAW_recipes.csv |
| `date` | the date that the reivew was posted |
| `rating` | the star rating of the recipe, from 1 - 5 |
| `review` | the text review of the recipe |


### Question: What types of recipes have the most calories?
The analysis in this notebook will be centered around this one question. The columns that may be relevant to the analysis include `mintues`, `nutrition` (this contains the calorie information), `tags`, `n_steps`, `n_ingredients`, `ingredients`, and `rating.`
### What's the point?

By investigating this question, a person attempting a diet may be able to avoid high calorie recipes based on the key factors that correlated with high caloric recipes. 

------


## Cleaning and EDA
### Data Cleaning
1. Read in the two datasets using `pd.read_csv()`
    - Recipes
    - | name  |     id |   minutes |   contributor_id | submitted   | tags | nutrition  |   n_steps | steps | description  | ingredients  |   n_ingredients |
|:--------|-------:|----------:|-----------------:|:------------|:----------|:------|----------:|:-----------------|:------------|:-----|----------------:|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  | ['60-minutes-or-less', | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]     |        10 | ['heat the oven to 350f |               9 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  | ['60-minutes-or-less', | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] |        12 | ['pre-heat oven the 350 degrees |  11 |
    - interactions 
    - |    user_id |   recipe_id | date |   rating | review                           |
|-----------:|------------:|:-----------|---------:|:---------------------------------|
|    1293707 |       40893 | 2011-12-21 |        5 | So simple, so delicious! Great fo|
|     126440 |       85009 | 2010-02-27 |        5 | I made the Mexican topping and to|
|      57222 |       85009 | 2011-10-01 |        5 | Made the cheddar bacon topping...|
|     124416 |      120345 | 2011-08-06 |        0 | Just an observation, so I will no|
| 2000192946 |      120345 | 2015-05-10 |        2 | This recipe was OVERLY too sweet.|




2. Created an average ratings column called `avg-rating` using a left merge.
```py
merged = recipes.merge(interactions, how = 'left', left_on = 'id', right_on = 'recipe_id')
```
    - I only want the reviews for recipes that are in RAW_recipes.csv, which is why I perform a left merge. This drops all reviews for recipes that are nonexistent in RAW_recipes.csv.
3. In the merged df, I see that some rating values are 0. If I look at a few rows with rating of 0, I see that these 0 star ratings are actually unreliable. When inspecting the review column for rows with a 0 star rating, I observe that some reviews say that the recipe is wonderful and delicous, while some statet that it was unpleasant. Therefore, I will elect to fill all 0s in the average rating column with np.nan, as this is likely some kind of error that occured during the data generating process. Perhaps the reviewer did not provide a star rating for the recipe since it was optional, and only wrote a written review for it.
4. Fix the data types of various columns
    1. `"submitted"` column was a string, so I changed it a datetime object using `pd.to_datetime()`
    2. `"tags"` was a string, which looked like a list
        1. I first converted the string back to a real list using `.transform()`
        2. Used `MultiLabelBinarizer()` from the sklearn.preprocessing module to perform one-hot encoding
        3. I kept the one-hot encoded df seperate from the recipes df, since it had more than 500 columns
    3. "nutrition" was also a string, which looked like a list formatted like : `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV)]`
        1. I first converted the string back to a list of strings, then the strings to floats
        2. I sliced the list by index and assigned the values to their respective columns like calories, total fat, etc
5. Dropping columns = [`'description'`, `'contributor_id'`] because there is no use for those columns when trying to answer the question.
6. Preparing to convert `calories (#)` into a categorical variable by using `dfcut()` method, which takes in a dataframe and a bin width to "cut" the dataframe's `calories (#)` column by the given width.


------


## Assessment of Missingness




------



## Hypothesis Testing

