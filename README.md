# Analyzing the Recipes Dataset: Beat the Calories!
by So Hirota (hirotaso92602@gmail.com)

------

## Introduction

### Understanding the Datasets
The recipes dataset contains two .csv files: the RAW_recpies and the RAW_interactions dataset.
RAW_recipes.csv contains `83782 rows` and `12 columns`. The rows represent the recipes, and the columns contain `name`, `id`, `minutes`, `contributor_id`, `submitted`, `tags`, `nutrition`, `n_steps`, `steps`, `description`, `ingredients`, `n_ingredients`. `nutrition` is in "Percentage Daily Value (PDV)" besides `calories (#)`, which is kilocalories. 

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

* Recipes
    - | name  |     id |   minutes |   contributor_id | submitted   | tags | nutrition  |   n_steps | steps | description  | ingredients  | n_ingredients |
|:------------|-------:|----------:|-----------------:|:------------|:-----|:-----------|----------:|:------|:-------------|:-------------|--------------:|
| 1 brownies in the world best ever | 333281 | 40 | 985201 | 2008-10-27  | ['60-minutes-or-less', | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0] | 10 | ['heat the oven to 350f |  these are the most; | ['bittersweet chocolate', | 9 |
| 1 in canada chocolate chip cookies   | 453467 | 45 | 1848091 | 2011-04-11 | ['60-minutes-or-less', | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] | 12 | ['pre-heat oven the 350 | this is the recipe that| ['white sugar', 'brown| 11 |

* interactions 
    - |    user_id |   recipe_id | date |   rating | review                           |
|-----------:|------------:|:-----------|---------:|:---------------------------------|
|    1293707 |       40893 | 2011-12-21 |        5 | So simple, so delicious! Great fo|
|     126440 |       85009 | 2010-02-27 |        5 | I made the Mexican topping and to|
|      57222 |       85009 | 2011-10-01 |        5 | Made the cheddar bacon topping...|
|     124416 |      120345 | 2011-08-06 |        0 | Just an observation, so I will no|
| 2000192946 |      120345 | 2015-05-10 |        2 | This recipe was OVERLY too sweet.|


2. Did a left merge of the two datasets with the code below.
```py
merged = recipes.merge(interactions, how = 'left', left_on = 'id', right_on = 'recipe_id')
```
    * I only want the reviews for recipes that are in RAW_recipes.csv, which is why I perform a left merge. This drops all reviews for recipes that are nonexistent in RAW_recipes.csv.
4. In the merged df, I see that some rating values are 0. If I look at a few rows with rating of 0, I see that these 0 star ratings are actually unreliable. When inspecting the review column for rows with a 0 star rating, I observe that some reviews say that the recipe is wonderful and delicous, while some state that it was unpleasant. Therefore, I will elect to fill all 0s in the average rating column with np.nan, as this is likely some kind of error that occured during the data generating process. Perhaps the reviewer did not provide a star rating for the recipe since it was optional, and only wrote a written review for it.
5. Fix the data types of various columns
    1. `"submitted"` column was a string, so I changed it a datetime object using `pd.to_datetime()`
    2. `"tags"` was a string, which looked like a list
        1. I first converted the string back to a real list using `.transform()`
        2. Used `MultiLabelBinarizer()` from the `sklearn.preprocessing` module to perform one-hot encoding
        3. I kept the one-hot encoded df seperate from the recipes df, since it had more than 500 columns
        - A subset of the one-hot encoded dataframe can be seen below
        - |     id |   3-steps-or-less |   30-minutes-or-less |   4-hours-or-less |   5-ingredients-or-less |
          |-------:|------------------:|---------------------:|------------------:|------------------------:|
          | 286009 |                 0 |                    0 |                 1 |                       0 |
          | 475785 |                 0 |                    0 |                 1 |                       0 |
          | 500166 |                 1 |                    1 |                 0 |                       0 |
    3. "nutrition" was also a string, which looked like a list formatted like : `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV)]`
        1. I first converted the string back to a list of strings, then the strings to floats
        2. I sliced the list by index and assigned the values to their respective columns like calories, total fat, etc
6. Dropping columns = [`'description'`, `'contributor_id'`] because there is likely no use for those columns when trying to answer the question.
7. Preparing to convert `calories (#)` into a categorical variable by using `dfcut()` method, which takes in a dataframe and a bin width to "cut" the dataframe's `calories (#)` column by the given width. This effectively transforms the calories column into a categorical variable. This helps later on when I want to graph relationships between calories and another variable, or when I want to calculate TVD for hypothesis / permutation testing.

At the end, dataframe `post_clean` looks like this.

| name                                 |     id |   minutes | submitted           |   n_steps |   n_ingredients |   avg_rating |   calories (#) |   total fat (PDV) |   sugar (PDV) |   sodium (PDV) |   protein (PDV) |   saturated fat (PDV) |   carbohydrates (PDV) |   cal_bins |                   steps |             ingredients |
|:-------------------------------------|-------:|----------:|:--------------------|----------:|----------------:|-------------:|---------------:|------------------:|--------------:|---------------:|----------------:|----------------------:|----------------------:|-----------:|:------------------------|:------------------------|
| 1 brownies in the world    best ever | 333281 |        40 | 2008-10-27 00:00:00 |        10 |               9 |            4 |          138.4 |                10 |            50 |              3 |               3 |                    19 |                     6 |        200 | ['heat the oven to 350f | [bittersweet chocolate, |
| 1 in canada chocolate chip cookies   | 453467 |        45 | 2011-04-11 00:00:00 |        12 |              11 |            5 |          595.1 |                46 |           211 |             22 |              13 |                    51 |                    26 |        600 | ['pre-heat oven the 350 | [white sugar, brown     |
| 412 broccoli casserole               | 306168 |        40 | 2008-05-30 00:00:00 |         6 |               9 |            5 |          194.8 |                20 |             6 |             32 |              22 |                    36 |                     3 |        200 | ['preheat oven to 350   | [frozen broccoli cut    |

------

### Univariate Analysis
1. `n_ingredients`
    - Plot type: histogram,  box plot
    <iframe src = 'assests/dist_n_ingredients.html' width = 800 height = 800 frameborder = 0> </iframe>
    - This is a historgram and box plot that describes the distributions of the column `n_ingredients`
    - Observations
        1. Overall, a unimodal shape with a slight right skew, a peak at n = 8, and a max of 37.
        2. Generally, the recipes that had a lot of ingredients were recipes for bbq, chili, soups, and other dishes that uses a lot of spices. This makes sense, because there are many types of spices, and often multiple types of spices are used together to create flavor.

2. `calories (#)`
    - Plot type: histogram, box plot
    <iframe src = 'assests/dist_calories.html' width = 800 height = 800 frameborder = 0> </iframe>
    - This is a historgram and box plot that describes the distributions of the column `calories (#)`
    - Observations
        1. Overall, a unimodal shape with a heavy right skew, a peak around 150-220 calories, and a max = 45609.0.
        2. The recipe with the most calories was the "powdered hot cocoa mix" recipe, which was a recipe for 1/2 GALLON of hot cocco. No wonder it has 45609 calories...
    - Exploring the outliers
        * Using this python code to explore the top 10 recipes by calories (output is show below) for relevant columns 
        ```python
        post_clean.sort_values(by = 'calories (#)').iloc[-10:][['name', 'calories (#)', 'minutes', 'n_steps', 'n_ingredients', 'avg_rating']]
        ``` 
        * Recipe with most calories is a powdered hot cocoa mix. I visited the food.com page for this website on the internet and it yield 1/2 gallons, so the incredible caloric count is actually not surprising.
        * The recipes with the most calories seem to be whole meat dishes (ribs) and whole baked goods. Again, no surprise there.

    | name                                                            |   calories (#) |   minutes |   n_steps |   n_ingredients |   avg_rating |
    |:----------------------------------------------------------------|---------------:|----------:|----------:|----------------:|-------------:|
    | granny jones  secret salty sweet biscuit recipe                 |        17551.6 |       150 |         5 |               5 |            3 |
    | homesteader s fireweed honey                                    |        17554   |        60 |         6 |               6 |            5 |
    | cracker snack mix                                               |        18268.7 |        50 |         4 |              10 |            5 |
    | algerian khobz el dar    semolina bread                         |        18656   |       155 |        17 |              11 |            5 |
    | alternate honey barbecue sauce with riblets  applebee s copycat |        21497.8 |       225 |        16 |              12 |            5 |
    | coffee glazed doughnuts                                         |        22371.2 |        69 |        19 |              15 |            5 |
    | hocus pocus cottage cake                                        |        26604.4 |      3000 |        81 |              27 |            5 |
    | ultimate coconut cake ii                                        |        28930.2 |       120 |        53 |              16 |            5 |
    | moonshine  easy                                                 |        36188.8 |      7200 |        27 |               4 |            5 |
    | powdered hot cocoa mix                                          |        45609   |        10 |         4 |               4 |            5 |
    

### Bivariate Analysis
1. Mean `total fat (PDV)` vs `Calories (#)`
    - Plot type: scatterplot
    <iframe src = 'assests/fat_calories.html' width = 800 height = 800 frameborder = 0> </iframe>
    - The x position represents the bin of the calorie (goes up by 200), and the y position of the bar represents the average percentage value for `total fat (PDV)`
    - Observations
        1. There seems to be a linear relationship bewteen `total fat (PDV)` and `calories (#)`, but after around 11000 calories, the data looks somewhat random
        2. The linear relationship is positive, meaning that the higher the caloric value, the higher the total fat percentage is going to be.

2. Median Calories for the Tags with Top 20 Median Calorie Values
    - For this one, I'll use median as the measure of center, as I don't want the measure of center to be skewed due to outliers (and there were quite a few outliers, per the univariate analysis)
    - Additionally, I'll only consider tags that have at least 100 dishes with that tag, as I don't want random tags with a few high calorie dishes to skew the results
    - Plot type: horizontal bar chart
    <iframe src = 'assests/top20tag_calories.html' width = 800 height = 800 frameborder = 0> </iframe>
    - Each bar represents a tag category (such as `meat`, `main dish`) and the lengths of the bar represents the median value for `calories (#)`
    - Observations
        1. `pork-rib`, `whole-chicken`, and `wings` were the top three 
        2. Meat dishes and sphagetti dishes seem to be the most common. Meat dishes tend to be large in portion size, and spaghetti dishes are high in carbohydrates, likely resulting in the high calories. 

## Assessment of Missingness




------



## Hypothesis Testing

