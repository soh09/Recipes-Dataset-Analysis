# Analyzing the Recipes Dataset: Beat the Calories!

A Data Science Project For DSC80 At The University of California, San Diego

by So Hirota (hirotaso92602@gmail.com)

Published 2/24/2023

------

## Introduction

I will analyze the dataset provided using Pandas, Numpy, Plotly, hypothesis testing, and permutation testing.  

### Understanding the Datasets
The recipes dataset contains two .csv files: the RAW_recpies and the RAW_interactions dataset.


RAW_recipes.csv contains `83782 rows` and `12 columns`. The rows represent the recipes, and the columns contain `name`, `id`, `minutes`, `contributor_id`, `submitted`, `tags`, `nutrition`, `n_steps`, `steps`, `description`, `ingredients`, `n_ingredients`. `nutrition` is in "Percentage Daily Value (PDV)" besides `calories (#)`, which is kilocalories. 

| column name | meaning |
|-----|-----|
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

| columns name | meaning |
|-----|-----|
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

```python
recipes.head(2)
```


| name                               | id     | minutes | contributor_id | submitted  | tags                   | nutrition                                    | n_steps | steps                   | description             | ingredients               | n_ingredients |
|-:----------------------------------|--:-----|--:------|--:-------------|-:----------|-:----------------------|-:--------------------------------------------|--:------|-:-----------------------|-:-----------------------|-:-------------------------|--:------------|
| 1 brownies in the world best ever  | 333281 | 40      | 985201         | 2008-10-27 | ['60-minutes-or-less', | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]     | 10      | ['heat the oven to 350f | these are the most;     | ['bittersweet chocolate', | 9             |
| 1 in canada chocolate chip cookies | 453467 | 45      | 1848091        | 2011-04-11 | ['60-minutes-or-less', | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] | 12      | ['pre-heat oven the 350 | this is the recipe that | ['white sugar', 'brown    | 11            |

```python
interactions.head(5)
``` 


|    user_id |   recipe_id | date       |   rating | review                           |
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
4. In the merged df, I see that some rating values are 0.
    1. Some some reviews say that the recipe is wonderful and delicous, while some state that it was unpleasant. Seems weird.
    2. To get further information, I signed up for food.com and confirmed that one can review a recipe without ever providing a star rating.
        - A "rating of 0" actually means that there was no star rating for that recipe, aka a missing value.
        - This is a direct result of the data generation process, which makes providng a star rating optional
    3. I filled all the 0s with np.nan to clearly indicate that they are missing values. 
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

```python
post_clean.head(3)
```


| name                                 |     id |   minutes | submitted           |   n_steps |   n_ingredients |   avg_rating |   calories (#) |   total fat (PDV) |   sugar (PDV) |   sodium (PDV) |   protein (PDV) |   saturated fat (PDV) |   carbohydrates (PDV) |   cal_bins |                   steps |             ingredients |
|--------------------------------------|--------|-----------|---------------------|-----------|-----------------|--------------|----------------|-------------------|---------------|----------------|------------------|-----------------------|-----------------------|------------|-------------------------|-------------------------|
| 1 brownies in the world best ever     | 333281 |        40 | 2008-10-27 00:00:00 |        10 |               9 |            4 |          138.4 |                10 |            50 |              3 |               3 |                    19 |                     6 |        200 | 'heat the oven to 350f' | '[bittersweet chocolate' |
| 1 in canada chocolate chip cookies   | 453467 |        45 | 2011-04-11 00:00:00 |        12 |              11 |            5 |          595.1 |                46 |           211 |             22 |              13 |                    51 |                    26 |        600 | 'pre-heat oven the 350' | '[white sugar', 'brown' |
| 412 broccoli casserole               | 306168 |        40 | 2008-05-30 00:00:00 |         6 |               9 |            5 |          194.8 |                20 |             6 |             32 |              22 |                    36 |                     3 |        200 | 'preheat oven to 350'   | '[frozen broccoli cut'  |


### Univariate Analysis

Here, I will inspect the relevant columns of the dataframe individually. 


1. `n_ingredients`
    - Plot type: histogram,  box plot
    <iframe src = 'assests/dist_n_ingredients.html' width = 800 height = 600 frameborder = 0> </iframe>
    - This is a historgram and box plot that describes the distributions of the column `n_ingredients`
    - Observations
        1. Overall, a unimodal shape with a slight right skew, a peak at n = 8, and a max of 37.
        2. Generally, the recipes that had a lot of ingredients were recipes for bbq, chili, soups, and other dishes that uses a lot of spices. This makes sense, because there are many types of spices, and often multiple types of spices are used together to create flavor.

2. `calories (#)`
    - Plot type: histogram, box plot
    <iframe src = 'assests/dist_calories.html' width = 800 height = 600 frameborder = 0> </iframe>
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

Here, I will inspect the revelant columns of the dataframe in relation to the `calories (#)` or `cal_bins` column. 


1. Mean `total fat (PDV)` vs `Calories (#)`
    - Plot type: scatterplot
    <iframe src = 'assests/fat_calories.html' width = 800 height = 500 frameborder = 0> </iframe>
    - The x position represents the bin of the calorie (goes up by 200), and the y position of the bar represents the average percentage value for `total fat (PDV)`
    - Observations
        1. There seems to be a linear relationship bewteen `total fat (PDV)` and `calories (#)`. Interestingly, the association is very apparent until 3800 calories, then becomes a bit more variable to around 12000, then becomes very random after that. This may be due to the decreasing number of data points at higher calories.
        2. The linear relationship is positive, meaning that the higher the caloric value, the higher the total fat percentage is going to be.

2. Median Calories for the Tags with Top 20 Median Calorie Values
    - For this one, I'll use median as the measure of center, as I don't want the measure of center to be skewed due to outliers (and there were quite a few outliers, per the univariate analysis)
    - Additionally, I'll only consider tags that have at least 100 dishes with that tag, as I don't want random tags with a few high calorie dishes to skew the results
    - Plot type: horizontal bar chart
    <iframe src = 'assests/top20tag_calories.html' width = 1000 height = 600 frameborder = 0> </iframe>
    - Each bar represents a tag category (such as `meat`, `main dish`) and the lengths of the bar represents the median value for `calories (#)`
    - Observations
        1. `pork-rib`, `whole-chicken`, and `wings` were the top three 
        2. Meat dishes and sphagetti dishes seem to be the most common. Meat dishes tend to be large in portion size, and spaghetti dishes are high in carbohydrates, likely resulting in the high calories. 



### Aggregation

- The relationship of [avg_rating, minutes, n_ingredients] with calories, with mean and median aggregation
    - I will only look at a subset of the recipes, where calories >= 1500. This is because the bulk of the data lies in that range, and to make the length of the resulting pivot table managable.
- Code to generate the table
    ```python
    # cutting the post_clean df by 100s for the calories column
    hundred_width = dfcut(post_clean, 100)
    hundred_width = hundred_width.loc[hundred_width['calories (#)'] <= 1500]
    hundred_width.pivot_table(index = 'cal_bins', values = ['minutes', 'n_ingredients', 'avg_rating'], aggfunc = ['mean', 'median'])
    ```


    | cal_bins | ('mean', 'avg_rating') | ('mean', 'minutes') | ('mean', 'n_ingredients') | ('median', 'avg_rating') | ('median', 'minutes') | ('median', 'n_ingredients') |
    |--:-------|--:---------------------|--:------------------|--:------------------------|--:-----------------------|--:--------------------|--:--------------------------|
    | 100      | 4.64557                | 142.603             | 7.07242                   | 5                        | 20                    | 7                           |
    | 200      | 4.62707                | 71.1157             | 8.23044                   | 5                        | 30                    | 8                           |
    | 300      | 4.62246                | 84.0302             | 9.09422                   | 5                        | 35                    | 9                           |
    | 400      | 4.62012                | 91.412              | 9.6538                    | 5                        | 40                    | 9                           |
    | 500      | 4.6191                 | 195.309             | 10.0194                   | 5                        | 44                    | 10                          |
    | 600      | 4.62436                | 100.17              | 10.3268                   | 5                        | 45                    | 10                          |
    | 700      | 4.61904                | 91.6952             | 10.5283                   | 5                        | 45                    | 10                          |
    | 800      | 4.62502                | 128.646             | 10.6066                   | 5                        | 45                    | 10                          |
    | 900      | 4.64032                | 276.698             | 10.7848                   | 5                        | 50                    | 10                          |
    | 1000     | 4.61836                | 125.211             | 10.7967                   | 5                        | 50                    | 10                          |
    | 1100     | 4.58361                | 120.967             | 10.7433                   | 5                        | 45                    | 10                          |
    | 1200     | 4.62894                | 142.561             | 10.3385                   | 5                        | 45                    | 10                          |
    | 1300     | 4.58186                | 230.44              | 10.1769                   | 5                        | 45                    | 10                          |
    | 1400     | 4.64782                | 281.164             | 10.6641                   | 5                        | 50                    | 10                          |
    | 1500     | 4.6636                 | 124.933             | 10.1953                   | 5                        | 50                    | 9                           |


- Observations
    - `avg_rating` has no observable pattern in relation to `calories`: the means are all around 4.6, while the medians are all 5.0. Perhaps it is uniformly distributed.
    - `minutes` seem random when looking at the mean, but when observing the median, we see a general positive correlation
        - The more calories a recipe has, the longer it takes to prepare
        - It is likely harder to observe this pattern in the means because the mean is more susceptible to outliers
    - `n_ingredients` seems to have a trend of positive corrlation for both means and medians, but levels off at around 600 calories.

-----

## Assessment of Missingness

Let's quickly go over the missingness types. Definitions will be borrowed from the UCSD DSC80 class, lecture 12.

| Type | Definition |
|------|------------|
| Missing By Design (MD) | Missing values can be exactly determined in a column by looking at other columns. |
| Not Missing At Random (NMAR) | Missingness of values dependend on the values themselves. |
| Missing At Random (MAR) | Missingness of Values dependent on other column(s) in the dataset. |
| Missing Completely At Random (MCAR) | Missingness of values does not depend on the column itself or other columns. |

To start looking at missing values, I will identify which columns have missing values.
```python
new_recipes.isna().sum()[new_recipes.isna().sum() != 0]
```
results in
```
name              1
description      70
avg_rating     2609
cal_bins         27
dtype: int64
```


### NMAR (Not Missing At Random) Analysis

`avg_rating` may be NMAR. From the data cleaning process, I filled in the `avg_rating` of 0 with `np.nan`, thus "creating" missing values in this column artificially. However, this step is reasonable, and was justified in the data cleaning step. We can reason that reviewers may be more likely to provide star ratings to recipes that they either enjoyed or hated. Therefore, the missingness of `avg_rating` may be dependent on the star rating itself.

If food.com were to change their reviewing process and added a required question like "How much did you like the recipe", where the options are thumbs up, thumbs sideways, and thumbs down, I think `avg_rating` may become MAR. If we assume my reasoning for `avg_rating` being NMAR is correct, then the "0 star rating" should be more associated with the "thumbs sideways" response than the "thumbs up" or "thumbs down" response.


### Dependency Analysis

I will attempt to find depndency in missingness in the `calories (#)` columns by performing two permutation tests. The significance level will be 0.05.

Null Hypotheses: The distributions of `cal_bins` / `n_ingredients` are **the same** when `avg_rating` is missing and not missing.

Alternative Hypotheses: The distributions of `cal_bins` / `n_ingredients` are **not the sam**e when `avg_rating` is missing and not missing.

The significance level will be set at 0.05.

#### Permutation Testing

I have two permutations that I want to attempt: one that will shuffle the `cal_bins` column, and another one that will shuffle the `n_ingredients` column. 

1. Is avg_rating missingness dependent on the `calories (#)` column?
    <iframe src = 'assests/miss_dist_cal.html' width = 800 height = 800 frameborder = 0> </iframe>
    The distributions of calories (`cal_bins`) by Missingness of avg_rating (graph only shows calories >= 1800)
    - Test statistic: **Total Variation Distance**
        - In order to use TVD as the test statistic, I converted the `calories (#)` into a categorical variable using bin width of 10 calories.
        - Observed statistic: 0.117
    - Method: Permutating the `missing_rating` column, and calculating the test statistic after each run
    <iframe src = 'assests/cal_p_result.html' width = 800 height = 400 frameborder = 0> </iframe>
    Result of running the permutation test 10000 times
    - Result:
        - a slightly right-skewed unimodal distribution with a center at around 0.08
        - **a p-value of 0**.
    - Conclusion
        - Since p-value is <0.05, we reject the null hypothesis
        - **We have sufficient evidence to suggest that the missingness of `avg_rating` column is dependent on the `cal_bins` column (and therefore the `calories (#)` column).**

2. Is the avg_rating missingness dependent on the `n_ingredients` column?
    <iframe src = 'assests/miss_dist_ing.html' width = 800 height = 800 frameborder = 0> </iframe>
    <figcaption>n_ingredients by Missingness of avg_rating
    - How to read the graph
        - For example, for n_ingredients = 5, there is a 8.6% chance of a missing rating, and a 7
    - Test statistic: **Total Variation Distance**
        - Observed statistic: 12.76
    - Method: Permutating the `missing_rating` column, and calculating the test statistic after each run
    <iframe src = 'assests/ing_p_result.html' width = 800 height = 400 frameborder = 0> </iframe>
    Result of running the permutation test 10000 times
    - Result:
        - a multimodal distribution with a center at around 12
        - **a p-value of 0.1819**.
    - Conclusion
        - Since p-value is >0.05, we fail to reject the null hypothesis 
        - **We do *not* have sufficient evidence to suggest that the missingness of `avg_rating` column is dependent on the `n_ingredients` column.**

##### Result

We found that there is strong evidence to suggest that `avg_rating` is MAR, dependent on `calories (#)`, but is likely not dependent on `n_ingredients`. This means that reviewers are more likely to review recipes, but not leave a star rating (thus, the "0 stars rating") on recipes that have higher calories. I cannot come up with a reasonable explanation for this correlation; it may be the case that `calories (#)` is a proxy for some other metric, and the `avg_rating` is actually NMAR on that column, and appears NAMR dependent on `calories (#)` just because of the proxy relationship.

------

## Hypothesis Testing

With the power of hypothesis testing, I will try to answer the original question: what kinds of recipes have the highest calories? Recall how I plotted the tags with the top 20 highest median calories? We will go back to the `tag` column to help us answer this quetion.
The tag associated with the highest median calories was `pork-ribs`. Therefore, I will construct a hypothesis test to determine if there is an association betwen pork-ribs and calories. 

***Null Hypothesis***: `calories (#)` and `pork-ribs` tag are **not related** - the high median `calories (#)` of recipes with `pork-ribs` tag is **due to random chance alone**.
***Alternative Hypothesis***: `calories (#)` and `pork-ribs` tag **are related** - the high median `calories (#)` of recipes with `pork-ribs` tag is **not due to random chance alone**.

The significance level will be set at 0.05.

### Setup

Test statistic: Median calories
- Observed statistics (median calories for recipes with `pork-ribs` tag): 790.1
Method: Randomly draw a `calories (#)` value 209 times and compute the median 1,000,000 times.
- The sample size is 209 because there are 209 recipes with the `pork-ribs` tag

### Result
<iframe src = 'assests/hyp_test_result.html' width = 800 height = 400 frameborder = 0> </iframe>
Result of 1,000,000 Runs

- Result
    - p-value of 0.0. We **reject** the null hypothesis.
    - This means that: the `calories (#)` and `pork-ribs` are likely to be related - it is unlikely that results this extreme (p-val of 0) would appear by pure chance. Since the observed statistic was all the way to the right compared to the empircal distribution of median calories (refer to graph below), recipes with the tag `pork-ribs` do indeed seem to have higher calories, compared to the calories distribution of the entire recipes dataset (but we can never say for sure!).

------

## Conclusion

### So, what kinds of recipes should we avoid?
Recipes for pork ribs seem have to have a tendency for higher calorie values. If you are on a diet, make sure to avoid pork rib dishes! :smile:

### General Findings
- "0 star ratings" are missing values - it happens when a reviewer does not add a star rating
- Many of the distrubutions, including `calories (#)` and `n_ingredients` are right skewed in this dataset.
- Somewhat unsurprisingly, total fat and calories have a positive correlation. This is likley the case for other nutritional values as well.
- Meat dishes and spaghetti dishes seem to have the highest overall calories, although we do not have evidence to support this, becuase a hypotheiss test was only conducted for `pork-ribs`.
- The missinginess of `avg_rating` seems to be dependent on `calories (#)`, although it is hard to pinpoint why it would be dependent on calories. As mentioned in the missingness section, it may be the case that `calories (#)` is a proxy for some other factor which `avg_rating` is truly dependent on; it just seems like `avg_rating` is dependent on `calories (#)`.

### Closing Notes
I wish I had a column for serving size, so I could normalize the various metrics of recipes, such as `calories (#)` or the other nutritional values. This would have been very useful, as I would have been able to more effectively compare the calorie counts accross different recipes, and perhaps identified a better indicator for high calorie recipes. 

Thank you!
