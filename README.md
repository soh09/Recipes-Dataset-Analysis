# Recipes Dataset Analysis: Beat the Calories!
by So Hirota (hirotasp92602@gmail.com)

------

## Introduction

### Understanding the Datasets
The recipes dataset contains two .csv files: the RAW_recpies and the RAW_interactions dataset.
* RAW_recipes.csv contains `83782 rows` and `12 columns`. The rows represent the recipes, and the columns contain `name`, `id`, `minutes`, `contributor_id`, `submitted`, `tags`, `nutrition`, `n_steps`, `steps`, `description`, `ingredients`, `n_ingredients`.

| column name | meaning ||---|---|
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


* RAW_interactions.csv contains `731927 rows` and `5 columns`. The rows represent an individual review of a recipe, and the columns contain `user_id`, `recipe_id`, `date`, `rating`, `review`.

| columns name | meaning ||---|---|
| `user_id` | user id of the user who posted a review |
| `recipe_id` | recipe id for the review, same as the ones in RAW_recipes.csv |
| `date` | the date that the reivew was posted |
| `rating` | the star rating of the recipe, from 1 - 5 |
| `review` | the text review of the recipe |


### Question: What types of recipes have the most calories?
The analysis in this notebook will be centered around this one question. The columns that may be relevant to the analysis include `mintues`, `nutrition` (this contains the calorie information), `tags`, `n_steps`, `n_ingredients`, `ingredients`, and `rating.`
### Why care?

By investigating this question, a person attempting a diet may be able to avoid high calorie recipes based on the key factors that correlated with high caloric recipes. 

------


## Cleaning and EDA



------


## Assessment of Missingness




------



## Hypothesis Testing

