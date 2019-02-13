# Project 1: Predictive machine learning
This assignment introduces you to machine learning. Machine learning is cool
because we can predict the future. Actually machine learning is just a fancy
word for how machines can use old data to predict new data. The simplest
predictions are linear and uses the formula you probably know from your
high-school math:

    y = ax + b

To predict the outcome of a linear model you need to know the ``a`` and ``b``,
so you can 'predict' the ``y`` value, when you input an ``x`` value.

If you want to become a bit fancier, you can use a linear model with *two* inputs (`x1` and `x2`):

    y = ax1 + bx2 + c

This will make our predictions even more precise, because we have one more explanatory factor to find out what the right value of `y` should be.

## How to hand in
The hand-in (on Moodle) should be a link to a GitHub **release** containing a
**single** Jupyter Notebook file (`.ipynb`) with the code, plots and written text for the 
assignment parts
  - The file must be clearly identifiable. Please name it accordingly.
    (for instance ``project1.ipynb``)

# Part 1: Predicting Hackernews points with linear regression
For this part we would like to predict how long a user has to be active
on Hackernews to achieve 1000 points. In this experiment our input variable is
time and our output variable is points. X (time) and Y (points).

### Part 1.1: Data preprocessing
The data can be found in json format in the file [``users.json``](https://github.com/datsoftlyngby/soft2019spring-ai/tree/master/projects/project1/users.json) under the
``project1`` folder here: [https://github.com/datsoftlyngby/soft2019spring-ai/tree/master/projects](https://github.com/datsoftlyngby/soft2019spring-ai/tree/master/projects).

Your job is to build and train a model that predicts how long new and hopeful users 
will have to work to accumulate 1000 points. 
Before you can know what model to use, have a look at the data: load it into a Pandas DataFrame 
and plot it on a graph. Remember that your first axis is time (since registration on Hackernews) and the second
axis is number of points.

Hand-in a plot that shows time on the x-axis and number of points on the y-axis.

### Part 1.2: Model selection
The next part is to select an appropriate model. Study your data either visually or 
by going through the .csv file; what are you finding? What kind of model would it fit?

You can assume that models existing in `sklearn` is appropriate. If you don't know what
to use, we recommend a linear regression model.

Describe with at least two lines of text which model you chose and why. Also if you chose
the linear model.

### Part 1.3: Model training
Next you'll have to train your model. 
This requires you to create the model using `sklearn`, then feed it data to *train* it.

Hand-in the code that trains the data.

### Part 1.4: Model evaluation

Now that you have a trained model we need to evaluate it. This should be done both
intuitively and visually. 

Hand-in:
1. A description of the model parameters
  * Why are they there? What is their purpose in the model?
  * What do they explain for your data? How are they helping you to predict how much 
    time you need to be active in order to gain Hackernews points?
2. A visual plot of the model that explains the description above.
  * In the case of a linear regression this is a line that shows how the model expects 
    how long a user needs to be active (x value) in order to get a certain number of points (y value).
3. Explain: What does it take to get 1000 points? How long would you have to be active on Hackernews?

# Part 2: Predicting Hackernews points with multivariate linear regression
So far you pretended to be a Hackernews user with the only
goal of reaching 1000 points. But our model was pretty limited. We only looked
at time as the input variable. Who knows, perhaps it depends on other things
as well?

Linear regression is great for predicting inputs and outputs. But the linear
model we discussed in class:

    y = ax + b

only include a single input variable. What if we want to look at something else?
Like the number of posts a user made? Then the model could look like this:

    y = ax_1 + bx_2 + c

And what if we want to include one more thing? Like number of upvotes?

    y = ax_1 + bx_2 + cx_3 + d

And so on. Your model is still linear, but now it contains many variables. If
you think about it, each variable is the same as a dimension in space. So in
the model where time predicted points, you had a 2-dimensional plot. If you
include the number of posts you have a 3-dimensional.

Going back to the Hackernews data set from the last assignment, use a linear model, where you use the time (x-axis) to predict the number of points (y-axis). But, in this model, include the number of posts as an addition to the predictive value of time.
That is, instead of only using time to predict number of points, use time **and** number of posts to predict the number of points.

Hand-in:
* A description of the parameters
* An analysis of how much time you need to be active on Hackernews, plus how many posts you need to reach 1000 points.
* Explain: how much does the number of posts contribute to get the points? How much do the time since registration explain?

# Part 3: Prediction of breast cancer with logistic regression
In our previous examples we tried to predict a numerical variable, namely
points. Points can have values from 0 and upwards, in a continuous space.
But what if we work with things that aren't continuous? Like grades
(-3, 0, 2, 4, 7, 10, 12) or cars
(Nissan, Toyota, Tesla...) or perhaps genders (male, female, other)?

To do that we need logistic regression. Logistic regression allows us to predict
things that are not a continuous number, but instead divided into categories.
This putting things into categories is also referred to as classification.

We will be using a new dataset based on
[breast cancer patients](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) in Wisconsin,
USA, and our goal is to _predict whether cancer is benign or malign_. One or
the other.

## 3.1: Looking at the data
Load the data from this ``.csv``:
[http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data).
Report the ``head`` of the table. Have a
look at the data before you get started with the analysis.
How big is your dataset? What are the different variables?

Write two lines about notable features of the data.

## 3.2: Building the model
Our goal is to predict whether the patients have benign or malign cancer.
For this part you can choose the input variables, but use at least two:
which ones do you think have the most impact?

Hand-in: A description of your chosen variables and why you think they could help predict breast cancer. 

## 3.3 Evaluating the model
Now take the data from dataset and train your logistic regression model. 
To figure out how good your model was, use the function here
``sklearn.metrics.accuracy_score``. 
It gives you an accuracy that tells you how good your model is 
able to predict whether a tumor is benign or malignant.

Hand-in: The accuracy of the score as well as two lines describing what that number actually mean. What can the number tell you when you inform patiens whether their cancer is benign or malign?
