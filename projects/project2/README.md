## Project 2: Pipelines and optimisations

This assignment still focuses on predictive machine learning, but starts to spread
into the area of data preprocessing and model optimisation.
You will still be working with supervised classification tasks, but start to use more 
powerful machine learning tools from `sklearn` and the data processing library `pandas`.
We will also be asking you to work on a completely new type of data (voice) and 
reflect on the external validity of your model.

The data is based on the [Kaggle](https://kaggle.com) dataset [Gender Recognition by Voice](https://www.kaggle.com/primaryobjects/voicegender).

### Part 1: Data exploration
Your first task is to explore the data. What features are there?
How are they related?
Hand two lines that describes
* what a frequency is
* what the median frequency means
* what the output label is

### Part 2: Data preparation
When we train our model we'll use a 10-fold `KFold` 
cross-validator **with** shuffling.
Instantiate a `KFold` class and store it in a meaningful variable.

When that is done, illustrate that you indeed do get 10 
iterations of your data by iterating over the folds and simply
printing *the shape of* the four variables: `x_train`, `y_train`, `x_test` and
`y_test` (see 
the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) for examples on
how to do this).

Hand in
* the instantiation of the k-fold cross-validator
* a loop that prints the *shape* of `x_train`, `y_train`, `x_test` and `y_test`

###  Part 3: Model construction
We will use three different classification models for this task:
1. Logistic Regression
2. Support Vector Machine classifier
3. Decision Tree classifier

Instantiate the three different classifiers in *three different 
pipelines*.
For now the default parameters are fine.

Hand in 
* the code for constructing the three pipelines
* one line of text per model describing how you think the classifier will perform, given the data type you are working with (voice)

## Part 4: Model validation
Now the time comes to train and validate your model.
This training and testing **should happen for all three models**.
The easiest way to do this is to use the `cross_val_score` 
function from `sklearn` once for all the three models.
The code should look something like this:
```python
pipeline1 = ...
pipeline2 = ...
pipeline3 = ...
my_kfold_validator = ...
for model in [pipeline1, pipeline2, pipeline3]:
    score = cross_val_score(model, xs, ys, cv=my_kfold_validator)
    print(score)
```

Hand in
* a list per model (three lists in total) of 10 values each, 
  showing the scores of the 10 folds,
* at least one line of text that describes what the 'score' means
* at least one line of text that describes why the scores are different

## Part 5: Model optimisation: scaling
On the website of the [Gender Recognition by Voice](https://www.kaggle.com/primaryobjects/voicegender) dataset, they say
we can do better. So let's try!

One thing that's very easy to do is to use a 
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
It's particularly easy, because it fits right into your existing
pipelines. So simply add three (separate!) instances of the
`StandardScaler` to the pipelines, one for each pipeline.

Now repeat the above validation code, where you run the 
`cross_val_score` for *each* of the three pipelines. But this 
time the `StandardScaler` is included in the pipeline.

Hand in
* the code for your new pipelines that includes the `StandardScaler`)
* at least one line of text that describes what scaling actually is
* the **mean** of the 10 scores of the three models (this time it's only **one** number per model
* at least two lines of text describing which model performed well, and whether this aligned with your expectation from part 3