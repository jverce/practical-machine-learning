Introduction
------------

The purpose of this document is to describe the process of creating a machine learning model to predict the class of exercise from certain motion variables. The original experiment is called **Human Activity Recognition** and more information can be found here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>

The experiment basically took place by adding several motion detectors in different parts of the human body and recording their output during specific gym routines. These routines where performed by different subjects and in a supervised way. The main goal of the original experiment is to determine which exercise a user is performing and whether or not it's properly performing it.

In our case we'll be using the training data acquired in the original experiment and try to build a machine learning model that predicts the class of exercise being performed based on the motion variables.

Data Cleanup
------------

In this section we'll be performing a data cleanup since the original files contain a lot of undefined information that can negatively impact our analysis.

### Load Data

We start by loading both the training and testing data and storing it in the `training` and `testing` variables, respectively:

``` r
training <- read.csv('./pml-training.csv')
testing <- read.csv('./pml-testing.csv')
```

### Glimpse

We take a quick glimpse at our data to find some inconsistent/improper variables to discard. *Keep in mind that we're hiding the results from this report since it's too verbose, but they're still reproducible*.

``` r
head(training)
```

``` r
summary(training)
```

Based on the output we can find the following variables to remove:

-   `X`: it's a variable that indicates the record number and should definitely be removed prior to our analysis
-   `user_name`: similar to the previous variable, the user name should not be used as a predictor for a general case.
-   `new_window`: this variable presents inconsistencies in its values and it's better to leave it out.
-   Timestamp variables: these are variables that contain `timestamp` in their name. They do not provide information about the subject's movement and so they shouldn't affect our prediction.
-   `NA` variables: these variables have undefined values and since we don't have any way to supply them we have no option but to remove them.
-   `#DIV/0!` variables: looks like variables that experienced some strange issue during their recording and we can't rely on their values, hence we'll be removing them.

### Variable Removal

As per the previous section, we proceed to remove the mentioned variables:

``` r
# Remove `X` var
training <- subset(training, select = -c(X))

# Remove `user_name` var
training <- subset(training, select = -c(user_name))

# Remove `new_window` var
training <- subset(training, select = -c(new_window))

# Remove timestamp vars
training <- subset(training, select = -grep('timestamp', colnames(training)))

# Remove `NA` and `#DIV/0!` vars
for (col in names(training)) {
  nas <- sum(is.na(training[, col])) > 0
  div0 <- any(grep("DIV/0", training[, col]))
  if (nas || div0) {
    training <- subset(training, select=(!names(training) %in% c(col)))
  }
}
```

Finally we see how many variables are left in our training set:

``` r
dim(training)
```

    ## [1] 19622    54

Build Model
-----------

To build our machine learning model we'll be using the *random forest* algorithm since it provides a fair amount of accuracy. The only aspect we need to be aware of is the possibility of overfitting. We can play around with the `mtry` variable in order to balance between the accuracy against the training set and the amount of variables sampled as candidates: we can reduce overfitting and simplifying the model at the same time by reducing the `mtry` value we provide as input.

We try with the following `mtry` values:

-   2 (minimum)
-   27 (half the max)
-   54 (maximum)

``` r
library(randomForest)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
randomForest(classe ~ ., data = training, mtry = 2)
```

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = training, mtry = 2) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 0.28%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 5579    1    0    0    0 0.0001792115
    ## B    6 3790    1    0    0 0.0018435607
    ## C    0   13 3409    0    0 0.0037989480
    ## D    0    0   28 3187    1 0.0090174129
    ## E    0    0    0    4 3603 0.0011089548

``` r
randomForest(classe ~ ., data = training, mtry = 27)
```

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = training, mtry = 27) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.15%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 5578    1    0    0    1 0.0003584229
    ## B    4 3790    3    0    0 0.0018435607
    ## C    0    5 3416    1    0 0.0017533606
    ## D    0    0    8 3207    1 0.0027985075
    ## E    0    0    0    5 3602 0.0013861935

``` r
randomForest(classe ~ ., data = training, mtry = 54)
```

    ## Warning in randomForest.default(m, y, ...): invalid mtry: reset to within
    ## valid range

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = training, mtry = 54) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 53
    ## 
    ##         OOB estimate of  error rate: 0.29%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 5576    2    1    0    1 0.0007168459
    ## B   17 3774    5    1    0 0.0060574137
    ## C    0    6 3413    3    0 0.0026300409
    ## D    0    1   13 3201    1 0.0046641791
    ## E    0    1    0    5 3601 0.0016634322

We can see that the best performance is achieved with `mtry = 27`. Since the value is below the max number we're basically discarding variables which reduces the chance of overfitting.

We can now do a prediction to check if the output looks consistent:

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

    ## Warning in as.POSIXlt.POSIXct(Sys.time()): unknown timezone 'zone/tz/2017c.
    ## 1.0/zoneinfo/America/Vancouver'

``` r
fit <- randomForest(classe ~ ., data = training, mtry = 27)
predict(fit, testing)
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

As per the **final quiz**, the result is 100% accurate.

Conclusions
-----------

As seen before, it is very important to perform a data inspection prior to any analysis since there might be dummy, dirty or auxiliary variables that are not consistent with the processing and information we need to extract.

We observed that by applying the random forest algorithm we were able to obtain a prediction model that is accurate enough without overfitting the initial training data. Even though we used the `mtry` parameter to balance the precision and complexity of the model, there are other parameters that can be adapted as well (check the official `randomForest` [documentation](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf) for more information).

Finally we used the obtained model to perform a prediction on the tesing dataset with 100% accuracy according to the final quiz of the course.
