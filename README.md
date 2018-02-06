Introduction
------------

The purpose of this document is to describe the process of creating a
machine learning model to predict the class of exercise from certain
motion variables. The original experiment is called **Human Activity
Recognition** and more information can be found here:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>

The experiment basically took place by adding several motion detectors
in different parts of the human body and recording their output during
specific gym routines. These routines where performed by different
subjects and in a supervised way. The main goal of the original
experiment is to determine which exercise a user is performing and
whether or not it's properly performing it.

In our case we'll be using the training data acquired in the original
experiment and try to build a machine learning model that predicts the
class of exercise being performed based on the motion variables.

Data Cleanup
------------

In this section we'll be performing a data cleanup since the original
files contain a lot of undefined information that can negatively impact
our analysis.

### Load Data

We start by loading both the training and testing data and storing it in
the `training` and `testing` variables, respectively:

    training <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')
    testing <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv')

We also take the chance to load the libraries now:

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning in as.POSIXlt.POSIXct(Sys.time()): unknown timezone 'zone/tz/2017c.
    ## 1.0/zoneinfo/America/Vancouver'

    library(randomForest)

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

### Glimpse

We take a quick glimpse at our data to find some inconsistent/improper
variables to discard. *Keep in mind that we're hiding the results from
this report since it's too verbose, but they're still reproducible*.

    colnames(training)
    colnames(testing)

We can see that the `classe` variable is missing from the testing set,
which means that we won't be able to evaluate the models we create
unless we partition the training set and work with that.

    head(training)

    summary(training)

Based on the output we can find the following variables to remove:

-   `X`: it's a variable that indicates the record number and should
    definitely be removed prior to our analysis
-   `user_name`: similar to the previous variable, the user name should
    not be used as a predictor for a general case.
-   `new_window`: this variable presents inconsistencies in its values
    and it's better to leave it out.
-   Timestamp variables: these are variables that contain `timestamp` in
    their name. They do not provide information about the subject's
    movement and so they shouldn't affect our prediction.
-   `NA` variables: these variables have undefined values and since we
    don't have any way to supply them we have no option but to
    remove them.
-   `#DIV/0!` variables: looks like variables that experienced some
    strange issue during their recording and we can't rely on their
    values, hence we'll be removing them (we can identify them as
    "factor" variables).

### Variable Removal

As per the previous section, we proceed to remove the mentioned
variables:

    # Remove `X` var
    training <- subset(training, select = -c(X))

    # Remove `user_name` var
    training <- subset(training, select = -c(user_name))

    # Remove `new_window` var
    training <- subset(training, select = -c(new_window))

    # Remove timestamp vars
    training <- subset(training, select = -grep('timestamp', colnames(training)))

    # Remove `NA` vars
    training <- training[, (colSums(is.na(training)) == 0)]

    # Remove `#DIV/0!` vars (be careful not to remove the output var which is a factor)
    predictors <- which(lapply(training, class) %in% c('integer', 'numeric'))
    y <- which(colnames(training) == 'classe')
    training <- training[, c(predictors, y)]

Finally we see how many variables are left in our training set:

    dim(training)[2]

    ## [1] 54

We now proceed with the data partition:

    inTrain <- createDataPartition(training$classe, p = .7, list = FALSE)
    trainingSet <- training[inTrain, ]
    testingSet <- training[-inTrain, ]

Build Model
-----------

To build our machine learning model we'll be using the *random forest*
algorithm since it provides a fair amount of accuracy. The only aspect
we need to be aware of is the possibility of overfitting. We can play
around with the `mtry` variable in order to balance between the accuracy
against the training set and the amount of variables sampled as
candidates: we can reduce overfitting and simplifying the model at the
same time by reducing the `mtry` value we provide as input.

We try with the following `mtry` values:

-   2 (minimum)
-   27 (half the max)
-   53 (maximum)

<!-- -->

    randomForest(classe ~ ., data = trainingSet, mtry = 2)

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = trainingSet, mtry = 2) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 0.43%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3905    0    0    0    1 0.0002560164
    ## B    8 2647    3    0    0 0.0041384500
    ## C    0   11 2382    3    0 0.0058430718
    ## D    0    0   27 2224    1 0.0124333925
    ## E    0    0    0    5 2520 0.0019801980

    randomForest(classe ~ ., data = trainingSet, mtry = 27)

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = trainingSet, mtry = 27) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.23%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3904    1    0    0    1 0.0005120328
    ## B    5 2650    3    0    0 0.0030097818
    ## C    0    3 2392    1    0 0.0016694491
    ## D    0    0   10 2242    0 0.0044404973
    ## E    0    1    0    6 2518 0.0027722772

    randomForest(classe ~ ., data = trainingSet, mtry = 53)

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = trainingSet, mtry = 53) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 53
    ## 
    ##         OOB estimate of  error rate: 0.44%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3900    2    1    0    3 0.001536098
    ## B   15 2636    6    1    0 0.008276900
    ## C    0    1 2391    4    0 0.002086811
    ## D    1    0   15 2235    1 0.007548845
    ## E    0    2    1    8 2514 0.004356436

We can see that the best performance is achieved with `mtry = 27`.
However, even with `mtry = 2` we have a great performance as well, with
a simpler processing. Hence we'll be choosing this last option for our
algorithm.

We can now do a prediction to check if the output looks consistent:

    fit <- randomForest(classe ~ ., data = trainingSet, mtry = 2)
    predictions <- predict(fit, testingSet)

    cm <- confusionMatrix(predictions, testingSet$classe)
    print(cm)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    5    0    0    0
    ##          B    0 1133    7    0    0
    ##          C    0    1 1019   20    0
    ##          D    0    0    0  944    1
    ##          E    0    0    0    0 1081
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9942         
    ##                  95% CI : (0.9919, 0.996)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9927         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9947   0.9932   0.9793   0.9991
    ## Specificity            0.9988   0.9985   0.9957   0.9998   1.0000
    ## Pos Pred Value         0.9970   0.9939   0.9798   0.9989   1.0000
    ## Neg Pred Value         1.0000   0.9987   0.9986   0.9960   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1925   0.1732   0.1604   0.1837
    ## Detection Prevalence   0.2853   0.1937   0.1767   0.1606   0.1837
    ## Balanced Accuracy      0.9994   0.9966   0.9944   0.9895   0.9995

We can calculate the **out-of-sample** error based on the confusion
matrix data we just obtained:

    1 - cm$overall[[1]]

    ## [1] 0.0057774

As predicted, this error is very low so we're confident about performing
predictions in the original test set. This can also be verified with
against the **final quiz** of the course:

    predict(fit, testing)

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

As per the **final quiz**, the result is 100% accurate.

Conclusions
-----------

As seen before, it is very important to perform a data inspection prior
to any analysis since there might be dummy, dirty or auxiliary variables
that are not consistent with the processing and information we need to
extract.

We observed that by applying the random forest algorithm we were able to
obtain a prediction model that is accurate enough without overfitting
the initial training data. Even though we used the `mtry` parameter to
balance the precision and complexity of the model, there are other
parameters that can be adapted as well (check the official
`randomForest`
[documentation](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf)
for more information).

Finally we used the obtained model to perform a prediction on the tesing
dataset with 100% accuracy according to the final quiz of the course.
