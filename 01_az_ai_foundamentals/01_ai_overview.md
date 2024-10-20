
# fundamental AI concepts
## introduction to AI
## understand machine learning
## understand computer vision
## understand nlp
## understand doc intelligence and knowledge mining
## understand gen ai
## challenges and risk with AI
## understand responsible AI
## knowledge check

# fundamentals of machine learning
## what is machine learning
origins in statistics and mathematical modeling of data 
- statistics, refer to study and analysis data, where statistic methods are used to draw conclusions or make inferences 
- mathematical modeling of data, create math representations of real-world processes or systems to analyze and predict outcomes based on data
the fundamental idea of ml is to use data from past observations to predict unknown outcomes or values. 
### ml a a function 
a ml model is a software app that encapsulates a <u>function</u> to calculate an output value based on one or more input values. <br>
the process of defining that function is known as <u>training</u> <br>
use the function to predict new values in a process called <u>inferencing</u><br>
steps involved in training and inferencing:
1. training data consists of past observations. include observed attributes(features) of the thing being observed; and known value of this thing you want to train a model to predict (know as label)
in math terms, x-> attributes(features); y->label; usually, an observation consists of multiple feature values. so x is actually a vector, like [x1,x2,...xn]
2. an <u>algorithm</u> is applied to the data to try to determine a relationship between the features and the label
3. the result of the algorithm is a <u>model</u> that encapsulates the calculation derived by the algorithm as a function. y=f(x)
4. now the <u>training</u> phase is complete, the trained model can be used for <u>inferencing</u> -> ŷ (y-hat)

## type of machine learning
![types](../imgs/machine-learning-types.png)
### supervised ml
training data includes both feature values and known label values.
- regression, lable predicted by model is a number value. 
- binary classification, the label determines whether observed item is (or isn't) an instance of a specific class (exclusive outcome)
- multiclass classification, predict a label that represents one of multiple possible classes.  
### unsupervised ml
training data consists only feature values without any known labels
- clustering, identifies similarities between observations based on their features, and groups them into discrete clusters. 
<i>discrete: distinct group, no gradation, clear boundaries</i><br>
multiclass vs clustering; known label for multiclass, unknown lable for clustering  

## regression
what: regression models are trained to predict numeric label values based on training data that includes both features and known labels.
how: multiple iterations. use appropriate algorithm to training a model, evaluate model's predictive performance, and refine the model by repeating the training process with different algorithms and parameters until achieve an acceptable level of predictive accuracy 
![training process](https://learn.microsoft.com/en-us/training/wwl-data-ai/fundamentals-machine-learning/media/supervised-training.png)
1. split training data (randomly) to subset a (for training) and another subset (for validation)
2. use an algorithm to fit the training data to a model
3. use validation data to test the model by predicting labels for the features 
4. compare the known actual labels in validation subset to the labels the model predicted. then aggregate the differences between predicted and actual label values to calculate metric (accurate)

regression evaluation metrics (ignore)
- mean absolute error (mae)
- mean squared error (mse)
- root mean squared error (rmse)
- coefficient of determination (r<sup>2</sup>)

## binary classification
train classification model calculate <u>probability</u> values for class assignment
metrics: compare predicted classes to the actual classes<br>
binary classification are used to train a model predicts one of two possible labels for a single class. label(y) is either 1 or 0.

binary classification evaluation metrics 
confusion matrix
- ŷ=0 and y =0, True negatives (TN)
- ŷ=1 and y =0, False positives (FP)
- ŷ=0 and y =1, False negative (FN)
- ŷ=1 and y =1, True positive (TP)

accuracy = (TN + TP) / (TN + TP + FN + TP); proportion of predications that model got right
recall = TP / (TP + FN); proportion of positive case that model identified correctly
precision = TP / (TP + FP); proportion of predicted positive cases where the true label is actually positive
f1 = (2 * Precision * Recall) / (Precision + Recall)

## multi-class classification
predict to which of multiple possible classed an observation belongs

### training a multiclass classification model
fit training data to a function that calculates a probability value for each possible class.
- One-vs-Rest algorithm
- Multinomial algorithm 

## clustering
## deep learning
## azure machine learning
## exercise - explore automated machine learning in az machine learning 
## knowledge check 

# ai service on az
## create az ai service resources
## use az ai services
## understand authentication for az ai service
## exercise - explore az ai services
## knowledge check