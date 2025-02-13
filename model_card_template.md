# Model Card
## Model Details
This model was created 2/17/2025 by Jaxon Goble. 

It is logistic regression using the following hyperparameters:
max_iter=6000, 
solver='lbfgs', 
random_state=14

## Intended Use
This model is being used to predict whether or not an individual had a salary of more than or less than $50K a year using key features from U.S. census data.

## Data
The original dataset was obtained from UC Irvine using the following link:
https://archive.ics.uci.edu/dataset/20/census+income

This dataset is a subset of the larger U.S. Census from 1994. It contains 32561 rows, and 15 features.

### Training Data
The training data is an 85% subset of the original dataset created by using sklearn.test_train_split(). 

### Evaluation Data
The test data is the remaining 15% subset of the original dataset that isn't included in the training data. 

## Metrics
Three metrics were used to evaluate model performance:
1. Precision: 0.6971 (Model Performance)
2. Recall: 0.5951 (Model Performance)
3. F1: 0.6421 (Model Performance)

## Ethical Considerations
Dataset is from 1994, and also is a limited subset of the collected census data. Present-day application will inherently be subject to the biases inherent to data from a significantly different time period, and therefore, should not be applied for any present day prediction of outcomes in real-world practice.

## Caveats and Recommendations
The number of maximum iterations should be increased for improvement in model performance. It has been set lower than optimal due to local computer capabilities.