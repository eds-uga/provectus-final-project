# Click Through Rate Prediction

###Overview

User responses i.e. Click Through Rate is a critical part of many web applications including web search, personalised recommendation and online advertising. Click Through Rate(CTR) measures the response of a user towards an advertisement. The motivation of the project come from a Kaggle competition named Display Advertising Challenge of 2014.The following python scripts are used for predicting:

- __For Random-forest__
  - preprocessing.py: This file preprocess the input data.
  - features_VectorAssembler.py: Creates a vector reprensentation for the data.
  - random_forest.py: This python script creates the classifier model using the output of pipeline.py
  - rf_transform_model.py: This file predicts the result using the random forest model.

- __For Tensor-Flow__
  - tensorflow_precossing.py: This file preprocess the input data.
  - tf_wide_and_deep.py: Creates a model using the pre-processed data.
  - wide_deep_evaluate_predict.py: Gives prediction using the learned model.
 
- __For Gradient Boosted Decision Trees__
  - gb_clasiifier.py : This file takes the pre-processed file and predict the results.

###Problem Description

The objective of the project was to predict whether an adversiment will be clicked, based on the traffic logs. The dataset was publicly available and provided by Criteo lab. According to the description, the data was a week's data traffic. Data was conprised of two types of features: categorical and continuos. The training file consist of around 45 million records whereas we performed testing on 6 million records.

The approach followed for this project involved three machine learning algorithm: TensorFlow's Wide and Deep learning, Random Forest and Gradient Boosted Decision Trees. We followed different feature enineering techniques for each algorithm. Detailed description of each is given in project report. Finally, the results are compared.

### __How to Run__

- __Random-forest__
  - Run pipeline.py rf \<source:input-file> \<destination:pre-processed-file>
  - Run random_forest \<source:pre-processed-file> \<destination:model-file>
  - Run rf_transform_model \<source:model-file>

- __Tensor-Flow__
  - Run tensorflow_precossing.py
  - Run tf_wide_and_deep.py
  - Run wide_deep_evaluate_predict.py
 
- __Gradient Boosted Decision Trees__
  - Run gb_clasiifier.py \<source-input-file>

###Project Report

Please refer dsp_final_report.pdf for detailed information of the project implementation.
