# ML models for RSRP prediction: #
<br>
	
*	Model Training:  
Four models are trained:  
	•	Linear Regression using fitlm  
	•	Support Vector Regression (SVR) using fitrsvm  
	•	Random Forest using TreeBagger (with out-of-bag prediction enabled)  
	•	Neural Network Regression using fitrnet  

*	Data Splitting:
The data is split into training (80%) and test (20%) sets using MATLAB’s cvpartition.
*	Evaluation:
Predictions are made on the test set for each model, and the RMSE (Root Mean Squared Error) is calculated to compare performance.
*	Visualization:
Scatter plots compare actual RSRP values versus predicted RSRP values for each model. The reference line y = x helps visualize prediction accuracy.
<img src="ML_models/ML_models_comparison.jpg" width="300"> 
<img src="ML_models/ML_models_error_comparison.jpg" width="300">

# performance

random forest performs better than other three algos. reason being more number of decision trees are used in random forest. can be configured to match optimum error while retaining performance. Neural networks on other hand perform better but can be improved by increasing more number of hidden layers.

Below graphs illustrate error histogram and number of epochs required for obtaining minimum MSE.
hyperparamters used for fine tuning in random forests is number of decision trees.

<img src="ML_models/nn_training_state.jpg" width="300">
	•	First Plot (Gradient vs. Epochs)
	•	The gradient is the rate of change of the error function.
	•	A decreasing gradient shows the model is converging.
	•	Second Plot (Mu vs. Epochs)
	•	Mu (learning rate adaptation factor for Levenberg-Marquardt training) changes over epochs to optimize convergence.
	•	Third Plot (Validation Failures vs. Epochs)
	•	Shows when validation performance did not improve.
	•	If failures reach the validation threshold (here, 6), training stops.


<img src="ML_models/nn_error_histogram.jpg" width="300">
	•	The plot is divided into:
	•	Training data errors (Blue)
	•	Validation data errors (Green)
	•	Test data errors (Red)
	•	The orange vertical line represents zero error, meaning perfect predictions.
	•	The histogram shows that most errors are centered around zero, indicating a well-trained network, though some errors are significantly high.  
 
<img src="ML_models/nn_best_validation_performance.jpg" width="300">
Here, The plot shows how the MSE reduces over time.
	•	The training stopped at epoch 166 because of validation performance criteria.
	•	The best validation performance occurred at epoch 160 with an MSE of 30.9109.
	•	This means after this point, further training did not improve validation performance, and early stopping was applied.
  
 
<img src="ML_models/nn_training and validation.jpg" width="300">

 This image contains four scatter plots, showing the relationship between the target (actual RSRP values) and output (predicted values from the neural network) for different datasets.

Each plot represents the correlation (R-value) between the actual and predicted values:
	1.	Top Left: Training Data (R = 0.9125)  
	•	Shows how well the neural network fits the training data.  
	•	The blue line represents the regression fit.  
	•	The dotted diagonal line (Y = T) is the ideal case where predictions perfectly match targets.  
	•	A high R-value (0.9125) indicates strong correlation, meaning the model has learned patterns well.  
	2.	Top Right: Validation Data (R = 0.90797)  
	•	This evaluates performance on unseen validation data.  
	•	The green line represents the regression fit.  
	•	The R-value is slightly lower than training (0.90797), indicating good generalization.  
	•	Since the validation R is close to training R, the model does not overfit significantly.  
	3.	Bottom Left: Test Data (R = 0.90968)  
	•	Shows model performance on completely unseen test data.  
	•	The red line represents the regression fit.  
	•	The R-value is 0.90968, close to training and validation, confirming strong generalization.  
	4.	Bottom Right: All Data Combined (R = 0.9114)  
	•	This combines training, validation, and test data to assess overall performance.  
	•	The black regression line fits all data points.  
	•	The high overall R-value (0.9114) suggests the model performs consistently across all datasets.  






 
        
# after fine tuning:
 <br> Neural networks error is minimized.    
here in model graphs, RMSE of neural networks decreases by 1 to 1.5 dB after adding extra layer of hidden neurons.

<img src="nn_default_LM\neural architecture.png" width="300" height="300">
architeture with 50 and 25 neurons in hidden layer. this is fine tune Neural network architecture.

<img src="nn_default_LM\models_comparison.jpg" width="300" height="300">

|Before Fine Tuning                        |                    After Fine Tuning           |
|:---------------------------------------:|:-----------------------------------------------:|
 <img src="https://github.com/user-attachments/assets/77866657-0c59-482e-8b49-019d5b79c96b" width = "300" height = "300">|<img src="nn_default_LM\nn_training_perf.jpg" width="300" height="300">




<img src="nn_default_LM\regression_after_fine_tuning.jpg" width="300" height="300">



<img src="nn_default_LM\nn_training_state.jpg" width="300" height="300">

* Regression Line Fit
  - The solid blue, green, red, and black lines represent the best linear fit to the data.
  - The dotted line (Y = T) is the ideal fit (perfect predictions).
  - Your model’s predictions are very close to the ideal fit, meaning it performs well.


## Measurements: ##

| Algorithmn  | MAE | RMSE  |
|----|----|----|
| Linear Regression  |   9.9985 |   12.394|
| Random Forest  |    2.8281   | 3.8075 |
| SVM  |  4.2522   | 5.5051 |
| Neural Network |   3.8284 |   4.9443 |






