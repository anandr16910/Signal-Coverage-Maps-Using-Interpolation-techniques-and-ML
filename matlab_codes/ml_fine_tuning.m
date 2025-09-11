close all;
clear all;
data = readtable("cleaned2_network_data.xlsx");


lat =str2double(data.Lattitude);
lon = str2double(data.Longitude);
rsrp = str2double(data.RSRP_54_); % Replace with the actual column name
alt  = str2double(data.Altitude);

% Create feature matrix (X) and target variable (y)
X = [lat, lon, alt]; % Features: Latitude & Longitude
y = rsrp;       % Target: RSRP

% Split into Training (80%) and Testing (20%) Sets
cv = cvpartition(length(y), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Initialize Model Performance Metrics
model_names = {'Linear Regression', 'Random Forest', 'SVM', 'Neural Network'};
mae_values = zeros(length(model_names),1);
rmse_values = zeros(length(model_names),1);

% 1. Linear Regression Model
linModel = fitlm(X_train, y_train);
y_pred_lin = predict(linModel, X_test);
mae_values(1) = mean(abs(y_test - y_pred_lin));
rmse_values(1) = sqrt(mean((y_test - y_pred_lin).^2));

% 2. Random Forest (TreeBagger)
%rfModel = TreeBagger(50, X_train, y_train, 'OOBPredictorImportance', 'on');
rfModel = TreeBagger(1800, X_train, y_train, 'Method', 'regression', ...
    'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');
y_pred_rf = predict(rfModel, X_test);
%y_pred_rf  = Y_pred_rf;
%y_pred_rf = str2double(y_pred_rf); % Convert to double
mae_values(2) = mean(abs(y_test - y_pred_rf));
rmse_values(2) = sqrt(mean((y_test - y_pred_rf).^2));

% 3. Support Vector Machine (SVM)
svmModel = fitrsvm(X_train, y_train, 'KernelFunction', 'rbf', 'Standardize', true);
y_pred_svm = predict(svmModel, X_test);
mae_values(3) = mean(abs(y_test - y_pred_svm));
rmse_values(3) = sqrt(mean((y_test - y_pred_svm).^2));

% 4. Neural Network (Feedforward)
net = feedforwardnet(100,'trainlm'); % 10 hidden neurons


net = train(net, X_train', y_train'); % Train model
y_pred_nn = net(X_test'); % Predict
y_pred_nn = y_pred_nn'; % Convert back to column vector
mae_values(4) = mean(abs(y_test - y_pred_nn));
rmse_values(4) = sqrt(mean((y_test - y_pred_nn).^2));

% Compare Models
disp('Model Performance Comparison:');
disp(table(model_names', mae_values, rmse_values, 'VariableNames', {'Model', 'MAE', 'RMSE'}));

% Plot Model Errors
figure;
bar([mae_values, rmse_values]);
set(gca, 'XTickLabel', model_names);
legend('MAE', 'RMSE');
ylabel('Error (dB)');
title('Comparison of ML Models for RSRP Prediction');
grid on;
%% Fine tuning neural networks
% Load your data
%load('your_data.mat'); % Replace with your dataset

% Define input (features) and output (target)
X = X;  % Feature matrix
Y = rsrp; % Target values

% Define neural network
hiddenLayerSize = [100 75 30]; % Two hidden layers (adjustable)
net = fitnet(hiddenLayerSize, 'trainlm'); % Using Levenberg-Marquardt (LM)

% Use ReLU activation function
for i = 1:length(hiddenLayerSize)
    net.layers{i}.transferFcn = 'poslin'; % ReLU function
end

% Train/Test Split
net.divideParam.trainRatio = 0.7; % 70% for training
net.divideParam.valRatio = 0.15;  % 15% for validation
net.divideParam.testRatio = 0.15; % 15% for testing

% Set training options
net.trainParam.epochs = 200;        % Maximum epochs
net.trainParam.lr = 0.01;           % Learning rate
net.trainParam.min_grad = 1e-6;     % Minimum gradient
net.trainParam.max_fail = 15;       % Early stopping

% Regularization (L2 Weight Decay)
net.performParam.regularization = 0.001; % L2 regularization strength

% Train the network
[net, tr] = train(net, X', Y');

% Predictions
%Y_pred = predict(net,X');
Y_pred = net(X');
% Performance evaluation
trainPerf = perform(net, Y(tr.trainInd)', Y_pred(tr.trainInd)');
valPerf = perform(net, Y(tr.valInd)', Y_pred(tr.valInd)');
testPerf = perform(net, Y(tr.testInd)', Y_pred(tr.testInd)');

fprintf('Training Performance: %.4f\n', trainPerf);
fprintf('Validation Performance: %.4f\n', valPerf);
fprintf('Test Performance: %.4f\n', testPerf);

% Plot regression analysis
figure;
plotregression(Y(tr.trainInd)', Y_pred(tr.trainInd)', 'Training', ...
               Y(tr.valInd)', Y_pred(tr.valInd)', 'Validation', ...
               Y(tr.testInd)', Y_pred(tr.testInd)', 'Testing', ...
               Y', Y_pred', 'Overall');

% Compute RMSE and MAE for Neural Network Predictions

% Mean Absolute Error (MAE)
mae_nn = mean(abs(Y' - Y_pred));

% Root Mean Square Error (RMSE)
rmse_nn = sqrt(mean((Y' - Y_pred).^2));

% Display RMSE in Table Format
model_names = {'Neural Network'}; % You can add other models here
mae_values = mae_nn;
rmse_values = rmse_nn;

disp(table(model_names', mae_values, rmse_values, 'VariableNames', {'Model', 'MAE', 'RMSE'}));

% Plot Model Errors
figure;
bar([mae_values, rmse_values]);
set(gca, 'XTickLabel', model_names);
legend('MAE', 'RMSE');
ylabel('Error (dB)');
title('Comparison of Neural Network RMSE and MAE for RSRP Prediction');
grid on;