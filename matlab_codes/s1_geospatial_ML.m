%[data, text, raw] = xlsread('Network_data.xlsx', '51');
data = readtable("cleaned2_network_data.xlsx");

lat =str2double(data.Lattitude);
lon = str2double(data.Longitude);
rsrp = str2double(data.RSRP_54_); % Replace with the actual column name

%F = scatteredInterpolant(lon, lat, rsrp, 'natural', 'none');
F = scatteredInterpolant(lon,lat,rsrp,'linear','nearest');
% Define Grid for Interpolation
lonGrid = linspace(min(lon), max(lon), 100);
latGrid = linspace(min(lat), max(lat), 100);
[LonGrid, LatGrid] = meshgrid(lonGrid, latGrid);

% Perform Interpolation
RSRP_Grid = F(LonGrid, LatGrid);

% Visualize the Interpolated Surface
figure;
surf(LonGrid, LatGrid, RSRP_Grid);
shading interp;
colorbar;
xlabel('Longitude');
ylabel('Latitude');
zlabel('Interpolated RSRP');
title('Geospatial Interpolation of RSRP Measurements');

rng(1); % For reproducibility

% Load the dataset
%data = readtable('network_data.xlsx'); % Replace with your file name

% Convert all variables to numeric if necessary (handles text or cell arrays)
dataNumeric = varfun(@(x) str2double(string(x)), data);

% Clean the data: Remove rows with -999 or missing values
%dataNumeric = dataNumeric(~any(dataNumeric.Variables == -999 | isnan(dataNumeric.Variables), 2), :);

% Extract predictors (features) and target (RSRP)
predictorVars = dataNumeric(:, setdiff(dataNumeric.Properties.VariableNames, {'Fun_RSRP_54_'}));
targetVar = dataNumeric.Fun_RSRP_54_;

% Split the dataset into training (80%) and testing (20%)
cv = cvpartition(size(dataNumeric, 1), 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx = test(cv);

X_train = predictorVars{trainIdx, :};
y_train = targetVar(trainIdx);
Y_train = y_train;
X_test = predictorVars{testIdx, :};
y_test = targetVar(testIdx);
Y_test = y_test;

% Train a Random Forest model
numTrees = 100; % Specify the number of trees
randomForestModel = TreeBagger(numTrees, X_train, y_train, ...
    'Method', 'regression', 'OOBPrediction', 'on','OOBPredictorImportance','on', 'PredictorNames', predictorVars.Properties.VariableNames);

% Evaluate model performance
y_pred = predict(randomForestModel, X_test);

% Calculate performance metrics
mse = mean((y_test - y_pred).^2); % Mean Squared Error
rmse = sqrt(mse); % Root Mean Squared Error
disp(['MSE: ', num2str(mse)]);
disp(['RMSE: ', num2str(rmse)]);

% Plot predicted vs actual RSRP
figure;
scatter(y_test, y_pred, 'filled');
hold on;
plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], 'r', 'LineWidth', 2);
xlabel('Actual RSRP');
ylabel('Predicted RSRP');
title('Random Forest: Predicted vs Actual RSRP');
grid on;

% Feature importance
figure;
bar(randomForestModel.OOBPermutedPredictorDeltaError);
xlabel('Predictors');
ylabel('Importance');
title('Feature Importance');
xticks(1:numel(predictorVars.Properties.VariableNames));
xticklabels(predictorVars.Properties.VariableNames);
xtickangle(45);
grid on;

%%Train Different Machine Learning Models

% --- Model 1: Linear Regression ---
mdl_lin = fitlm(X_train, Y_train);
Y_pred_lin = predict(mdl_lin, X_test);
rmse_lin = sqrt(mean((Y_test - Y_pred_lin).^2));
disp(['Linear Regression RMSE: ' num2str(rmse_lin)]);

% --- Model 2: Support Vector Regression (SVR) ---
mdl_svr = fitrsvm(X_train, Y_train, 'Standardize', true);
Y_pred_svr = predict(mdl_svr, X_test);
rmse_svr = sqrt(mean((Y_test - Y_pred_svr).^2));
disp(['SVR RMSE: ' num2str(rmse_svr)]);

% --- Model 3: Random Forest Regression (TreeBagger) ---
numTrees = 300;
mdl_rf = TreeBagger(numTrees, X_train, Y_train, 'Method', 'regression', ...
    'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');
Y_pred_rf = predict(mdl_rf, X_test);
rmse_rf = sqrt(mean((Y_test - Y_pred_rf).^2));
disp(['Random Forest RMSE: ' num2str(rmse_rf)]);

% --- Model 4: Neural Network Regression ---
% Using fitrnet requires R2019a or later
mdl_nn = fitrnet(X_train, Y_train, 'Standardize', true);
Y_pred_nn = predict(mdl_nn, X_test);
rmse_nn = sqrt(mean((Y_test - Y_pred_nn).^2));
disp(['Neural Network RMSE: ' num2str(rmse_nn)]);

%% 4. Visualize Predictions
figure;
subplot(2,2,1);
scatter(Y_test, Y_pred_lin, 'filled');
xlabel('Actual RSRP'); ylabel('Predicted RSRP');
title('Linear Regression');
grid on;
refline(1,0);

subplot(2,2,2);
scatter(Y_test, Y_pred_svr, 'filled');
xlabel('Actual RSRP'); ylabel('Predicted RSRP');
title('Support Vector Regression');
grid on;
refline(1,0);

subplot(2,2,3);
scatter(Y_test, Y_pred_rf, 'filled');
xlabel('Actual RSRP'); ylabel('Predicted RSRP');
title('Random Forest Regression');
grid on;
refline(1,0);

subplot(2,2,4);
scatter(Y_test, Y_pred_nn, 'filled');
xlabel('Actual RSRP'); ylabel('Predicted RSRP');
title('Neural Network Regression');
grid on;
refline(1,0);

%% 5. Summary of Results
fprintf('\nRMSE Summary:\n');
fprintf('Linear Regression: %f\n', rmse_lin);
fprintf('SVR:             %f\n', rmse_svr);
fprintf('Random Forest:   %f\n', rmse_rf);
fprintf('Neural Network:  %f\n', rmse_nn);

% Create feature matrix (X) and target variable (y)
X = [lat, lon]; % Features: Latitude & Longitude
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
rfModel = TreeBagger(100, X_train, Y_train, 'Method', 'regression', ...
    'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');
y_pred_rf = predict(rfModel, X_test);
y_pred_rf = str2double(y_pred_rf); % Convert to double
mae_values(2) = mean(abs(y_test - y_pred_rf));
rmse_values(2) = sqrt(mean((y_test - y_pred_rf).^2));

% 3. Support Vector Machine (SVM)
svmModel = fitrsvm(X_train, y_train, 'KernelFunction', 'rbf', 'Standardize', true);
y_pred_svm = predict(svmModel, X_test);
mae_values(3) = mean(abs(y_test - y_pred_svm));
rmse_values(3) = sqrt(mean((y_test - y_pred_svm).^2));

% 4. Neural Network (Feedforward)
net = feedforwardnet(10); % 10 hidden neurons
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
