% --- Part 1: Data Setup and Initialization ---

close all;
clear all;

% Load data (Ensure 'cleaned2_network_data.xlsx' is in the MATLAB path)
data = readtable("cleaned2_network_data.xlsx");

% Extract and convert data columns
% NOTE: Assuming your data columns are correctly named and contain numeric data,
% str2double is used as in your original code to handle potential string-like
% representations of numbers.
lat  = str2double(data.Lattitude);
lon  = str2double(data.Longitude);
rsrp = str2double(data.RSRP_54_); % Response variable Y
alt  = str2double(data.Altitude);

% Predictor matrix X: [Lattitude, Longitude, Altitude]
X = [lat, lon, alt];
% Response vector Y: RSRP
Y = rsrp;

% Define 9-Fold Cross-Validation
cv = cvpartition(size(X,1), 'KFold', 9);
numFolds = cv.NumTestSets;

% Preallocate outputs for K-Fold results
rmse_rf_cv   = zeros(numFolds, 1);
rsrp_diff_rf = cell(numFolds, 1); % Stores (Predicted - Measured) difference
Y_pred_rf    = cell(numFolds, 1); % Stores the predicted RSRP values


% --- Part 2: K-Fold Cross-Validation Loop ---

fprintf('Starting 9-Fold Cross-Validation...\n');

for i = 1:numFolds
    fprintf('  Processing Fold %d of %d...\n', i, numFolds);
    
    % Get indices for the current fold
    trainIdx = training(cv, i);
    testIdx  = test(cv, i);
    
    % Extract training and testing data
    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx);
    X_test  = X(testIdx, :);
    Y_test  = Y(testIdx);
    
    % 1. Train the model using the training data
    % The function is defined below, it returns the trained model and validation RMSE
    [trainedModel, ~] = trainRegressionModel(X_train, Y_train);
    
    % 2. Predict RSRP for the test set
    % We use the predictFcn provided by the trainedModel struct
    Y_pred = trainedModel.predictFcn(X_test);
    
    % 3. Calculate metrics for the current fold
    
    % Root Mean Squared Error (RMSE)
    rmse_rf_cv(i) = sqrt(mean((Y_test - Y_pred).^2));
    
    % RSRP Difference (Error): Predicted - Measured
    rsrp_diff_rf{i} = Y_pred - Y_test;
    
    % Store the predictions for later use (e.g., plotting)
    Y_pred_rf{i}    = Y_pred;
end

fprintf('Cross-Validation Complete.\n');

% --- Part 3: Analysis and Visualization ---

% Find the best performing fold (minimum RMSE)
[best_rmse_rf, bestFold_rf] = min(rmse_rf_cv);

% Extract the results for the best fold
best_rsrp_diff_rf = rsrp_diff_rf{bestFold_rf};
best_pred_rf      = Y_pred_rf{bestFold_rf};
testIdx_best      = test(cv, bestFold_rf); % Get the indices of the test points for the best fold

fprintf('\n--- Results Summary ---\n');
fprintf('Best Fold: %d (Lowest RMSE: %.4f dB)\n', bestFold_rf, best_rmse_rf);
fprintf('Worst Fold: %d (Highest RMSE: %.4f dB)\n', find(rmse_rf_cv == max(rmse_rf_cv), 1, 'first'), max(rmse_rf_cv));


% Create the geoscatter plot for the best fold's prediction error
figure;
% The latitude and longitude of the test points in the best fold
geoscatter(lat(testIdx_best), lon(testIdx_best), ...
           60, best_rsrp_diff_rf, 'filled'); % 60 is the marker size
geobasemap topographic; % Use a topographic map base layer
colorbar;
colormap(parula); % Standard colormap
caxis([-15 15]); % Set the color limits for the error (Predicted - Measured)

% Title including the RMSE of the best fold
title(sprintf('Ensemble Predicted – Measured RSRP Diff (dB) - Best Fold %d (RMSE: %.2f)', ...
              bestFold_rf, best_rmse_rf));
xlabel('Longitude');
ylabel('Latitude');


% --- Part 4: Helper Function (Copied from your provided code) ---
% This function trains the Regression Ensemble model (Random Forest)

function [trainedModel, validationRMSE] = trainRegressionModel(trainingData, responseData)
    % Extract predictors and response
    inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3'});
    predictorNames = {'column_1', 'column_2', 'column_3'};
    predictors = inputTable(:, predictorNames);
    response = responseData;
    
    % Train a regression model (Regression Ensemble/Random Forest with Bagging)
    template = templateTree(...
        'MinLeafSize', 2, ...
        'NumVariablesToSample', 3);
    regressionEnsemble = fitrensemble(...
        predictors, ...
        response, ...
        'Method', 'Bag', ...
        'NumLearningCycles', 30, ...
        'Learners', template);
    
    % Create the result struct with predict function
    predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
    ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
    trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
    
    % Add additional fields to the result struct
    trainedModel.RegressionEnsemble = regressionEnsemble;
    
    % NOTE: Removed the internal K-Fold validation from the original function
    % as we are performing the K-Fold externally. The original model included
    % a 5-fold cross-validation *inside* the function, which is usually redundant
    % when performing an external K-fold loop.
    
    % Set validationRMSE to 0 or remove it if not needed, as the external loop
    % is calculating the true validation RMSE. For compatibility, we'll set it to 0.
    validationRMSE = 0; 
end
