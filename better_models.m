close all;
clear all;
data = readtable("cleaned2_network_data.xlsx");

lat =str2double(data.Lattitude);
lon = str2double(data.Longitude);
rsrp = str2double(data.RSRP_54_); % Replace with the actual column name
alt  = str2double(data.Altitude);

X =  [lat, lon, alt];  % Feature matrix
Y = rsrp; % Target values

cv = cvpartition(size(X,1), 'KFold', 5); % 5-fold CV
rmse_nn_cv = zeros(cv.NumTestSets,1);

parfor i = 1:1:3
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    % Define Neural Network
    hiddenLayerSize = [15 5];
    net = fitnet(hiddenLayerSize, 'trainlm');
    
    % Train the network
    [net, tr] = train(net, X(trainIdx,:)', Y(trainIdx)');
    
    % Predict RSRP
    Y_pred_nn = net(X(testIdx,:)');
    
    % Compute RMSE
    rmse_nn_cv(i) = sqrt(mean((Y(testIdx)' - Y_pred_nn).^2));
end

rmse_nn = mean(rmse_nn_cv);
disp(['Cross-Validated RMSE (Neural Network): ', num2str(rmse_nn)]);

%random forest
rmse_rf_cv = zeros(cv.NumTestSets,1); % Store RMSE for each fold

for i = 1:1:3
    % Training and test indices
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    % Train Random Forest
    mdl_rf = TreeBagger(20, X(trainIdx,:), Y(trainIdx), 'Method', 'regression');
    
    % Predict RSRP
    Y_pred_rf = predict(mdl_rf, X(testIdx,:));
    
    % Compute RMSE for this fold
    rmse_rf_cv(i) = sqrt(mean((Y(testIdx) - Y_pred_rf).^2));
end

% Final RMSE (average over all folds)
rmse_rf = mean(rmse_rf_cv);
disp(['Cross-Validated RMSE (Random Forest): ', num2str(rmse_rf)]);


rmse_svr_cv = zeros(cv.NumTestSets,1);

for i = 1:cv.NumTestSets
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    mdl_svr = fitrsvm(X(trainIdx,:), Y(trainIdx), 'KernelFunction', 'rbf', 'Standardize', true);
    
    Y_pred_svr = predict(mdl_svr, X(testIdx,:));
    
    rmse_svr_cv(i) = sqrt(mean((Y(testIdx) - Y_pred_svr).^2));
end

rmse_svr = mean(rmse_svr_cv);
disp(['Cross-Validated RMSE (SVR): ', num2str(rmse_svr)]);

% gradient boost 
rmse_gb_cv = zeros(cv.NumTestSets,1);

for i = 1:cv.NumTestSets

    mdl_gb = fitrensemble(X(trainIdx,:), Y(trainIdx,:), 'Method', 'LSBoost', 'NumLearningCycles', 100);

% Predict RSRP
    Y_pred_gb = predict(mdl_gb, X(testIdx,:));

% Compute RMSE
    rmse_gb_cv(i) = sqrt(mean((Y(testIdx) - Y_pred_gb).^2));
end
rmse_gb = mean(rmse_gb_cv);
disp(['Gradient Boosting RMSE: ', num2str(rmse_gb)]);

%KNN model

rmse_knn_cv = zeros(cv.NumTestSets,1);

for i=1:cv.NumTestSets

mdl_knn = fitcknn(X(trainIdx,:), Y(trainIdx,:), 'NumNeighbors', 5,'Distance','euclidean');

% Predict RSRP
Y_pred_knn = predict(mdl_knn, X(testIdx,:));

% Compute RMSE
rmse_knn_cv(i) = sqrt(mean((Y(testIdx) - Y_pred_knn).^2));
end

rmse_knn = mean(rmse_knn_cv);
disp(['KNN RMSE: ', num2str(rmse_knn)]);

%GLM 
rmse_glm_cv = zeros(cv.NumTestSets,1);

for i=1:cv.NumTestSets

mdl_glm = fitglm(X(trainIdx,:), Y(trainIdx,:));

% Predict RSRP
Y_pred_glm = predict(mdl_glm, X(testIdx,:));

% Compute RMSE
rmse_glm_cv(i) = sqrt(mean((Y(testIdx) - Y_pred_glm).^2));

end
rmse_glm = mean(rmse_glm_cv);
disp(['GLM RMSE: ', num2str(rmse_glm)]);

model_names = {'Neural Network', 'Random Forest', 'SVR', 'Gradient Boosting', 'KNN', 'GLM'};
rmse_values = [rmse_nn, rmse_rf, rmse_svr, rmse_gb, rmse_knn, rmse_glm];

disp(table(model_names', rmse_values', 'VariableNames', {'Model', 'RMSE'}));

% Plot RMSE comparison
figure;
bar(rmse_values);
set(gca, 'XTickLabel', model_names);
ylabel('RMSE (dBm)');
title('Comparison of ML Models for RSRP Prediction');
grid on;