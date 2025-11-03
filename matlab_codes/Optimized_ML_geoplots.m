close all;
clear all;
data = readtable("cleaned2_network_data.xlsx");

lat =str2double(data.Lattitude);
lon = str2double(data.Longitude);
rsrp = str2double(data.RSRP_54_); % Replace with the actual column name
alt  = str2double(data.Altitude);
delete(gcp('nocreate'));

% Start a process-based parallel pool
parpool('Processes');
% Corrected and improved version of RSRP prediction cross-validation with geographic plotting

% Data setup
X = [lat, lon, alt];
Y = rsrp;

cv = cvpartition(size(X,1), 'KFold', 9);
numFolds = cv.NumTestSets;

% Preallocate outputs
rmse_nn_cv = zeros(numFolds,1);
rsrp_diff_all = cell(numFolds,1);  % Store fold-wise differences
Y_pred_all = cell(numFolds,1);

parfor i = 1:numFolds
    trainIdx = training(cv, i);
    testIdx  = test(cv, i);

    % Define Neural Network
    hiddenLayerSize = [25 15];
    net = fitnet(hiddenLayerSize, 'trainlm');
    net.trainParam.showWindow = false;

    % Train
    [net, tr] = train(net, X(trainIdx,:)', Y(trainIdx)');

    % Predict
    Y_pred = net(X(testIdx,:)');

    % Calculate RMSE and difference
    rmse_nn_cv(i) = sqrt(mean((Y(testIdx)' - Y_pred).^2));
    rsrp_diff_all{i} = Y_pred' - Y(testIdx);
    Y_pred_all{i} = Y_pred';
end

% Find best-performing fold (minimum RMSE)
[~, bestFold] = min(rmse_nn_cv);
best_rsrp_diff = rsrp_diff_all{bestFold};
bestYpred = Y_pred_all{bestFold};
fprintf('Best RMSE found at fold %d: %.4f\n', bestFold, rmse_nn_cv(bestFold));

% Visualization for best-performing fold
testIdx = test(cv, bestFold);  % indices for the corresponding test set
figure
geoscatter(lat(testIdx), lon(testIdx), 60, best_rsrp_diff, 'filled')
geobasemap topographic
colorbar
colormap(parula)
caxis([-15 15])
title(sprintf('Predicted – Measured RSRP Difference (dB) - Best Fold (%d)', bestFold))

X = [lat, lon, alt];
Y = rsrp;

cv = cvpartition(size(X,1), 'KFold', 9);
numFolds = cv.NumTestSets;

% Preallocate outputs
rmse_rf_cv   = zeros(numFolds,1);
rsrp_diff_rf = cell(numFolds,1);
Y_pred_rf    = cell(numFolds,1);

% Optional: enable parallelization internal to TreeBagger as well
% paroptions = statset('UseParallel', true);  % requires Process pool

parfor i = 1:numFolds
    trainIdx = training(cv, i);
    testIdx  = test(cv, i);

    % Train Random Forest (Regression)
    mdl_rf = TreeBagger(75, X(trainIdx,:), Y(trainIdx), ...
                        'Method', 'regression' ...
                        ... % ,'Options', paroptions   % uncomment to parallelize trees
                        );

    % Predict on test fold
    Y_pred = predict(mdl_rf, X(testIdx,:));

    % RMSE and difference
    rmse_rf_cv(i)   = sqrt(mean((Y(testIdx) - Y_pred).^2));
    rsrp_diff_rf{i} = Y_pred - Y(testIdx);
    Y_pred_rf{i}    = Y_pred;
end

% Pick best fold (minimum RMSE)
[best_rmse_rf, bestFold_rf] = min(rmse_rf_cv);
best_rsrp_diff_rf = rsrp_diff_rf{bestFold_rf};
best_pred_rf      = Y_pred_rf{bestFold_rf};
testIdx_best      = test(cv, bestFold_rf);

fprintf('Random Forest best fold: %d, RMSE: %.4f\n', bestFold_rf, best_rmse_rf);

% Plot difference for best fold only
figure
geoscatter(lat(testIdx_best), lon(testIdx_best), 60, best_rsrp_diff_rf, 'filled')
geobasemap topographic
colorbar
colormap(parula)
caxis([-15 15])
title(sprintf('RF Predicted – Measured RSRP Diff (dB) - Best Fold %d', bestFold_rf))
