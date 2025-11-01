% for given dataset, using optimized Guassian process regression (Linear Basis with kernel function: ardsquaredexponential)
% Caution: This code's execution time: 30 -45 min using parallel computing toolbox with 4 GPU cores.
data = readtable("cleaned2_network_data.xlsx");
% Data setup
lat = str2double(data.Lattitude);
lon = str2double(data.Longitude);
rsrp = str2double(data.RSRP_54_); % Replace with the actual column name
alt  = str2double(data.Altitude);


X = [lat, lon, alt];
Y = rsrp;
cv = cvpartition(size(X,1), 'KFold', 9);
numFolds = cv.NumTestSets;

% Preallocate outputs
rmse_rf_cv   = zeros(numFolds,1);
rsrp_diff_rf = cell(numFolds,1);
Y_pred_rf    = cell(numFolds,1);
delete(gcp('nocreate'));

% Start a process-based parallel pool
pool = parpool('Processes');

addAttachedFiles(pool, {'/Users/anandrajgopalan/Documents/MATLAB/project1/testcodes/test4.m'});
% Optional: enable parallelization internal to TreeBagger as well
% paroptions = statset('UseParallel', true);  % requires Process pool
tic
parfor i = 1:numFolds
    trainIdx = training(cv, i);
    testIdx  = test(cv, i);

    % Train Random Forest (Regression)
    
    cvMdl = fitrgp(X(trainIdx,:), Y(trainIdx),'FitMethod', 'sr', 'PredictMethod', 'fic','BasisFunction','linear','KernelFunction','ardsquaredexponential');
    % Predict on test fold
    Y_pred = predict(cvMdl, X(testIdx,:));
toc;
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
