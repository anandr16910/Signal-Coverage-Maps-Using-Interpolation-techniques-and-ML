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



% Hyperparameter Grid
numTrees = [25,50, 60, 75];  % Number of trees
leafValues = 1;   % MinLeafSize
splitValues = [5, 50, 70,170];  % MaxNumSplits

% Initialize Results Table
results = [];
best_rmse = Inf;
best_params = struct();

% Grid Search Loop
for i = 1:length(numTrees)
    for j = 1:length(leafValues)
        for k = 1:length(splitValues)
            temp_rmse = zeros(cv.NumTestSets, 1);
            
            for fold = 1:cv.NumTestSets
                % Get Train/Test Split
                trainIdx = training(cv, fold);
                testIdx = test(cv, fold);
                
                X_train = X(trainIdx, :);
                Y_train = Y(trainIdx);
                X_test = X(testIdx, :);
                Y_test = Y(testIdx);
                
                % Define Tree Template with Current Parameters
                treeTemplate = templateTree('MinLeafSize', leafValues(j), 'MaxNumSplits', splitValues(k));
                
                % Train Random Forest Model
                model = fitrensemble(X_train, Y_train, 'Method', 'LSBoost', ...
                                     'NumLearningCycles', numTrees(i), ...
                                     'Learners', treeTemplate,'LearnRate',0.1);
                
                % Predict on Test Set
                Y_pred = predict(model, X_test);
                
                % Compute RMSE
                temp_rmse(fold) = sqrt(mean((Y_test - Y_pred).^2));
            end
            
            % Store Average RMSE Across Folds
            mean_rmse = mean(temp_rmse);
            results = [results; numTrees(i), leafValues(j), splitValues(k), mean_rmse];
            
            % Update Best Parameters
            if mean_rmse < best_rmse
                best_rmse = mean_rmse;
                best_params.NumTrees = numTrees(i);
                best_params.MinLeafSize = leafValues(j);
                best_params.MaxNumSplits = splitValues(k);
            end
        end
    end
end

% Convert Results to Table
results_table = array2table(results, 'VariableNames', {'NumTrees', 'MinLeafSize', 'MaxNumSplits', 'RMSE'});

% Display Best Parameters
disp('Best Hyperparameters:');
disp(best_params);
fprintf('Best RMSE: %.4f dB\n', best_rmse);

% Train Final Model Using Best Found Parameters
final_tree = templateTree('MinLeafSize', best_params.MinLeafSize, 'MaxNumSplits', best_params.MaxNumSplits);
final_model = fitrensemble(X, Y, 'Method', 'Bag', ...
                           'NumLearningCycles', best_params.NumTrees, ...
                           'Learners', final_tree);

% Predict on Full Dataset
Y_pred_final = predict(final_model, X);

% Compute RMSE on Full Dataset
final_rmse = sqrt(mean((Y - Y_pred_final).^2));
fprintf('Final Model RMSE: %.4f dB\n', final_rmse);

figure;
plot(results(:,2), results(:,4), '-o', 'LineWidth', 2);
xlabel('MinLeafSize');
ylabel('RMSE (dB)');
title('Effect of MinLeafSize on RMSE');
grid on;