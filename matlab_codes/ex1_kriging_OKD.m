% Load the processed data
%data = load('processed_network_data.mat');
data = readtable('cleaned2_network_data.xlsx');
%data = ta(data);
%data = cell2mat(data) ;
% if ~exist('latitudes', 'var') || ~exist('longitudes', 'var') || ~exist('distances', 'var')
%     error('The file processed_network_data.mat must contain latitudes, longitudes, and distances variables.');
% end
% Extract variables
latitudes = str2double(data.Lattitude(:));
longitudes = str2double(data.Longitude(:));
rsrp_values = str2double(data.RSRP_54_(:));
%lat_grid = str2double(data.Lattitude);
%lon_grid = str2double(data.Longitude);
lonGrid = linspace(min(longitudes), max(longitudes), 50);
latGrid = linspace(min(latitudes), max(latitudes), 50);
[LonGrid, LatGrid] = meshgrid(lonGrid, latGrid);
% Create interpolation function (using Inverse Distance Weighting or similar)
F = scatteredInterpolant(latitudes, longitudes, rsrp_values, 'linear', 'none');

% Interpolate RSRP values over the grid
%[lat_mesh, lon_mesh] = meshgrid(lat_grid, lon_grid);
rsrp_grid = F(LonGrid, LatGrid);

% Plot the coverage map
figure;
contourf(LonGrid, LatGrid, rsrp_grid, 20, 'LineColor', 'none'); % Filled contour plot
colorbar;
title('Baseline Coverage Map (RSRP)');
xlabel('Longitude');
ylabel('Latitude');

% Overlay measurement points
hold on;
scatter(longitudes, latitudes, 20, rsrp_values, 'filled', 'MarkerEdgeColor', 'k');
legend('Interpolated RSRP', 'Measurement Points');
hold off;

% Load the processed data


% Create a meshgrid for the grid
[lat_mesh, lon_mesh] = meshgrid(LatGrid, LonGrid);

% Define spatial points and values
spatial_points = [latitudes, longitudes];
grid_points = [lat_mesh(:), lon_mesh(:)];

% Perform Ordinary Kriging (OKD)
% Use MATLAB's fitrgp for Kriging if Geostatistics Toolbox is unavailable
% or create a custom covariance matrix and perform prediction
mdl = fitrgp(spatial_points, rsrp_values, 'KernelFunction', 'squaredexponential');

% Predict RSRP over the grid
predicted_rsrp = predict(mdl, grid_points);

% Reshape the predicted values to the grid format
kriging_rsrp_grid = reshape(predicted_rsrp, size(lat_mesh));

% Plot the Kriging-based coverage map
figure;
contourf(lon_mesh, lat_mesh, kriging_rsrp_grid, 20, 'LineColor', 'none'); % Filled contour plot
colorbar;
title('Kriging-Based Coverage Map (RSRP)');
xlabel('Longitude');
ylabel('Latitude');

% Overlay measurement points
hold on;
scatter(longitudes, latitudes, 20, rsrp_values, 'filled', 'MarkerEdgeColor', 'k');
legend('Kriging Interpolated RSRP', 'Measurement Points');
hold off;

% Load the processed data
lat_grid = linspace(min(latitudes), max(latitudes), 30);  % 30 points instead of 50
lon_grid = linspace(min(longitudes), max(longitudes), 30);

% Create a meshgrid for interpolation points
[lat_mesh, lon_mesh] = meshgrid(lat_grid, lon_grid);
grid_points = [lat_mesh(:), lon_mesh(:)];  % Grid points as [latitude, longitude]

% Define spatial points and values
spatial_points = [latitudes, longitudes];
idx = randperm(length(latitudes), min(100, length(latitudes)));  % Use up to 500 points
spatial_points = [latitudes(idx), longitudes(idx)];
rsrp_values = rsrp_values(idx);
distance_matrix = pdist2(spatial_points, spatial_points);
% Step 1: Compute experimental variogram
distances = pdist2(spatial_points, spatial_points); % Pairwise distances
semi_variances = 0.5 * (rsrp_values - rsrp_values').^2; % Semi-variogram values
[distance_bins, ~, bin_indices] = histcounts(distances(:), 50); % Group distances
avg_variance = accumarray(bin_indices, semi_variances(:), [], @mean); % Average variance per bin


% Step 2: Fit a theoretical variogram model (e.g., Spherical model)
% Model parameters: Nugget, Sill, and Range
nugget = 0; % Minimal variance at zero distance
sill = max(avg_variance); % Variance plateau
range = max(distance_matrix(:)) * 0.3;  % Use 30% of max distance
%range = distance_bins(find(avg_variance < sill * 0.95, 1, 'first')); % Effective range
variogram_model = @(h) nugget + (sill - nugget) * (1.5 * (h / range) - 0.5 * (h / range).^3).*(h <= range) + sill*(h > range);
%compute covr matrix
cov_matrix = variogram_model(distance_matrix);
cov_matrix = [cov_matrix, ones(size(cov_matrix, 1), 1); ones(1, size(cov_matrix, 1)), 0]; % Add Lagrange multiplier
% Step 3: Kriging Interpolation
num_points = size(grid_points, 1);
kriged_values = zeros(num_points, 1);
kriged_values = zeros(size(grid_points, 1), 1);

% Parallel Kriging (OKD) for faster computation 
parfor i = 1:size(grid_points, 1)
    % Compute distances to grid point
    d_grid_to_points = pdist2(grid_points(i, :), spatial_points);
    gamma = variogram_model(d_grid_to_points);
    gamma = [gamma(:); 1];
    
    % Solve Kriging system
    weights = cov_matrix \ gamma;
    kriged_values(i) = weights(1:end-1)' * rsrp_values;
end
%for i = 1:num_points
    % Distance between grid point and known points
    %d_grid_to_points = sqrt(sum((spatial_points - grid_points(i, :)).^2, 2));
    % d_grid_to_points = pdist2(grid_points(i, :), spatial_points); % Use pdist2 for pairwise distances
    % % Variogram values for these distances
    % gamma = variogram_model(d_grid_to_points);
    % 
    % % Create Kriging system
    % %distance_matrix = sqrt(sum((spatial_points - spatial_points').^2, 3));
    % distance_matrix = pdist2(spatial_points, spatial_points); % Pairwise distance matrix
    % 
    % gamma = [gamma(:); 1];
    % 
    % % Solve Kriging system
    % weights = cov_matrix \ gamma;
    % 
    % % Compute Kriged value
    %kriged_values(i) = weights(1:end-1)' * rsrp_values;
%end

% Reshape the Kriged values into grid format
kriged_rsrp_grid = reshape(kriged_values, size(lat_mesh));

% Step 4: Plot the Ordinary Kriging map
figure;
contourf(lon_mesh, lat_mesh, kriged_rsrp_grid, 20, 'LineColor', 'none'); % Filled contour plot
colormap("parula")
colorbar;
title('Ordinary Kriging-Based Coverage Map (RSRP)');
xlabel('Longitude');
ylabel('Latitude');
n = min([length(latitudes), length(longitudes), length(rsrp_values)]);
latitudes = latitudes(1:n);
longitudes = longitudes(1:n);
rsrp_values = rsrp_values(1:n);
assert(length(latitudes) == length(longitudes), 'Latitude and longitude arrays must have the same length.');
assert(length(latitudes) == length(rsrp_values), 'RSRP values must match the number of latitude/longitude points.');

% Normalize RSRP values for color mapping
c = rsrp_values(:);  % Ensure column vector
c = (c - min(c)) / (max(c) - min(c));  % Normalize to [0, 1]

% Scatter plot

% Overlay measurement points
% hold on;
% scatter(longitudes, latitudes, 30, c, 'filled', 'MarkerEdgeColor', 'k');  % Scatter plot
% %scatter(longitudes, latitudes, 20, c, 'filled', 'MarkerEdgeColor', 'k');
% legend('Kriging Interpolated RSRP', 'Measurement Points');
% hold off;

hold on;
scatter(longitudes, latitudes, 20, rsrp_values, 'filled', 'MarkerEdgeColor', 'k');
legend('ord Kriging Interpolated RSRP', 'Measurement Points');
hold off;
