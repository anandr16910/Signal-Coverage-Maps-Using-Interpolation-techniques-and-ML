% Load the dataset
data = readtable('cleaned2_network_data.xlsx'); % Replace with your file name

% Clean the data: Remove rows with -999 or missing values
%data = data(~any(data{:,:} == -999 | isnan(data{:,:}), 2), :);
latitudes = str2double(data.Lattitude); 
longitudes = str2double(data.Longitude); 
rsrp_values = str2double(data.RSRP_54_); 

% Save processed data into .MAT file
save('processed_network_data1.mat', 'latitudes', 'longitudes', 'rsrp_values');

% Display success message
disp('Data processing complete. File saved as processed_network_data1.mat.');

load('processed_network_data1.mat'); % Ensure this file contains latitudes, longitudes, rsrp_values, and distances
tx_latitude = mean(latitudes);
tx_longitude = mean(longitudes);
%tx shd be mean of lat longitudes.
% Extract transmitter and receiver coordinates
tx_lat = tx_latitude; % Replace with the correct column name for transmitter latitude
tx_lon = tx_longitude; % Replace with the correct column name for transmitter longitude
rx_lat = latitudes; % Replace with the correct column name for receiver latitude
rx_lon = longitudes; % Replace with the correct column name for receiver longitude

% Compute distances between transmitter and receiver (Haversine formula)
R = 6371; % Earth's radius in km
lat1 = deg2rad(tx_lat);
lon1 = deg2rad(tx_lon);
lat2 = deg2rad(rx_lat);
lon2 = deg2rad(rx_lon);
a = sin((lat2 - lat1) / 2).^2 + cos(lat1) .* cos(lat2) .* sin((lon2 - lon1) / 2).^2;
c = 2 * atan2(sqrt(a), sqrt(1 - a));
distance_km = R * c; % Distance in kilometers

% Extract the RSRP values
%rsrp = data.RSRP_54_;

% Convert distance to meters
distance_m = distance_km * 1000;

% Reference distance (d0) and initial guess for path loss exponent (n)
d0 = 30; % Reference distance in meters
PL_d0 = mean(rsrp_values(distance_m <= d0)); % Approximate path loss at reference distance
n_initial = 2; % Initial guess for path loss exponent

% Fit the log-distance model
log_dist = @(n, d) PL_d0 + 10 * n * log10(d / d0);
model = @(n) sum((log_dist(n, distance_m) - rsrp_values).^2); % Cost function
n_fitted = fminsearch(model, n_initial); % Optimize path loss exponent

% Predicted RSRP using the fitted model
rsrp_pred = log_dist(n_fitted, distance_m);

% Evaluate the model
mse = mean((rsrp_values - rsrp_pred).^2);
rmse = sqrt(mse);
disp(['Fitted Path Loss Exponent (n): ', num2str(n_fitted)]);
disp(['MSE: ', num2str(mse)]);
disp(['RMSE: ', num2str(rmse)]);

% Plot actual vs predicted RSRP
figure;
scatter(distance_m, rsrp_values, 'b', 'filled', 'DisplayName', 'Actual RSRP');
hold on;
scatter(distance_m, rsrp_pred, 'r', 'filled', 'DisplayName', 'Predicted RSRP');
xlabel('Distance (m)');
ylabel('RSRP (dB)');
legend show;
title('Log-Distance Propagation Model');
grid on;