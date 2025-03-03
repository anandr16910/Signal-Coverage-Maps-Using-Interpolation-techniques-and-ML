% Load the processed network data
% Load data from Excel file
filename = 'cleaned2_network_data.xlsx';

% Read table from Excel
data = readtable(filename);

% Check required columns
requiredColumns = {'Lattitude', 'Longitude', 'RSRP_54_'}; % Adjust column names if needed
if ~all(ismember(requiredColumns, data.Properties.VariableNames))
    error('Missing required columns: Latitude, Longitude, or RSRP');
end

% Extract necessary columns
latitudes = str2double(data.Lattitude); 
longitudes = str2double(data.Longitude); 
rsrp_values = str2double(data.RSRP_54_); 

% Remove NaN or invalid values
% validIdx = ~isnan(latitudes) & ~isnan(longitudes) & ~isnan(rsrp_values);
% latitudes = latitudes(validIdx);
% longitudes = longitudes(validIdx);
% rsrp_values = rsrp_values(validIdx);

% Save processed data into .MAT file
save('processed_network_data.mat', 'latitudes', 'longitudes', 'rsrp_values');

% Display success message
disp('Data processing complete. File saved as processed_network_data.mat.');

load('processed_network_data.mat'); % Ensure this file contains latitudes, longitudes, rsrp_values, and distances
tx_latitude = mean(latitudes);
tx_longitude = mean(longitudes);

tx = txsite("Name", "Base Station", ...
            "Latitude", tx_latitude, ...
            "Longitude", tx_longitude, ...
            "AntennaHeight", 30, ...  % 30m antenna height
            "TransmitterPower", 20, ... % 20W power
            "TransmitterFrequency", 3.5e9); % 3.5 GHz (5G mid-band)

% Define Receiver Sites (Measurement Locations)
rx = rxsite("Latitude", latitudes, ...
            "Longitude", longitudes, ...
            "AntennaHeight", 1.5); % Typical UE height (1.5m)

distances = haversine(tx_latitude, tx_longitude, latitudes, longitudes);
function d = haversine(lat1, lon1, lat2, lon2)
    R = 6371000; % Earth radius in meters
    dlat = deg2rad(lat2 - lat1);
    dlon = deg2rad(lon2 - lon1);
    a = sin(dlat/2).^2 + cos(deg2rad(lat1)) .* cos(deg2rad(lat2)) .* sin(dlon/2).^2;
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    d = R * c;
end
% Check if necessary data is loaded
if ~exist('latitudes', 'var') || ~exist('longitudes', 'var') || ~exist('distances', 'var')
    error('The file processed_network_data.mat must contain latitudes, longitudes, and distances variables.');
end

% Define parameters for the theoretical path loss model
frequency = 3.5e9; % Frequency in Hz (e.g., 3.5 GHz for 5G)
transmitPower = 5; % Transmit power in dBm (e.g., 43 dBm = 20 W)
txGain = 0; % Transmitter antenna gain in dBi
rxGain = 0; % Receiver antenna gain in dBi

% Choose a path loss model (e.g., Log-Distance)
%plModel = propagationModel("log-distance", "PathLossExponent", 3.5); % Path loss exponent for urban areas
% Define Log-Distance Propagation Model
%plModel = propagationModel("logdistance", "Exponent", 3.5);
%plModel = propagationModel("gas");
%plModel = propagationModel("close-in");
% Define Close-In Propagation Model (Similar to Log-Distance)
plModel = propagationModel("close-in", "ReferenceDistance", 1, "PathLossExponent", 3.5);
%plModel = propagationModel("itu-r", "Region", "Urban");
% Compute the path loss for each distance
pathLoss = pathloss(plModel, rx,tx);
%pathLoss = pathloss(plModel, frequency, distances); % Path loss in dB

% Compute the theoretical received signal strength (RSS)
rss_theoretical = transmitPower + txGain + rxGain - pathLoss; % RSS in dBm

% Plot the theoretical RSS against distances
figure;
scatter(distances, rss_theoretical, 25, 'filled', 'MarkerFaceColor', 'r'); % Plot theoretical RSS
hold on;

% Overlay measured RSRP values if available
if exist('rsrp_values', 'var')
    scatter(distances, rsrp_values, 25, 'filled', 'MarkerFaceColor', 'b'); % Plot measured RSRP
end

% Add plot details
grid on;
title('Theoretical vs Measured Signal Strength');
xlabel('Distance (meters)');
ylabel('Signal Strength (dBm)');
legend('Theoretical RSS', 'Measured RSRP', 'Location', 'Best');
hold off;

% Display statistics for comparison
fprintf('Theoretical RSS Range: [%f, %f] dBm\n', min(rss_theoretical), max(rss_theoretical));
if exist('rsrp_values', 'var')
    fprintf('Measured RSRP Range: [%f, %f] dBm\n', min(rsrp_values), max(rsrp_values));
end