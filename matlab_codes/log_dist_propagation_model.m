% Load and clean dataset
data = readtable('cleaned2_network_data.xlsx'); 

% Convert to numeric and filter valid data
latitudes = str2double(data.Lattitude); 
longitudes = str2double(data.Longitude); 
rsrp_values = str2double(data.RSRP_54_); 

% Remove extreme values & NaNs
valid_idx = ~isnan(latitudes) & ~isnan(longitudes) & ~isnan(rsrp_values) ...
            & rsrp_values > -130 & rsrp_values < -50; 
latitudes = latitudes(valid_idx);
longitudes = longitudes(valid_idx);
rsrp_values = smoothdata(rsrp_values(valid_idx), 'movmean', 5); % Moving Average Smoothing

% Save processed data
save('processed_network_data1.mat', 'latitudes', 'longitudes', 'rsrp_values');

% Compute Transmitter (TX) Mean Position
tx_lat = mean(latitudes);
tx_lon = mean(longitudes);

% Compute Distance using Haversine Formula (Vectorized)
R = 6371e3; % Earth's radius in meters
lat1 = deg2rad(tx_lat);
lon1 = deg2rad(tx_lon);
lat2 = deg2rad(latitudes);
lon2 = deg2rad(longitudes);
dlat = lat2 - lat1;
dlon = lon2 - lon1;
a = sin(dlat/2).^2 + cos(lat1) .* cos(lat2) .* sin(dlon/2).^2;
c = 2 * atan2(sqrt(a), sqrt(1 - a));
distance_m = R * c; % Distance in meters

% Reference Distance and Improved PL_d0 Estimate
d0 = 30; 
idx_d0 = distance_m <= d0;
PL_d0 = sum(rsrp_values(idx_d0) ./ distance_m(idx_d0)) / sum(1 ./ distance_m(idx_d0)); % Weighted Mean

% Define Dual-Slope Path Loss Model
%%
% Define Log-Normal Shadowing Model
% Compute local RSRP standard deviation in bins
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off');
% Plot New Model
% Define Dual-Slope Path Loss Model
%multi_slope_model = @(p, d) PL_d0 - 10 * p(1) * log10(d / d0) - p(2) * (d > d0) ...
    %                         - p(3) * (d > 2*d0) - p(4) * (d > 3*d0);
dual_slope_model = @(p, d,d0_test,PL_d0) ...
   (d <= p(3)) .* (PL_d0 - 10 * p(1) * log10(d / d0)) + ... % Near Distance Slope
   (d > p(3))  .* (PL_d0 - 10 * p(2) * log10(d / d0)) ; % Far Distance Slope
% dual_slope_model = @(p, d, d0, PL_d0) PL_d0 - 10 * p(1) * log10(d / d0) - p(2) * (d > d0) - p(3) + p(4);
%dual_slope_model = @(p, d, d0_test, PL_d0) PL_d0 - 10 * p(1) * log10(d / d0) - p(2) * (d > d0) - p(3) + p(4);

% Optimize [n1, n2, db] using fminunc
%cost_function = @(p) sum((dual_slope_model(p, distance_m) - rsrp_values).^2);
%p_opt = fminunc(cost_function, [2, 4, 100], options); % Initial Guess: [n1=2, n2=4, db=100m]


% Predict RSRP with Dual-Slope Model

% Compute RMSE
%rmse = sqrt(mean((rsrp_values - rsrp_pred).^2));
d0_values = [ 700, 90, 20,12,15]; % Test different reference distances
best_rmse = inf;
for d0_test = d0_values
    PL_d0 = mean(rsrp_values(distance_m <= d0_test)); % Adjust based on new d0
   % cost_function = @(p,d) sum((dual_slope_model(p, distance_m, d0_test, PL_d0) - rsrp_values).^2);
   cost_function = @(p, d) dual_slope_model(p, d, d0_test, PL_d0);
   
   %p_opt = fminunc(cost_function, [2, 4, 100]); 
    %p_opt = lsqcurvefit(cost_function, [2, 4, 100], distance_m, rsrp_values);
    % Set initial guess for parameters
    %p_initial = [2, 4, 100];
    p_initial_values = [
    2, 4, 100;
    2.5, 3.8, 120;
    1.8, 5, 90;
];
    % Set lower and upper bounds for optimization
    lb = [1, 2, 80];  % Ensure parameters stay within a reasonable range
    ub = [4, 8, 180];
    distance_m = distance_m(:);
    rsrp_values = rsrp_values(:);
    % Optimize path loss parameters using lsqcurvefit
    for p_init = p_initial_values'
    p_opt = lsqcurvefit(cost_function, p_init, distance_m, rsrp_values, lb, ub);
    end
    %p_opt = lsqcurvefit(cost_function, p_initial, distance_m, rsrp_values, lb, ub);
    rsrp_pred = dual_slope_model(p_opt, distance_m,d0_test,PL_d0);
    rmse_test = sqrt(mean((dual_slope_model(p_opt, distance_m,d0_test,PL_d0) - rsrp_values).^2));
    if rmse_test < best_rmse
        best_rmse = rmse_test;
        best_d0 = d0_test;
        best_p_opt = p_opt;
        
    end
end
disp(['Best d0: ', num2str(best_d0)]);
disp(['New RMSE: ', num2str(best_rmse)]);
% Display Optimized Parameters
disp(['Optimized Near Path Loss Exponent (n1): ', num2str(p_opt(1))]);
disp(['Optimized Far Path Loss Exponent (n2): ', num2str(p_opt(2))]);
disp(['Optimized Breakpoint Distance (db): ', num2str(p_opt(3))]);
disp(['Reduced RMSE with Dual-Slope Model: ', num2str(best_rmse)]);

% Plot Updated Model
figure;
scatter(distance_m, rsrp_values, 'b', 'filled', 'DisplayName', 'Actual RSRP');
hold on;
scatter(distance_m, rsrp_pred, 'r', 'filled', 'DisplayName', 'Predicted RSRP (Dual-Slope)');
xlabel('Distance (m)');
ylabel('RSRP (dB)');
legend show;
title('Optimized Dual-Slope Path Loss Model');
grid on;

% Optimize Path Loss Exponent Using fminunc
