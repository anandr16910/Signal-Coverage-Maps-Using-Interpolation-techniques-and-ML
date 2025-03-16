clc; clear; close all;
data = readtable('cleaned2_network_data.xlsx');
latitudes = str2double(data.Lattitude(:));
longitudes = str2double(data.Longitude(:));

utmstruct = defaultm('utm');
utmstruct.zone = '32U';  % Change as needed
utmstruct.geoid = wgs84Ellipsoid;  
utmstruct = defaultm(utmstruct);

[x, y] = projfwd(utmstruct, latitudes, longitudes);

rsrp_values = str2double(data.RSRP_54_(:));

latitudes = x;
longitudes = y;

%rsrp_values = normalize(rsrp_values); 
%rsrp_
% Compute the experimental variogram
[bins, gamma] = compute_variogram(x, y, rsrp_values, 50);
%% optimized model 


% Compute Experimental Variogram
%[gamma, bins] = variogram(x, y, z, 0);  % Custom function or use variogramfit

% Initial Guess for Parameters: [nugget, range, sill]
init_params = [min(gamma), max(bins)/3, max(gamma)];  

% Fit models using optimization
exp_params = fminsearch(@(p) variogram_error(p, bins, gamma, 'exponential'), init_params);
sph_params = fminsearch(@(p) variogram_error(p, bins, gamma, 'spherical'), init_params);
gau_params = fminsearch(@(p) variogram_error(p, bins, gamma, 'gaussian'), init_params);

% Compute Fitted Models
gamma_exp = exponential_variogram(bins, exp_params(1), exp_params(2), exp_params(3));
gamma_sph = spherical_variogram(bins, sph_params(1), sph_params(2), sph_params(3));
gamma_gau = gaussian_variogram(bins, gau_params(1), gau_params(2), gau_params(3));

% Plot Comparison
figure; hold on;
scatter(bins, gamma, 'bo', 'DisplayName', 'Experimental Variogram');
plot(bins, gamma_exp, 'r-', 'LineWidth', 2, 'DisplayName', 'Exponential Model');
plot(bins, gamma_sph, 'g-', 'LineWidth', 2, 'DisplayName', 'Spherical Model');
plot(bins, gamma_gau, 'b-', 'LineWidth', 2, 'DisplayName', 'Gaussian Model');

xlabel('Lag Distance'); ylabel('Semivariance');
title('Experimental vs. Modeled Variograms');
legend show; grid on;
%%
% Fit an exponential variogram model
range = max(bins) / 3;  % Initial guess for range parameter
sill = max(gamma);       % Initial guess for sill
nugget = min(gamma);     % Initial guess for nugget

fitted_gamma = exponential_variogram(bins, nugget, range, sill);
sph_gamma = spherical_variogram(bins,nugget,range,sill);
gaussian_gamma = gaussian_variogram(bins,nugget,range,sill);
% Plot results
figure;
scatter(bins, gamma, 'bo', 'DisplayName', 'Experimental Variogram'); hold on;
plot(bins, fitted_gamma, 'r-', 'LineWidth', 2, 'DisplayName', 'Exponential Model');
hold on;
plot(bins, sph_gamma, 'g-', 'LineWidth', 2, 'DisplayName', 'Spherical Model');
hold on;
plot(bins, gaussian_gamma, 'b-', 'LineWidth', 2, 'DisplayName', 'Gaussian Model');
xlabel('Lag Distance');
ylabel('Semivariance');
legend;
title('Exponential sph guassian Variogram Fit');
grid on;
hold off;

%% Function: Compute Experimental Variogram
function [bins, gamma] = compute_variogram(x, y, values, nrbins)
    n = length(x);
    distances = [];
    semivariances = [];

    % Compute pairwise distances and squared differences
    for i = 1:n
        for j = i+1:n
            d = sqrt((x(i) - x(j))^2 + (y(i) - y(j))^2);
            gamma_ij = 0.5 * (values(i) - values(j))^2;
            distances = [distances; d];
            semivariances = [semivariances; gamma_ij];
        end
    end

    % Bin distances
    max_dist = max(distances);
    bin_edges = linspace(0, max_dist, nrbins+1);
    bins = zeros(nrbins, 1);
    gamma = zeros(nrbins, 1);
    
    for k = 1:nrbins
        in_bin = (distances >= bin_edges(k)) & (distances < bin_edges(k+1));
        if sum(in_bin) > 0
            bins(k) = mean(distances(in_bin));
            gamma(k) = mean(semivariances(in_bin));
        else
            bins(k) = NaN;
            gamma(k) = NaN;
        end
    end
    
    % Remove NaNs
    valid_idx = ~isnan(bins) & ~isnan(gamma);
    bins = bins(valid_idx);
    gamma = gamma(valid_idx);
end

%% Function: Exponential Variogram Model
function gamma_fit = exponential_variogram(bins, nugget, range, sill)
    gamma_fit = nugget + sill * (1 - exp(-bins / range));
end
%% function: spherical variogram model
function gamma_fit = spherical_variogram(bins,nugget,range,sill)
    gamma_fit = zeros(size(bins)); % Initialize output

    % Apply spherical model conditionally
    inside_range = bins <= range;
    gamma_fit(inside_range) = nugget + sill * (1.5 * (bins(inside_range) / range) - 0.5 * (bins(inside_range) / range).^3);
    
    % For bins > range, set to sill
    gamma_fit(~inside_range) = nugget + sill;;
end

%% function: gaussian model:
function gamma_fit = gaussian_variogram(bins,nugget,sill,range)
    gamma_fit = nugget + sill * (1 - exp(-(bins / range).^2));
end 
%% function: variogram error:
function error_val = variogram_error(params, bins, gamma_exp, model_type)
    nugget = params(1); range = params(2); sill = params(3);
    switch model_type
        case 'exponential'
            gamma_fit = exponential_variogram(bins, nugget, range, sill);
        case 'spherical'
            gamma_fit = spherical_variogram(bins, nugget, range, sill);
        case 'gaussian'
            gamma_fit = gaussian_variogram(bins, nugget, range, sill);
    end
    error_val = sum((gamma_exp - gamma_fit).^2); % Least squares error
end

%% Function: Convert Lat/Lon to UTM
function [x, y, zone] = deg2utm(lat, lon)
    k0 = 0.9996;
    a = 6378137.0; % WGS84 major axis
    e = 0.0818192; % WGS84 eccentricity

    % UTM Zone Calculation
    zone = floor((lon + 180) / 6) + 1;

    % Convert degrees to radians
    lat = deg2rad(lat);
    lon = deg2rad(lon);
    
    % Central meridian of the zone
    lon0 = deg2rad(-183 + 6 * zone);

    N = a ./ sqrt(1 - e^2 * sin(lat).^2);
    T = tan(lat).^2;
    C = e^2 ./ (1 - e^2) .* cos(lat).^2;
    A = cos(lat) .* (lon - lon0);

    M = a * ((1 - e^2/4 - 3*e^4/64 - 5*e^6/256) * lat ...
        - (3*e^2/8 + 3*e^4/32 + 45*e^6/1024) .* sin(2*lat) ...
        + (15*e^4/256 + 45*e^6/1024) .* sin(4*lat) ...
        - (35*e^6/3072) .* sin(6*lat));

    x = k0 * N .* (A + (1 - T + C) .* A.^3 / 6 + (5 - 18*T + T.^2 + 72*C - 58*e^2) .* A.^5 / 120) + 500000;
    y = k0 * (M + N .* tan(lat) .* (A.^2 / 2 + (5 - T + 9*C + 4*C.^2) .* A.^4 / 24 ...
        + (61 - 58*T + T.^2 + 600*C - 330*e^2) .* A.^6 / 720));

    % Adjust for Southern Hemisphere
    y(lat < 0) = y(lat < 0) + 10000000;
end
