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

% Fit an exponential variogram model
range = max(bins) / 3;  % Initial guess for range parameter
sill = max(gamma);       % Initial guess for sill
nugget = min(gamma);     % Initial guess for nugget

fitted_gamma = exponential_variogram(bins, nugget, range, sill);

% Plot results
figure;
scatter(bins, gamma, 'bo', 'DisplayName', 'Experimental Variogram'); hold on;
plot(bins, fitted_gamma, 'r-', 'LineWidth', 2, 'DisplayName', 'Exponential Model');
xlabel('Lag Distance');
ylabel('Semivariance');
legend;
title('Exponential Variogram Fit');
grid on;

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