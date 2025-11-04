## Signal-Coverage-Maps-Using-Interpolation-techniques-and-ML
Signal quality estimation using ML techniques

# Abstract:
Signal coverage metrics like SINR and RSRP using spatial interpolation techniques. By using ML techniques which enables to estimate or predict RSRQ for various locations in given terrain.

Machine learning models like random forest and deep neural networks work well.

# Given map area
RSRP corresponding to given latitudes, longitudes from given dataset.
![image](https://github.com/user-attachments/assets/04c2920e-740f-4152-804d-74965f4474f9)

# Description about the dataset

The provided data, sourced from a delimited file named cleaned2_network_data.xlsx - Sheet1.csv, contains typical metrics collected during mobile network performance measurements, such as a drive test or MDT logging session. The structure of the data includes the following fields: Lattitude, Longitude, Altitude, Speed, RSRP_54_, SINR_54_, and Grid.   

The geospatial columns—Lattitude and Longitude—provide the horizontal position of the measurement device. The data exhibits high spatial density within specific localized clusters (e.g., numerous points grouped around 49.4239X N, 7.753X E). This dense, path-specific sampling is highly advantageous for geostatistical techniques. 

The provided data, sourced from a delimited file named cleaned2_network_data.xlsx - Sheet1.csv, contains typical metrics collected during mobile network performance measurements, such as a drive test or MDT logging session. The structure of the data includes the following fields: Lattitude, Longitude, Altitude, Speed, RSRP_54_, SINR_54_, and Grid.   

The geospatial columns—Lattitude and Longitude—provide the horizontal position of the measurement device. The data exhibits high spatial density within specific localized clusters (e.g., numerous points grouped around 49.4239X N, 7.753X E). This dense, path-specific sampling is highly advantageous for geostatistical techniques designed to leverage spatial autocorrelation.

The provided data, sourced from a delimited file named cleaned2_network_data.xlsx - Sheet1.csv, contains typical metrics collected during mobile network performance measurements, such as a drive test or MDT logging session. The structure of the data includes the following fields: Lattitude, Longitude, Altitude, Speed, RSRP_54_, SINR_54_, and Grid.   

The geospatial columns—Lattitude and Longitude—provide the horizontal position of the measurement device. The data exhibits high spatial density within specific localized clusters (e.g., numerous points grouped around 49.4239X N, 7.753X E). This dense, path-specific sampling is highly advantageous for geostatistical techniques designed to leverage spatial autocorrelation.



## Table of Contents

- [Channel Propagation method](Channel_propagation.md)
- [Kriging and Ordinary Kriging method](steps_in_kriging.md)
- [ML in RSRP prediction](ML_in_RSRP_prediction.md)
   - [end to end pipeline for RSRP prediction using various ML models code](matlab_codes/RSRP_prediction_using_ML.mlx)

## How to run?

Download the RSRP_prediction_using_ML.mlx raw file and ensure the dataset i.e. cleaned2_network_data.xlsx is in the same folder.
Run this.




## Inferences

RMSE is around 3 to 6 dbm by using various machine learning models which is optimum for building signal coverage maps for RSRP prediction.
Other techniques like channel propagation methods which use channel modelling offer around 10 to 11 dBm RMSE in predicting RSRP, also with Kriging technique but RMSE obtained is quite high for RSRP prediction.
ML techniques are useful with decent RMSE as for this project dataset size is somewhat large.
Other techniques work well for small and moderate dataset sizes.

