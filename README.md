## Signal-Coverage-Maps-Using-Interpolation-techniques-and-ML
Signal quality estimation using ML techniques

# Abstract:
Signal coverage metrics like SINR and RSRP using spatial interpolation techniques. By using ML techniques which enables to estimate or predict RSRQ for various locations in given terrain.

Machine learning models like random forest and deep neural networks work well.

# Given map area
RSRP corresponding to given latitudes, longitudes from given dataset.
![image](https://github.com/user-attachments/assets/04c2920e-740f-4152-804d-74965f4474f9)

# Description about the dataset

The provided data, sourced from a delimited file named **cleaned2_network_data.xlsx** - Sheet1.csv, contains typical metrics collected during mobile network performance measurements, such as a drive test or MDT logging session. The structure of the data includes the following fields: Lattitude, Longitude, Altitude, Speed, RSRP_54_, SINR_54_, and Grid.   

The geospatial columns—Lattitude and Longitude—provide the horizontal position of the measurement device. The data exhibits high spatial density within specific localized clusters (e.g., numerous points grouped around 49.4239X N, 7.753X E). This dense, path-specific sampling is highly advantageous for geostatistical techniques. 

Raw latitude and longitude values serve poorly as direct inputs for predictive modeling due to their non-linear nature. Effective geospatial modeling requires calculating derived metric features that quantify spatial relationships.   

The first set of essential transformations involves establishing the physical relationship between the measurement device and the signal source. This is contingent upon correctly mapping the Grid identifiers to their corresponding Base Station (BS) coordinates. Once the BS location is known, calculating the distance between the measurement point and the BS is mandatory. This metric should be derived using accurate geometric models, such as Haversine distance, and projected into a planar system (e.g., UTM) for accurate distance calculations in meters. Distance is fundamental, as signal strength attenuation is logarithmically related to the separation between transmitter and receiver (path loss models). 

The choice of prediction architecture must balance high prediction accuracy with computational efficiency, particularly to enable the rapid regeneration of the predictive radio map. Modern approaches range from classic geostatistical methods optimized for spatial autocorrelation to advanced deep learning models that learn complex, non-linear feature relationships.

 - Geostatistical Interpolation Approaches
Geostatistical methods are explicitly designed to model spatially correlated data, a characteristic inherent in radio signal strength measurements, where adjacent locations often exhibit similar signal attenuation due to shadowing effects. The fact that the provided dataset exhibits high local density along specific measurement paths validates the strength of these approaches.   

 - Limitations of Simple Interpolation Methods
Simple linear models, such as Inverse Distance Weighting (IDW) interpolation, rely solely on weighting nearby measurement points inversely proportional to distance. While fast for inference , IDW fails to account for directionality, data clustering, and the specific statistical structure of wireless propagation environments. Consequently, IDW results in inferior prediction accuracy compared to methods that explicitly model the underlying spatial continuity.   

 - Advanced Geostatistical Modeling: Kriging and Regression-Kriging
Kriging is recognized as an optimal linear predictor that estimates values based on a variogram, a function that models the degree of spatial autocorrelation across distances. Ordinary Kriging (OK) minimizes prediction error and accounts for data clustering, reducing prediction bias. However, OK assumes stationarity, meaning the average signal level trend is constant across the service area. In reality, RSRP is affected by massive deterministic trends related to distance, clutter, and elevation—factors that violate the stationarity assumption
  





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

