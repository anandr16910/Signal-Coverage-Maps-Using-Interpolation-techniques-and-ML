## Signal-Coverage-Maps-Using-Interpolation-techniques-and-ML
Signal quality estimation using ML techniques

# Abstract:
Signal coverage metrics like SINR and RSRQ using spatial interpolation techniques. By using ML techniques which enables to estimate or predict RSRQ for various locations in given terrain.

Machine learning models like random forest and deep neural networks work well.

# Given map area

![image](https://github.com/user-attachments/assets/04c2920e-740f-4152-804d-74965f4474f9)



## Table of Contents

- [Channel Propagation method](Channel_propagation.md)
- [Kriging and Ordinary Kriging method](steps_in_kriging.md)
- [ML in RSRP prediction](ML_in_RSRP_prediction.md)

## Inferences

RMSE is around 3 to 6 dbm by using various machine learning models which is optimum for building signal coverage maps for RSRP prediction.
Other techniques like channel propagation methods which use channel modelling offer around 10 to 11 dBm RMSE in predicting RSRP, also with Kriging technique but RMSE obtained is quite high for RSRP prediction.
ML techniques are useful with decent RMSE as for this project dataset size is somewhat large.
Other techniques work well for small and moderate dataset sizes.

