Steps:
1. Preprocess Data
	•	Check for missing or anomalous values in RSRP, SINR, and geospatial data.
	•	Normalize RSRP and SINR values for consistent scaling (optional).

2. Generate Coverage Maps


![image](https://github.com/user-attachments/assets/051c84fa-0676-4e6e-ad82-ebfe38d8504a)
![image](https://github.com/user-attachments/assets/c1f03a96-f1b7-4d61-abbe-a18b4681dddb)


2.1 Baseline Map (Geospatial Interpolation)
	•	Interpolate RSRP values across the latitude and longitude grid using methods like Kriging or Inverse Distance Weighting (IDW).
	•	Generate a continuous coverage map over the geospatial area.

 difference between Kriging and Ordinary Kriging: 
 

 |Feature	|Kriging (General)   |	 Ordinary Kriging (OK)  |
|-----|-----|-------|
|Assumption	|Uses a known mean function for spatial trends.|	Assumes an unknown but constant mean over the study area. |
|Error Impact |	Can reduce error if the mean function is well-defined.|	Might have higher error if the assumption of constant mean is incorrect.|
|Bias	|Can be biased if the assumed trend function is incorrect.	|Less biased but may have higher variance in predictions.|
|Complexity |	More complex (requires defining a trend model).|	Simpler (no need for a trend model).|
 
 
Steps in kriging:


\gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} \left[ Z(x_i) - Z(x_i + h) \right]^2


2.2 Machine Learning-Based Map
	•	Train a Random Forest model using the geospatial coordinates (Lattitude, Longitude, Altitude) and speed as features, with RSRP as the target.
	•	Predict RSRP values for a dense grid of latitude and longitude points.


