# Signal-Coverage-Maps-Using-Interpolation-techniques-and-ML
Signal quality estimation using ML techniques

# abstract:
Signal coverage metrics like SINR and RSRQ using spatial interpolation techniques. By using ML techniques which enables to estimate or predict RSRQ for various locations in given terrain.

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
Steps in kriging:


2.2 Machine Learning-Based Map
	•	Train a Random Forest model using the geospatial coordinates (Lattitude, Longitude, Altitude) and speed as features, with RSRP as the target.
	•	Predict RSRP values for a dense grid of latitude and longitude points.
