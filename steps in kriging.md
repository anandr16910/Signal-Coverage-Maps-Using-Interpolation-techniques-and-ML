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


$$ \gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} \left[ Z(x_i) - Z(x_i + h) \right]^2 $$

1️⃣ Experimental Variogram (γ(h))

The semi-variogram measures the spatial dependence between observations. It is computed as:


$$ \gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} \left[ Z(x_i) - Z(x_i + h) \right]^2  $$


Where:
	•	 $$ \gamma(h)\ $$ = semi-variance for lag distance  h .
	•	 N(h)  = number of observation pairs separated by  h .
	•	 $$ Z(x_i) $$ = measured value at location  x_i .

 2️⃣ Variogram Model (Exponential, Spherical, Gaussian)

To fit a theoretical model to the experimental variogram:

Exponential Model:


$$ \gamma(h) = C_0 + C \left(1 - e^{-h/a}\right)  $$


Spherical Model:


$$ \gamma(h) =
\begin{cases}
C_0 + C \left[ \frac{3}{2} \frac{h}{a} - \frac{1}{2} \left( \frac{h}{a} \right)^3 \right], & 0 \leq h \leq a \\
C_0 + C, & h > a
\end{cases}  $$


Gaussian Model:


$$ \gamma(h) = C_0 + C \left(1 - e^{-(h/a)^2}\right) $$


Where:
	•	$$ C_0  = nugget (small-scale variation). $$
	•	 C  = sill (total variance).
	•	 a  = range (distance where spatial correlation vanishes).


# Simulations:
<br>

![variogram_comparison](https://github.com/user-attachments/assets/c882d744-8aae-42da-ab12-93bbbe25ab09)
<br>

![fitted_variogram](https://github.com/user-attachments/assets/cb428e2a-b088-4f4b-ae52-75736546f0b0)




