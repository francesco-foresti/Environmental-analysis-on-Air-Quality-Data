# Environmental analysis on Air Quality Data
A University Project on the Course 'Statistics and Stochastic Modeling' by the University of Bergamo. 

The final report can be found in the repo or [HERE](./Report.pdf)

## Data Description
The dataset contains hourly air quality data from 01/01/2017 to 31/12/2020 for the Bergamo area, including: Precipitation, Temperature, Humidity, Radiation, Wind Speed, Wind Direction, Nitric Oxide (NO), Nitrogen Dioxide (NO<sub>2</sub>), Carbon Monoxide (CO), Ozone (O<sub>3</sub>)

## Analysis and Modeling
1. Correlation Analysis
    - Calculated the correlation matrix to identify relationships between variables.
    - Found strong correlations between O<sub>3</sub> and variables like temperature, humidity, and NO<sub>2</sub>.

2. Linear Regression Models
    - Built simple linear regression models to study the impact of individual regressors on O<sub>3</sub>.
    - Developed multiple linear regression models to improve prediction accuracy.

3. Basis Functions
    - Explored Fourier and B-Spline basis functions to model temperature trends over time.
    - Used cross-validation to select the optimal number of bases and avoid overfitting.

4. Model Evaluation
   - Evaluated models using Mean Squared Error (MSE) and R-squared (R<sup>2</sup>) metrics.
   - Identified the best regression models for predicting O<sub>3</sub> levels and temperature trends.
