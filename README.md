#**Data-Driven-Sports-Score-Forecasting**
---

**Problem Statement**: <br/>
The objective of this project is to apply data mining techniques to analyze and predict the total runs scored in an innings of cricket match. This involves extracting relevant patterns and insights from the dataset, preprocessing the data to handle inconsistencies and irrelevant information, and employing various regression models to make accurate predictions. The primary goal is to explore different data mining methodologies to identify the most effective model for predicting match outcomes based on historical data. This project will utilize a combination of data cleaning, transformation, feature engineering, and machine learning techniques to achieve robust predictive performance.

---

**Dataset**: <br/>
The dataset used for this analysis is a cricket match dataset that contains several attributes such as Match Id, Batting team name, Bowling team name , date, Venue, Batsman, Bowler, runs, runs, wickets, overs , runs scored in last 5 overs, wickets taken in last 5 overs, striker, non-striker and total.80% of dataset used for training and 20% of data used for testing.

---

**Techniques Used**:

1.Data Preprocessing<br/>
2.Data Cleaning<br/>
3.Feature Engineering<br/>
4. Correlation Analysis<br/>
5. Data Splitting<br/>
6. Model Training and Evaluation<br/>
   
  i. Linear Regression: A basic regression technique that models the relationship between the dependent variable and one or more independent variables by fitting a linear equation to observed data.<br/>
            •	Performance Metrics: Evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).<br/>

  ii.Random Forest Regression: An ensemble learning method that constructs multiple decision trees during training and outputs the mean prediction of individual trees to improve predictive accuracy and control over-fitting.<br/>
           •	Performance Metrics: Similar to Linear Regression, evaluated using MAE, MSE, and RMSE.<br/>
           
  iii.Neural Network Regression : A type of artificial neural network used for regression tasks. In this case, an MLP (Multilayer Perceptron) with a logistic activation function was trained.
       <br/>
           •	Performance Metrics: Evaluated using MAE, MSE, and RMSE.<br/>
7. Model Performance Comparison:
Visualization: The performance of different models was compared by plotting their accuracy scores using Seaborn’s barplot, providing a clear visual representation of how each model performed in terms of predictive accuracy.

---

**Algorithm Used :**

•	Linear Regression

•	Random Forest Regression

•	Neural Network Regression

---
To run the file:

#1. **Install dependencies**:

pip install -r requirements.txt

#2.**Run the file**:

python Data-Driven-Sports-Score-Forecasting.py
