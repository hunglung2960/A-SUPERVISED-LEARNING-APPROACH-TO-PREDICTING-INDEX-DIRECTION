# A SUPERVISED LEARNING APPROACH TO PREDICTING STOCK INDEX DIRECTION

<!-- Table of Content -->
- [Introduction](#introduction)
- [Machine Learning Literature Review](#machine-learning-literature-review)
- [Methodology](#methodology)
	- [Data, Data Source and Sample Size](#data-data-source-and-sample-size)
	- [Data Cleaning](#data-cleaning)
	- [Feature Engineering](#feature-engineering)
	- [Stationarity Check and Transformation](#stationarity-check-and-transformation)
	- [Method](#method)
- [Result](#result)
	- [Overall Forecasting Performance](#overall-forecasting-performance)
	- [In Practice Cross-Validation Performance](#in-practice-cross-validation-performance)
	- [Interpretability of Models](#interpretability-of-models)
- [Discussion](#discussion)
	- [Most successful downward direction predictors](#most-successful-downward-direction-predictors)
	- [Most successful upward and long-horizon predictors](#most-successful-upward-and-long-horizon-predictors)
	- [Performance of technical indicators](#performance-of-technical-indicators)
	- [Usefulness of non-linear variables](#usefulness-of-non-linear-variables)
	- [ML techniques comparison](#ml-techniques-comparison)
- [Future Research](#future-research)
	- [A new benchmark to beat other than traditional econometric model](#a-new-benchmark-to-beat-other-than-traditional-econometric-model)
	- [Suggestion of predictors](#suggestion-of-predictors)
	- [Maintaining the good forecasting elements](#maintaining-the-good-forecasting-elements)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Appendix](#appendix)
- [Reference](#reference)
- [Remarks](#remarks)


<!-- Main Content -->
<!-- Introduction -->
# Introduction

An extensive literature has studied the performance of a variety of forecasting models for stock premium direction. Many researchers have shown that the use of econometric models for directional prediction of stock premium produces weak predictive power (Chevapatrakul, 2013; Leitch & Tanner, 1991; Leung, Daouk, & Chen, 2000; Nyberg, 2011; Pesaran & Timmermann, 1995; Pönkä, 2016).

Iworiso & Vrontos (2020) have applied a plethora of machine learning techniques on a set of financial variables obtained from Amit Goyal's website ranging from January 1960 to December 2016. They have demonstrated that ML models outperform the econometric models.

This project will include additional sets of technical indicators and macroeconomic variables as predictors as well as their non-linearised versions. We aim to study 1) whether the inclusion of technical indicators and macroeconomic variables will enhance the forecasting performance in OOS, 2) whether the inclusion of non-linear variables will increase the forecasting performance in OOS and 3) which ML method and its modification will have the best performance.


<!-- ML Literature Review -->
# Machine Learning Literature Review
Many researchers have employed econometric models including autoregressive moving average with exogenous inputs models (ARMAX), vector autoregressive-generalized autoregressive conditional heteroskedasticity models (VAR-GARCH), logistic regression model, and static/dynamic probit models to forecast the direction of stock premium (Chevapatrakul, 2013; Leitch & Tanner, 1991; Leung, Daouk, & Chen, 2000; Nyberg, 2011; Pesaran & Timmermann, 1995; Anatolyev and Gospodinov, 2010). Among all, Nyberg (2011) has shown that dynamic probit models have outperformed the previously mentioned econometric models.

Departing from traditional models, Iworiso & Vrontos (2020) have shown that bagging, random forest, and boosting outperform the dynamic probit model among a plethora of machine learning techniques implemented on a set of financial variables obtained from Amit Goyal's website ranging from January 1960 to December 2016. Besides, Ballings, Poel, Hespeels & Gryp (2015) show that random forest is the best performer among the tree-based methods.

It is worth mentioning that Goyal & Welch (2007) argue that historical average excess stock return forecasts outperform predictive regressions. They even claim that "the profession has yet to find some variable that has meaningful and robust empirical equity premium forecasting power." Subsequently, Campbell & Thompson (2008) argue that once weak restrictions are imposed on the signs of coefficients and return forecasts predictive regressions could outperform the historical average. Iworiso & Vrontos (2020) show that penalised binary probit model also delivers better results than conventional econometric models. 

In sum, the literature has shown the good performance of tree-based methods, which will be our primary focus. The findings and argument from Goyal & Welch (2007), Campbell & Thompson (2008), and Iworiso & Vrontos (2020) have also inspired us to compare our forecast with unconditional mean and apply the LASSO logit model. As a part of an effort to compare between traditional models and ML techniques, we will also consider probit models as a representative for econometric models for ML methods to beat. 

<!-- Methodology -->
# Methodology
## Data, Data Source and Sample Size
In this project, other than financial variables commonly used by researchers to predict stock return direction, we will additionally include technical indicators and macroeconomic variables to investigate whether they can improve prediction performance.  We obtain a set of 17 financial variables from Professor Arim Goyal's website, a set of 10 technical indicators generated by Harvey et al. (2020), and a set of 19 macroeconomic variables selectively retrieved from the FRED-MD database.

The set of financial variables mainly includes types such as return, dividend, stock volatility, financial ratio, and interest rate variables for the period Y1871M01 to Y2019M12 totalled 1788 observations. Next, the technical indicators include 4 average indicators, 2 momentum indicators, and 4 on-balance volume indicators for the period 1Y974M12 to Y2015M12 totalled 493 observations. Additionally, the selection of macroeconomic variables consists of data in the category of money & credit, consumption, output, and labor data for the period Y1959M01 to Y2021M01 totalled 745 observations. 


## Data Cleaning
**Financial variables:** We observed the missing value for financial variables. We see that most variables have missing values from 1 to around 600. The variable csp has missing values from 1 to 796 and 1585 to 1788. We decided to drop the variable csp so that we can keep the observations for other variables from 1585 to 1788. After dropping csp and removing all missing values, we have observations from period Y1926M12 to Y2019M12. The variable Index from the dataset is S&P 500 Index. We change it into a binary variable which 1 indicates positive movement in direction and otherwise.

**Macroeconomic variables:** We filter the selected variables from the FRED-MD data set. After we removed all observation that contains missing values, we have observations from period Y1959M01 to Y2021M01. 

**Technical indicators:** The data set does not have any missing value, so we have observations from period Y1974M12 to Y2015M12.  We construct a dataset with the coexistence of all 3 sets of variables over a certain period, we obtain a dataset of 493 observations from the period Y1974M12 to Y2015M12 with 41 variables. 


## Feature Engineering
We created non-linear variables from continuous variables. We take square and cubic on all continuous variables. We generate an additional 60 non-linear variables, resulting in a total number of 101 predictors. We keep the maximum degree of polynomial to 3 to limit the number of predictors.

We will take four lags of predictors. Thus, we created additional 3 predictors for each predictor. After data cleaning, we have 404 predictors from Y1976M01 to Y2015M12 (480 observations).


## Stationarity Check and Transformation
We perform an ADF test to check the stationarity properties of the data. The null hypothesis assumes the presence of unit roots. We set the significant level at 0.01. If the p-value obtained is less than the significant level, we reject the null hypothesis. We need to be careful to note that if we reject the null hypothesis it does not indicate the series is definitely stationary. Therefore, we also plot the variable against Index and check whether we should suspect there is non-stationarity. If we suspect there is, we proceed to stationarity transformation. 

To remove trend and seasonality from the data and focus on their signal for forecasting, we perform stationarity transformation based on FRED's recommendations (methods specified in Table 1). Besides, we transform the rest of the data by differencing method. Finally, we check again with an ADF test to verify the removal of unit roots in the data. 



## Method
We have binary variables as our response. Thus, we are having a classification problem. We chose the machine learning algorithms that are more suited to classification problems. For tree-based methods, we implement classification tree, bagging, random forests, and boosting. For the benchmark model, we have logistic LASSO, which is the logistic function with L1 penalty, unconditional mean and probit model.

We have 480 observations from Y1976M01 to Y2015M12. We reserved 132 observations from Y2005M01 to Y2015M12 as our test set. Thus, we have 348 observations from Y1976M01 to Y2004M12 as our training set. 

We applied expanding window on all our methods and 4 forecast horizons, which are 1-step, 3-step, 6-step, and 12-step ahead forecasts. For 1-step ahead forecast, we will first have 348 observations as our training set and the rest as the training set. We fit the model with the training data and predict the first observation in our test set. Next, we will move the first observation in the test set into the training set. Now, we have 349 observations in our training set and 131 observations in our test set. We will predict the respective observation by repeating the process until we have 132 predicted values.

For the k-step ahead forecast, we will first have (348−𝑘+1) observations in our training set, where k is the number of steps and the rest as our training set. This is because we want to keep our predicted values constant as 132. We fit the model with the training data and predict the kth observations in our test set. Next, we will move the first observation in the test set into the training set. Now, we have (348−𝑘+2) observations in our training set. We will predict the respective observation by repeating the process until we have 132 predicted values. By using the expanding window technique, we will have our predicted value from Y2005M01 to Y2015M12, which is the same period as we set our test set. 

For the classification tree, we first try to fit the tree without any restrictions on the complexity parameter and observed the result. Next, we try to fit a big tree with the complexity parameter set as 0.0001. Then, we pruned the tree with the complexity parameter chosen by 1SE rule and absolute minimum CV. Since bagging is a special case of random forests where the number of variables randomly sampled as candidates at each split, mtry is equal to the number of predictors, P, we used random forests function with mrty set equal to P and the rest of the parameters as default.

For random forests, we have chosen mtry based on the rule of thumb which is the √P and the rest of the parameters as default. For boosting, since we are dealing with a binary response, we set our distribution to Bernoulli distribution. We have the number of boosting iterations set to 1000, the depth of each tree set to 5, and the penalty parameter set to 0.01. Then we will select the optimal number of iterations based on 10-fold CV. For logistic LASSO, we have our penalties set at default, and without doing post-lasso estimation. We are using rlassologit function from hdm package.

We can see that when we are doing expanding window, the smallest window we have is the 12-step's first prediction, we will have (348−12+1) = 337 observations in our training set. We have 404 predictors for 4 lags. When the number of predictors is larger than the number of observations, we will be overfitting the model. This is not a problem in both tree-based methods and logistic LASSO. However, it will be overfitting with the probit model. Hence, we reduce the number of predictors by including only linear variables as our predictors. Then, we fit the probit model with 41 predictors. 


<!-- Result -->
# Result
## Overall Forecasting Performance
**Figure 1**

| Forecast Horizon  | Model  | Misclassifications  | % Classified Correctly  | % 0&#39;s Classified Correctly  | %1&#39;s Classified Correctly  |
| --- | --- | --- | --- | --- | --- |
| One-Step  | Tree  | 53  | 59.8  | 0  | 97.5  |
| One-Step  | Bagging  | 57  | 56.8  | 29.4  | 74.1  |
| One-Step  | Random Forest with theoretical plug-ins  | 52  | 60.6  | 23.5  | 84.0  |
| One-Step  | Boosting  | 54  | 59.1  | 5.9  | 92.6  |
| One-Step  | Lasso Logit with theoretical plug-ins  | 50  | 62.1  | 3.9  | 98.8  |
| Three-Step  | Tree  | 51  | 61.4  | 3.9  | 97.5  |
| Three-Step  | Bagging  | 58  | 56.1  | 33.3  | 70.4  |
| Three-Step  | Random Forest with theoretical plug-ins  | 53  | 59.8  | 23.5  | 82.7   |
| Three-Step  | Boosting  | 50  | 62.1  | 7.8  | 96.3  |
| Three-Step  | Lasso Logit with theoretical plug-ins  | 52  | 60.6   | 3.9  | 96.3  |
| Six-Step   | Tree  | 50  | 62.1  | 2.0  | 100  |
| Six-Step | Bagging  | 58  | 56.1  | 33.3  | 70.4  |
| Six-Step  | Random Forest with theoretical plug-ins  | 55  | 58.3  | 25.5  | 79.0  |
| Six-Step | Boosting  | 58  | 56.1  | 3.9  | 88.9  |
| Six-Step | Lasso Logit with theoretical plug-ins  | 51  | 61.4  | 0  | 100  |
| Twelve-Step   | Tree  | 51  | 61.4  | 0  | 100  |
| Twelve-Step | Bagging  | 70  | 47.0  | 13.7  | 67.9  |
| Twelve-Step | Random Forest with theoretical plug-ins  | 59  | 55.3  | 13.7  | 81.5  |
| Twelve-Step | Boosting  | 57  | 56.8  | 3.9  | 90.1  |
| Twelve-Step | Lasso Logit with theoretical plug-ins  | 52  | 60.6  | 0  | 98.8  |



**Note:** Forecast performance for one-step, three-step, six-step, twelve-step forecast horizon. 

Overall, we found that most models, except for the twelve-steps ahead forecast using bagging, were able to predict direction at around 55.3 to 62.1 percent accuracy as in figure 1. The relative performance of the models remained similar across the different forecasting horizons with lasso logit and the basic tree having the most consistent performance throughout.  
 
In the one-step-ahead forecast horizon, we found that lasso logit had the highest performance with 50 misclassifications and 62.1% prediction accuracy. In contrast, bagging had 57 misclassifications and a 56.8% prediction accuracy.

In the three-step ahead forecast horizon, boosting had the highest performance with a similar performance to lasso logit in the one-step-ahead forecast. Again, bagging had the worst performance with 58 misclassifications and a 56.1% prediction accuracy. In the six-step ahead forecast horizon, the basic tree had the best performance with 50 misclassifications and the lasso logit a close second with 51 misclassifications. Bagging and random forest performed the worst with 58 misclassifications each.

In the twelve-step forecast, the basic tree performed the best with 51 misclassifications and the lasso logit being a close second again with 52 misclassifications. This time boosting yielded terrible performance with 70 misclassifications equivalent to a 47% prediction accuracy.

To put this in perspective, a repeated coin flip would have yielded on average a better performance than boosting in the twelve-step ahead forecast horizon. However, this decline in performance is also seen across all models on average with the basic tree and the lasso logit being the most impervious to this effect. This is expected since each step in our model represents a whole month, a twelve-step ahead forecast would mean using information from a year ago to forecast current changes in price. It is important to note that out of the four forecasting horizons, the basic tree splits 4.5%, 26.5%, 15.9%, and 0% respectively for each horizon out of the 132 forecast predictions.

Hence, the majority of the forecast predictions in the tree model rely on the unconditional mean to make forecasts. This reflects in part the poor performance of predictors and could also explain why the tree's performance remains relatively unaffected by increasing forecast horizons. Likewise, we also see the same phenomenon in the lasso logit. The number of times the lasso logit selects 0 variables increases across horizons, selecting no variables 0%, 0%, 15.9%, and 45.5% respectively across increasing horizons out of 132 forecast predictions. We find that models that rely more on the unconditional mean tend to predict better and more consistently. 
 
We also note the poor performance of bagging, boosting, and random forest compared to the basic tree. While boosting and random forest performs decently in the one-step and three-step ahead forecast horizon, performance drops significantly in the twelve-step ahead forecast horizon especially so when compared to the leading models. This could be because we choose to use either default values or theoretical plug-ins to reduce computational complexity rather than use recursive cross-validation to tune parameters. We further investigate this by comparing the same models on default settings to models tune by theoretically incorrect methods that are prevalent in practice. Despite the lack of tunning, we note that random forest consistently performs better than bagging across forecast horizons. We believe that this could be because the predictors included in the dataset are highly correlated. As a result, by decorrelating trees, random forest improves performance over bagging.


**Figure 2**
![Image 1 of Figure 2](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/5835a6861c36da5581097b69b0f3c52b58835921/Plot/bagging1.png)

**Note:** Top 10 class-specific importance measure for one-step forecast horizon bagging computed as mean decrease in accuracy. 

One common issue with all models in figure 1 is the lack of ability to correctly classify 0's. Across all models and forecast horizons, the prediction accuracy for classifying 0's remains low at around 0% to 33.3%. In contrast, the models were able to predict 1's correctly at around 67.9% up to 100%. In our study, we defined direction 1 as a positive price movement in the following period and 0 as either no change or a downwards movement in price. To understand why we examined the importance measure computed by mean decrease in accuracy aggregated across all forecasts in each forecast horizon. The mean decrease in accuracy represents how much accuracy the model loses by excluding each variable. We note that out of the 41 original predictors in the one-step forecast horizon, only 2 predictors would reduce accuracy if removed for class 0 as in figure 2. The other 39 predictors reported a negative mean decrease in accuracy, suggesting that accuracy would improve if the predictors were removed. This supports our observation that predictors in our dataset perform poorly in classifying 0's. We take a deeper look into which variables are important in selecting 0's later.

**Figure 3**
| Forecast Horizon  | Model  | Misclassifications  | % Classified Correctly  | % 0&#39;s Classified Correctly  | % 1&#39;s Classified Correctly  |
| --- | --- | --- | --- | --- | --- |
| One-Step  | Unconditional Mean  | 51  | 61.4  | 0  | 100  |
| One-Step  | Probit  | 64  | 51.5  | 35.3  | 61.7  |
| Three-Step  | Unconditional Mean  | 51  | 61.4  | 0  | 100  |
| Three-Step  | Probit  | 60  | 54.5  | 39.2  | 64.2  |
| Six-Step  | Unconditional Mean  | 51  | 61.4  | 0  | 100  |
| Six-Step  | Probit  | 66  | 50.0  | 37.3  | 58.0  |
| Twelve-Step  | Unconditional Mean  | 51  | 61.4  | 0  | 100  |
| Twelve-Step | Probit  | 73  | 44.7  | 23.5  | 58.0  |



**Note:** Benchmark performance for one-step, three-step, six-step, twelve-step forecast horizon. 

In general, there were few models that managed to beat the performance of the unconditional mean. Notably, the lasso logit with theoretical plug-ins in the one-step forecast horizon, boosting in the three-step forecast horizon, and the basic tree in the six-step forecast horizon. Even then, the increase in performance is marginal. Given the upwards trend of the S&P index over the last 30 years, it is not unusual for the unconditional mean to perform so well. The unconditional mean predicts every forecast to be 1, which guarantees decent performance. However, this performance may be unique to the S&P and may not hold for more volatile assets. Moreover, traditional models like the probit perform poorly compared to other models which can capture non-linearity. The probit model is outperformed by all other models, overtaking bagging as the worst-performing model. 




## In Practice Cross-Validation Performance
In practice, it is common to find K-fold CV or LOOCV being used on time-series data to tune parameters. Theoretically, this ignores the autocorrelation between observations and would result in poor performing parameters being used, especially when compared to proper methods like recursive cross-validation. However, we wanted to explore if improper cross-validation techniques could improve forecasting performance over using default values. To this end, we re-estimated several models using cross-validation methods other than recursive cross-validation to compare the performance against the default values. 


**Figure 4**
| Forecast Horizon  | Model  | Misclassifications  |
| --- | --- | --- |
| One-Step  | Tree  | 53  |
| One-Step  | Tree selected on CP CV Min  | 55  |
| One-Step  | Tree selected on CP CV 1SE  | 55  |
| One-Step  | Boosting  | 54  |
| One-Step  | Boosting selected on CV  | 52  |
| Three-Step   | Tree  | 51  |
| Three-Step | Tree selected on CP CV Min  | 61  |
| Three-Step | Tree selected on CP CV 1SE  | 53  |
| Three-Step | Boosting  | 50  |
| Three-Step | Boosting selected on CV  | 53  |
| Six-Step  | Tree  | 50  |
| Six-Step  | Tree selected on CP CV Min  | 65  |
| Six-Step  | Tree selected on CP CV 1SE  | 54  |
| Six-Step  | Boosting  | 58  |
| Six-Step  | Boosting selected on CV  | 55  |
| Twelve-Step  | Tree  | 51  |
| Twelve-Step  | Tree selected on CP CV Min  | 71  |
| Twelve-Step  | Tree selected on CP CV 1SE  | 66  |
| Twelve-Step  | Boosting  | 57  |
| Twelve-Step  | Boosting selected on CV  | 51  |




**Note:** Forecast performance for one-step, three-step, six-step, twelve-step forecast horizon using cross-validation & default values. 

Our findings yield mixed results as in figure 4. Selecting the subtree using either the cross-validated minimum or the 1 standard error rule consistently yields worse performance than compared to the default values for building trees. The difference is small in the one-step horizon forecast but grows quickly as the forecast horizon increases. On the other hand, in practice cross-validation for boosting works well. Performance generally either improves or remains close across forecasting horizons. Furthermore, this improvement in performance is more noticeable for the twelve-step horizon forecast. Despite the above findings, it is not clear whether the results are indicative of the usefulness of in practice cross-validation or merely a difference due to sampling error. However, performing cross-validation significantly increases the computational time required to estimate the models, especially so if the model is already very computationally expensive. For example, estimating boosting on default values take approximately 5 minutes, whereas the same estimation with cross-validation takes nearly 2 hours. Given the increase in computational complexity, it is questionable if the benefits as seen in boosting are worth it. 


## Interpretability of Models
Lastly, we analyze the importance of variables as deemed by our models. For tree-based methods, we aggregated the relative importance as measured by the corresponding packages for each forecast horizon. And for lasso logit, we counted the number of times each variable was selected within a given forecast horizon. For the purpose of this analysis, we broke down which variables were important in the short-term forecast as opposed to the long-term forecast. And we did so by comparing variable importance in the one-step horizon forecast to the latest attainable forecast horizon (usually twelve-step forecast horizon). Moreover, we separated variable importance by linear and non-linear predictors to see if non-linear predictors are important predictors.

**Figure 5** 

![Image 1 of Figure 5](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/c48847837768ba8a302a3bcfba357a3a7f42b6f4/Plot/tree1c.png)

![Image 2 of Figure 5](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/c48847837768ba8a302a3bcfba357a3a7f42b6f4/Plot/tree6c.png)

![Image 3 of Figure 5](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/c48847837768ba8a302a3bcfba357a3a7f42b6f4/Plot/ll1.png)

![Image 4 of Figure 5](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/c48847837768ba8a302a3bcfba357a3a7f42b6f4/Plot/ll12.png)


**Note:** Variable importance for tree one-step horizon (First Image) & six-step horizon (Second Image). Variables selected for lasso logit one-step horizon (Third Image) & twelve-step horizon (Fourth Image). Not exhaustive. 

We identified predictors that were important across multiple models in the one-step & latest step forecast horizon. Figure 5 shows the variable importance for our leading models but is not an exhaustive illustration of all importance measures used. It was found that inflation (infl), S&P 500 index returns with dividend (CRSP_SPvw), 12-month moving sums of earnings on the S&P 500 index (E12), real M2 money stock (M2REAL), the ratio of book value to market value for the Dow Jones Industrial Average (b/m), and industrial production of consumer goods (IPCONGD) were important in predicting short-term directions. These variables consist of a good mix of financial indicators as well as macroeconomic indicators.

As for important long-term predictors, we found inflation (infl), real M2 money stock (M2REAL), industrial production of nondurable materials (IPNMAT), industrial production of materials (IPMAT), industrial production of manufactured goods (IPMANSICS), industrial production index (INDPRO), and industrial production of fuels (IPFUELS) to be important. The long-term predictors consist mainly of macroeconomic indicators with some financial indicators. This could be supported by economic theory which believes that growth in the long term is fueled by production capabilities. We also found that once non-linear terms were re-introduced, the importance variables were dominated by non-linear predictors. This suggests that nonlinear models that can capture nonlinearity and handle high dimensions may be better suited to forecasting S&P index direction than conventional linear models. 

**Figure 6** 

![Image 1 of Figure 6](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/c48847837768ba8a302a3bcfba357a3a7f42b6f4/Plot/rf1.png)

**Note:** Top 10 class-specific importance measure for one-step forecast horizon random forest computed as mean decrease in accuracy. 

The importance measure by the randomforest package computes variable importance as mean decrease in accuracy. This is useful for us to understand which variables are important in predicting across classes 0 & 1. And we do so by comparing across importance measures for random forest and bagging as in figure 2 and figure 6. Results reveal that there are many variables important for classifying 1's but few that help classify 0's. This explains why most of our models perform well in predicting 1's but terribly in predicting 0's. To put things into comparison, all 10 predictors ranked by highest relative importance were found to decrease accuracy for 1's classification if removed for the one-step forecast of horizon random forest. Whereas we find 9 predictors in the top 10 for classifying 0's that would improve accuracy if removed.  


**Figure 7**

![Image 1 of Figure 7](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/c48847837768ba8a302a3bcfba357a3a7f42b6f4/Plot/Rplot.png)

**Note:** E12 & S&P Index time series. E12 was scaled by the ratio of means for clearer visibility. 

The one variable that did was important for classifying 0's is the 12-month moving sums of earnings on the S&P 500 index (E12). We find that this predictor also shows up repeatedly across forecasting horizons and in the variable importance for bagging. E12 was found to be able to predict downwards movement in S&P index price relatively well as shown in figure 7. And in 2007 and 2014, E12 was able to predict the fall in the S&P Index price before the actual fall in price. Adding more predictors to the dataset that are better able at classifying 0's may be essential to improving model performance. 


<!-- Discussion -->
# Discussion
## Most successful downward direction predictors
In response to our 1st research question about the relative importance of predictors, we start with our most surprising finding. It indicates that E12 (12-month moving sums of earnings on the S&P 500 index) is the only useful predictor for the downward direction of S&P index price among our 39 financial, macroeconomic, and technical indicators in all 4 horizons. Kinney, Burgstahler & Martin (2002) has suggested that there is a strong positive relationship between earnings and stock returns. Skinner & Sloan has shown that the share prices of growth stocks are more likely to fall sharply when they experience negative earnings surprises. Informed by behavioral economic research, Myopic Loss Aversion (MLA) theory suggests that an agent is loss-averse if he is more aware of losses than gains of equal size. Haigh & List (2005) has found that professional traders exhibit MLA-consistent behaviors, even greater than unsophisticated investors such as undergraduate students. It is plausible that E12 is especially instrumental to explain MLA individual behaviors because of its signals of merely 12 months of corporate business performance rather than a longer horizon. In sum, several financial and behavioral research has provided justifications for the dominant role of E12's predictive power in the negative direction. 

To extend our discussion on the downward predictive power of E12 to multi-step ahead horizons, we introduce the concept of Post-earnings Announcement Drift (PEAD). It means that "stock prices drift in the direction of an earnings surprise for several months after earnings are announced" (Lee, 2012). Lee (2012) suggests that market participants may take time to digest the "hard-to-understand" information in earning reports as an explanation for the delayed impact of earnings on share prices. This could be plausible since the literature has found that managers strategically manipulate the readability of SEC filings to hide poor earnings performance (Lee, 2012). Thus, E12 is effective to forecast the negative direction of stock returns across multi-step horizons.

## Most successful upward and long-horizon predictors
Further, we have found that macroeconomic variables are particularly successful in upward movement and long-horizon forecast. Mainly, they are from money and credit, and industrial production categories. This is consistent with Hasan and Javed (2009), that money supply has both significant short-run and long-run effects on share prices. 

As for output, our finding is inconsistent with some literature that claims a statistically insignificant relationship between equity prices and industrial output in the long run. However, we here attempt to offer a possible explanation for the predictive power of industrial outputs. Chen et al. (1986) has found that oil price is an important economic risk factor for the US stock market. Valadkhani & Smyth (2017) suggest that rises in oil prices negatively affect industrial output in the US. As the oil price is correlated with industrial output, industrial output variables may act as a proxy for oil prices to predict equity return direction, and thus they have the predictive power. 

## Performance of technical indicators
Neely, Rapach, Tu & Zhou (2014) have shown that technical indicators are statistically significant for monthly equity risk premium predictions. This initially motivated us to incorporate them into our forecasting project. However, we have discovered that technical indicators have poor forecasting power in general across all our 4 horizons. It could be because Neely et al. (2014) estimated the statistical significance of technical indicators during the OOS periods between Y1966M1 and Y2011M12. In contrast, we estimate for the OOS periods between Y2005M1 and Y2015M12. While it is unclear whether technical indicators were more applicable in the past in the literature, we conjecture that as more public information about companies and macroeconomic data become freely available online to aid investors' decisions, the relative importance of these publicly available information may override technical indicators by providing more direct signals to the health of stocks.

Additionally, moving average (MA) indicators are that when shorter-term MA crosses above the longer-term MA it indicates a buy signal because the trending is shifting up. Momentum indicators measure the rate of a rise or fall in share prices. By looking into the nature of technical indicators, they appear to be quite arbitrary in predictions and hence have poor predictive power.
The only exception is that on-balance indicators are selected as the most important variable in our penalized logit model in 1-month, 3-month and 6-month ahead forecasts. On-balance indicators measure the positive or negative flow of trading volume to signal buying and selling pressures of a stock. The findings from Barberis, Greenwood, Jin & Shleifer (2018) may provide an account for this. They claim that price increases are often accompanied by high trading volume, which could signal a bubble as well. Thus, on-balance indicators were selected to be the most important variable by the penalized logit model.

## Usefulness of non-linear variables
To answer our 2nd research question, we have found that non-linear variables perform well compared to linear variables. Enke & Thawornwong (2005) has summarized the wide acceptance of the existence of non-linearity in the financial markets and confirmed the past results through ANN methods. Our findings are consistent with the literature.

## ML techniques comparison
We have compared our models with unconditional mean and found that there is only one model that can marginally beat it from each horizon, except for the 12-step ahead horizon. The unconditional mean forecast delivers an accuracy of 61.4%. Meanwhile, all models applied by Iworiso & Vrontos (2019) generally produce at best an accuracy at the lower bound of 60%. To some extent, these findings are consistent with Goyal & Welch's (2007) pessimism about stock premium forecasting. While it is surprising that unconditional mean generally beat the ML methods, the finding that none of our models can beat unconditional mean in the 12-step ahead forecasts may imply a reason for that. That is, S&P 500 index prices generally have an upward trend in the long horizon, and so a prediction of upward direction in future periods by unconditional mean matches well with the general trend in S&P 500 index prices.

Furthermore, we have confirmed Iworiso & Vrontos (2019) and Ballings, Poel, Hespeels & Gryp (2015)'s results that tree-based methods yield good predictive performance compared to traditional models. In particular, random forest performs better than bagging for all horizons. An explanation to this could be that trees are highly correlated and thus de-correlating could help improve forecasting performance.

Lastly, we have shown that LASSO logit with theoretical plug-ins has a good and stable performance between horizons. Campbell & Thompson's (2008) suggestion of restricting predictors may justify LASSO logit's good performance.


<!-- Future Research -->
# Future Research
## A new benchmark to beat other than traditional econometric model
It is noticeable that the unconditional mean for forecasting performs better than most of our models and certainly our benchmark model – probit model. This suggests that future research should focus on beating unconditional mean forecasting in order to justify whether a model is good at prediction by comparing it to an arbitrary forecast like the unconditional mean. 

## Suggestion of predictors
Additionally, as discussed, E12 has shown dominant importance in predicting downward movement. By connecting E12's superior performance to the behavioral literature, we have argued that predictors that are linked with loss-aversion can improve forecasting negative returns. Hence, we recommend future research to incorporate more predictors of this kind to address the underperformance of predicting negative directional change.

## Maintaining the good forecasting elements
Also, since macroeconomic variables are shown to be useful, especially for long horizons, future research should consider including them in addition to Professor Goyal's set of financial indicators. In particular, industrial output variables play an important role. We could consider including energy price variables since they affect output and could have a high predictive power of the level of industrial outputs.


<!-- Limitation -->
# Limitations
In our project, we use default and theoretical plug-ins in bagging, boosting, and random forest to reduce computational complexity, instead of recursive cross-validation. Since we do not maximize their potential by using recursive cross-validation, some of our findings that basic tree performs better than bagging, boosting and random forest may be invalid.

Further, we only rely on the confusion matrix to evaluate the relative performance of forecast models, rather than through formal testing such as DM test, due to its applicability to only continuous prediction. We could not formally conclude equal or superior predictive power of our models. Those models that appear to marginally beat unconditional mean may not be better in a different sample.


<!-- Conclusion -->
# Conclusion
In conclusion, we have implemented a variety of machine learning techniques and traditional econometric models with a focus on tree-based methods to forecast S&P500 directional change. Through the comparison of forecasting performance, we have reaffirmed the findings of the good performance by tree-based methods and penalization technique. Also, by evaluating the forecasts, we have discovered and further discussed several findings with strong implications for future research. Firstly, the inclusion of non-linear variables and macroeconomic variables can boost the forecasting performance, while technical indicators are generally of little use. Secondly, learning from the poor performance of predicting negative directional change, future research should incorporate more predictors that are similar to E12 or are linked with investors’ loss aversion to improve forecasting performance. Thirdly, the surprising result that unconditional mean forecasts outperform many of our machine learning models suggests future research should treat it as a major benchmark to beat. 

Overall, be it our results or other researchers’ results, the accuracy is stagnant at the lower bound of 60%. While economic and financial forecasting may seem to be a dismal pursuit, we should still persist. As G. E. P Box puts it, ‘all models are wrong, but some are useful’. We hope our research has contributed to the development of some of the usefulness.

<!-- Appendix -->
# Appendix

![Image 1 of Appendix](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/9f43a124ef54b8b897360ae8679b2a18f739eff4/Appendix/Appendix%20Table%201%20(part%201).png)


![Image 2 of Appendix](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/9f43a124ef54b8b897360ae8679b2a18f739eff4/Appendix/Appendix%20Table%201%20(part%202).png)


![Image 3 of Appendix](https://github.com/hunglung2960/A-SUPERVISED-LEARNING-APPROACH-TO-PREDICTING-INDEX-DIRECTION/blob/9f43a124ef54b8b897360ae8679b2a18f739eff4/Appendix/Appendix%20Table%201%20(part%203).png)


<!-- Reference -->
# Reference

Anatolyev, S., & Gospodinov, N. (2010). Modeling financial return dynamics via decomposition. 	Journal of Business and Economic Statistics, 28(2), 232–245.

Ballings, M., Van den Poel, D., Hespeels, N., & Gryp, R. (2015). Evaluating multiple classifiers for stock price direction prediction. EXPERT SYSTEMS WITH APPLICATIONS, 42(20), 7046–7056. doi: 10.1016/j.eswa.2015.05.013

Barberis, N., Greenwood, R., Jin, L. & Shleifer, A. (2018). Extrapolation and bubbles. Journal of 	Financial Economics, 129(1), 203-227. doi: 10.1016/j.jfineco.2018.04.007

Campbell, J. Y. & Thompson, S. B. (2008). Predicting Excess Stock Returns Out of Sample: Can 	Anything Beat the Historical Average? The Review of Financial Studies,21(4), 1509–1531.	doi: 10.1093/rfs/hhm055

Chen, N. F., Roll, R. & Ross, S. (1986) Economic Forces and the Stock Market. Journal of Business, 59(1), 383–403.

Chevapatrakul, T. (2013). Return sign forecasts based on conditional risk: Evidence from the UK stock market index. Journal of Banking and Finance, 37(7), 2342–2353.

Enke, D. & Thawornwong, S. (2005). The use of data mining and neural networks for forecasting stock market returns. Expert Systems with Applications, 29(1), 927-940. doi: 10.1016/j.eswa.2005.06.024

Goyal, A. & Welch, I. (2007). A Comprehensive Look at The Empirical Performance of Equity 	Premium Prediction. The Review of Financial Studies, 21(4), 1455–1508. doi: 10.1093/rfs/hhm014

Haigh, M. S. & List, A. J. (2005). Do Professional Traders Exhibit Myopic Loss Aversion? An Experimental Analysis. Journal of Finance, 60(1), 523-534. doi: 10.1111/j.1540-6261.2005.00737.x

Harvey, D. I., Leybourne, S. J., Sollis, R. & Taylor, A. M. R. (2020). Real‐time detection of regimes of predictability in the US equity premium. Journal of Applied Econometrics, 36(1), 45-70. 	doi: 10.1002/jae.2794

Hasan, A., & Javed, M. T. (2009). Macroeconomic influences and equity market returns: A study of an emerging equity market. Journal of Economics and Economic Education Research, 10(2), 47.

Iworiso, J. & Vrontos, S. (2019). On the directional predictability of equity premium using machine learning techniques. Journal of Forecasting, 39(1), 449-469. doi: 10.1002/for.2632

Kinney, W., Burgstahler, D. & Martin, R. (2002). Earnings Surprise “Materiality” as Measured by Stock Returns. Journal of Accounting Research, 40(5), 1297-1329. doi: 10.1111/1475-679X.t01-1-00055

Lee, Y. J. (2012). The Effect of Quarterly Report Readability on Information Efficiency of Stock 	Prices. Contemporary Accounting Research, 29(4), 1137-1170. doi:10.1111/j.1911-3846.2011.01152.x

Leitch, G., & Tanner, J. (1991). Economic forecast evaluation: Profits versus the conventional error measures. American Economic Review, 81(3), 580–590.

Leung, M. T., Daouk, H., & Chen, A.-S. (2000). Forecasting stock indices: A comparison of classification and level estimation models. International Journal of Forecasting, 16(2), 173–190.

Neely, C. J., Rapach, D. E., Tu, J. & Zhou, G. (2014). Forecasting the Equity Risk Premium: The 	Role of Technical Indicators. Management Science, 60(7):1772-1791. doi: 10.1287/mnsc.2013.1838

Nyberg, H. (2011). Forecasting the direction of the US stock market with dynamic binary probit models. International Journal of Forecasting, 27(2), 561–578. doi: 10.1016/j.ijforecast.2010.02.008

Pesaran, M. H., & Timmermann, A. (1995). Predictability of stock returns: Robustness and economic significance. Journal of Finance, 50(4), 1201–1228.

Skinner, D.J. & Sloan, R.G. (2002). Earnings Surprises, Growth Expectations, and Stock Returns or Don't Let an Earnings Torpedo Sink Your Portfolio. Review of Accounting Studies, 7(1), 289–312. doi: 10.1023/A:1020294523516

Valadkhani, A. & Smyth, R. (2017). How do daily changes in oil prices affect US monthly industrial output? Energy Economic, 67(1), 83-90. doi: 10.1016/j.eneco.2017.08.009





<!-- Remarks -->
# Remarks
Group members: O Hung Lung, Nathanael Lam Zhao Dian, Heng Soon Chien

