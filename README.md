### Abstract  

Market basket analysis is a widely used data mining technique to identify patterns of co-occurrence among products that customers frequently purchase together. The Apriori algorithm is a popular method for performing market basket analysis, which works by iteratively generating frequent itemsets and association rules. In this project, we have done market basket analysis with the Apriori algorithm, including its key concepts, methodologies, and practical applications. We discuss how the algorithm can be used to identify relevant product combinations and optimize product placement and promotion strategies. Additionally, we highlight the challenges and limitations of Apriori algorithm and offer insights into its future directions. Overall, this paper explains the purpose behind the market basket analysis and the Apriori algorithm as a valuable tool for improving business decision-making in the retail industry.

**Keywords** —Machine learning, apriori algorithm, market basket analysis, data mining, pyspark, sqlcontext, pandas, seaborn

## I.	INTRODUCTION

In this digital era, shopping on internet has grown largely popular among people, and it's delicate to find someone who has not been exposed to it. Let’s look at an introductory situation in one’s original grocery store to help with understanding what this paper explains about. For instance, if a store has increased chuck sales then it can further upsell it by lowering the price of adulation and jam, as a customer who buys chuck is more likely to buy jam and adulation, all together. So, when it comes to gaining client perceptivity, market basket analysis in data mining remains a crucial factor. As a result, market basket analysis in data mining is a fashion for relating useful and important styles of constantly bought products in a store’s sale history [1].
Market basket analysis (MBA) is a data mining technique that allows one to relate to purchase patterns in any retail terrain. MBA is a set of statistical affinity computations that punctuate coping patterns to help business leaders understand – and eventually serve – their guests more. MBA, in its utmost introductory form, quests for the most common product combinations in deals [2]. To state it simply, MBA is a data mining technique that allows a store proprietor to dissect and determine product combinations, which particulars are related, and which particulars guests constantly buy together. It’s a lovely strategy grounded on the introductory principle that if we buy commodity, we’re obliged to buy or avoid commodity differently (or a bunch of effects).

### II.	DATASET

The dataset which we used here consists of 7 attributes helpful which help to analyze customer buying patterns. The dataset contains 522065 rows. The 7 features are as follows:
* BillNo: 6-digit number assigned to each transaction. Nominal.
* Item name: Product name. Nominal. 
* Quantity: The quantities of each product per transaction. Numeric.
* Date: The day and time when each transaction was generated. Numeric.
* Price: Product price. Numeric.
* CustomerID: 5-digit number assigned to each customer. Nominal.
* Country: Name of the country where each customer resides. Nominal.

### III.	AGILE METHODOLOGY

1.	Finding and loading the data:

* This topic was new to us. So, we did research about the data online and at last, we found the dataset which we needed for our project. 
* As we have used pyspark in this project. Initially, we tried to do some research about this project and then we implemented the project.
* At first, we imported all the necessary libraries in our notebook and then we mentioned the SPARK_LOCAL_HOSTNAME=localhost to set an environment variable SPARK_LOCAL_HOSTNAME to localhost, which specifies the hostname or IP address that the Spark driver should bind to.
* At last, we loaded the dataset into a data frame.

2.	Data preprocessing and cleaning –

* After loading the data, the initial steps that we need to perform is data cleaning and preprocessing. 
* At first, we tried to check for the null values in our dataset and we did not find any null values in it.
* After that, in a column Quantity, we checked for 0 values, as we didn’t need the Quantity to be Zero so we removed those values from our dataset.
* Next, we tried to do Total_Price by multiplying Quantity with the Price so that we can get the total price of all the items. So that’s why we created one new column with these Total_Price values.

3.	Outlier Detection – 

* Outlier Detection we have used IQR method. After that we found around 42211 outliers in our dataset. 
* For IQR, we initially mentioned Upper Quantile and Lower Quantile by 0.75 and 0.25 As Q3 and Q1.
* Then, we calculated the IQR by this formula IQR = Q3 - Q1. We also found the upper bound and lower bound to check for the outliers.
* By using this method. We found around 42211 outliers in our dataset.
* Later, to deal with the outliers, we used the flooring and capping method. 
* We have used the same 75% and 25% values as capping and flooring values. 
* We found Lower bound of total_price 16.49 and upper bound 36.3. 
* So, at last, we have removed the values which were below to lower bound with the lower bound values and the values which are above the upper bound with the upper bound and removed the outliers.
* To get the better idea of outliers, we have plotted the box plot.

### Flooring and capping 

Flooring and capping are a method used for outlier removal in data analysis. Outliers are data points that are significantly different from the rest of the data and can have a big impact on statistical analysis. Flooring and capping are a technique used to deal with outliers by setting a minimum and maximum threshold value for the data. Any data point that falls below the minimum value is replaced with the minimum value, and any data point that is above the maximum value is replaced with the maximum value.

Flooring involves setting a minimum threshold value for the data. Any data point that falls below this threshold is replaced with the minimum value. Capping, on the other hand, involves setting a maximum threshold value for the data. Any data point that falls above this threshold is replaced with the maximum value.

The choice of the threshold values depends on the nature of the data and the analysis being performed. If the data is known to have extreme values, such as in financial data, the threshold values may be set wider. However, if the data is known to be relatively consistent, such as in experimental data, the threshold values may be set narrower.

It is important to note that flooring and capping can have an impact on the analysis results, as it alters the original data. Therefore, it is important to consider the potential impact on the analysis before using this technique. Additionally, there may be other outlier removal techniques that are more appropriate for the specific dataset and analysis being performed.


4.	Visualization – 

* After cleaning our data, we tried to do some visualization to understand the data better.
* For visualization we have used plotly and seaborn libraries. 

### Plotly –

* It is an open-source data visualization library for creating interactive, web-based visualizations in Python, R, and JavaScript. 
* It allows users to create a wide range of visualization types, including line charts, scatter plots, bar charts, heatmaps, and more. 
* Plotly supports various output formats, including standalone HTML files, Jupyter notebooks, and web applications.
* One of the key features of Plotly is its ability to create interactive visualizations that allow users to explore and analyze data. Users can zoom in and out of charts, hover over data points to display additional information, and click on elements to trigger actions such as filtering or highlighting data subsets.
* Plotly provides a high degree of customization, allowing users to control the appearance of their visualizations, including colors, labels, and fonts. 
* It also provides a range of built-in tools for creating more complex visualizations, such as subplots, animations, and 3D charts.
* Find below the visualization graph of country wise Quantity’s Total_Price values.

![image](https://github.com/harshilbhavsar7/Market-Basket-Analysis-with-Apriori/assets/60917314/264ed11d-775e-445e-987e-2e015672b3e2)
 
Fig 1 Country vs Total_price of Quantity

* Also, to analyze further, we have created a below graph of the quantity of items sold (bars) and the total gain for each item. We can see that the items which give the most gain are not always the ones which are sold in large quantity.

 
![image](https://github.com/harshilbhavsar7/Market-Basket-Analysis-with-Apriori/assets/60917314/558acb07-c740-4150-a1b4-076cafee3030)
Fig 2 Quantity of items sold vs total gain of each item

* At last, we have plotted the bell graph of Total_price to check the skweness of our dataset. 

### Skewness –

* Skewness is a statistical measure that describes the degree of asymmetry in a distribution. If a distribution is perfectly symmetrical, its skewness is zero. A positive skewness indicates that the distribution has a longer tail on the right-hand side, while a negative skewness indicates that the distribution has a longer tail on the left-hand side.
* To calculate the skewness of a dataset, we can use the following formula:

**Skewness = (3 * (mean - median)) / standard deviation**

* If the skewness value is greater than zero, the distribution is positively skewed, while if it is less than zero, it is negatively skewed.
* It is essential to note that skewness can impact the interpretation of statistical analyses. For example, if a dataset is positively skewed, the mean will be greater than the median, which may affect the interpretation of the central tendency of the data. Therefore, it is crucial to consider.
* So, by analyzing the above graph, we note that the distribution of the expenses per customer is symmetrical, with two peeks, in a log scale.


![image](https://github.com/harshilbhavsar7/Market-Basket-Analysis-with-Apriori/assets/60917314/4877f86e-2b13-491a-91ab-41cccabf4424)
Fig 3 Distribution of total_price data


5.	Model Building – 

* We have used Apriori algorithm in our project to get the list of items which customers are buying together in the various countries.
* As our dataset was big, to start with, we used 9 countries to implement our algorithm. Those countries are 'Belgium', 'Germany', 'Italy', 'Netherlands', 'Portugal', 'Norway', 'Spain', 'Sweden', 'Australia'.
* For that we have used minSupport and minConfidence parameters to 0.1 and 0.8, respectively, which means that the algorithm will find frequent item sets that appear in at least 10% of the transactions and generate association rules that have a confidence of at least 80%.

### Apriori Algorithm-

The Apriori algorithm is a data mining technique used to find frequent itemsets and association rules from large datasets. It works by generating a set of candidate itemsets and then using these itemsets to iteratively explore the space of larger itemsets until no more frequent itemsets can be found.

Here is a step-by-step explanation of the Apriori algorithm using an example:

Suppose we have a transaction database containing the following transactions:
```
Transaction 1: bread, milk
Transaction 2: bread, butter, beer, eggs
Transaction 3: milk, butter, beer, coke
Transaction 4: bread, milk, butter, beer
Transaction 5: bread, milk, butter, coke
```
Step 1: Set a minimum support threshold.

Before we can apply the Apriori algorithm, we need to set a minimum support threshold, which is the minimum number of transactions an itemset must appear in to be considered ‘frequent’. For this example, let's set the minimum support threshold to 3.

Step 2: Find frequent 1-itemsets.

The first step of the Apriori algorithm is to find all frequent 1-itemsets, which are single items that appear in at least the minimum number of transactions. To do this, we count the occurrence of each item in the database and check if it meets the minimum support threshold.

In our example, we have the following counts:
```
bread: 4
milk: 4
butter: 4
beer: 3
coke: 2
eggs: 1
```
Since all these items meet the minimum support threshold of 3, they are all considered frequent 1-itemsets.

Step 3: Generate candidate 2-itemsets

The next step of the Apriori algorithm is to generate candidate 2-itemsets, which are pairs of frequent items that appear in at least the minimum number of transactions. To generate these candidate itemsets, we take all pairs of frequent 1-itemsets and check if they satisfy the ‘Apriori property’, which states that any subset of a frequent itemset must also be frequent.

In our example, we have the following pairs of frequent 1-itemsets:
```
{bread, milk}
{bread, butter}
{bread, beer}
{milk, butter }
{milk, beer}
{butter, beer}
{bread, coke}
{milk, coke}
{butter, coke}
```
We can see that {bread, eggs} is not a candidate 2-itemset because eggs are not a frequent 1-itemset.

Step 4: Count the support of candidate 2-itemsets.

Next, we count the occurrence of each candidate 2-itemset in the database and check if it meets the minimum support threshold.

In our example, we have the following counts:

{bread, milk}: 3
{bread, butter}: 3
{bread, beer}: 2
{milk, butter}: 3
{milk, beer}: 2
{butter, beer}: 3
{bread, coke}: 1
{milk, coke}: 1
{butter, coke}: 1

Since {bread, coke}, {milk, coke}, and {butter, coke} do not meet the minimum support threshold of 3, they are not considered frequent 2-itemsets.

Step 5: Generate candidate 3-itemsets.

The next step of the Apriori algorithm is to generate candidate 3-itemsets, which are triples of frequent items that appear in at least the minimum number of transactions. To generate these candidate itemsets, we take all triples of frequent 2-itemsets and check if they satisfy the Apriori algorithm. 

Finally, we have applied the same algorithm to the whole dataset, considering a min support of 0.02 and a min confidence of 0.4.

6.	Model Evaluation – 

* After implementing the algorithm, we tried to evaluate it and found that, Ordering the results by lift, we can see that there are rules with very high lift, but low support. Due to the difference between different countries, a local analysis should be preferred over a global one.
* We also can observe that, when confidence, support, and lift are high for a rule, it means that the rule has a high probability of being true and is considered to be a strong association rule.
* For example, a high confidence value indicates that the consequent of the rule is frequently bought when the antecedent is bought, and a high lift value indicates that the likelihood of the consequent being bought increases when the antecedent is bought compared to the overall probability of buying the consequent.
* Therefore, if confidence, support, and lift are high for a rule, it suggests that the antecedent and consequent have a strong relationship and are likely to occur together, which can be useful for businesses in product placement and promotion strategies.
* By implementing the project, we have identified the following things –

* Total gain by country: We calculated the total revenue earned by each country based on the sales made. This information can be useful for businesses to identify their most profitable markets and allocate resources accordingly.
* Association rules: We used the FP-growth algorithm to identify the association rules between different products purchased by customers. This can help businesses in cross-selling and upselling their products and services to customers.
* Lift value: We calculated the lift value for the association rules, which helps in identifying the strength of the relationship between different products. This information can be useful in identifying which products are frequently purchased together and can be promoted as a bundle or in a package deal.
* Confidence and support values: We also calculated the confidence and support values for the association rules. These values help in identifying the reliability and frequency of the association rules, respectively.

### Conclusion

Overall, the project provides useful insights into the sales data and can be used by businesses to optimize their sales strategies and increase their revenue. We identified the top-selling products in each country based on the quantity sold. This information can be useful for retailers to make informed decisions on which products to stock up on and promote in each country.


### References

1.	Apriori Algorithm in Data Mining: Implementation With Examples. (2023, February 17). Software Testing Help. https://www.softwaretestinghelp.com/apriori-algorithm/
2.	Ganiyu, I. S. (2023, January 18). Market Basket Analysis in Data Mining Simplified 101 - Learn | Hevo. Learn | Hevo. https://hevodata.com/learn/market-basket-analysis-in-data-mining/ 
3.	Chen, J. (2019). Learn About Skewness. Investopedia.  https://www.investopedia.com/terms/s/skewness.asp               
4.	Apriori Algorithm in Data Mining: Implementation with Examples. (2019, November 10). Softwaretestinghelp.com.https://www.softwaretestinghelp.com/apriori-algorithm/
5.	Market_Basket_Analysis_Pyspark_Apriori. (n.d.). Kaggle.com. Retrieved April 14, 2023, from https://www.kaggle.com/code/gabrieleottaviani/market-basket-analysis-pyspark-apriori/input 
