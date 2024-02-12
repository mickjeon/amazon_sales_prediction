# amazon_sales_prediction

## Members
Juhak Lee and Jae Min Jeon

## Abstract
By analyzing the Amazon product dataset, we want to investigate the relationship between discounted prices and regular prices with product success, or the rating received from consumers. Then, we want to build models that predict the optimal discount rate and keyword in product name given a product category and its price. In order to achieve this goal, we plan to explore both supervised and unsupervised machine learning methods. We plan to use supervised learning to predict the optimal discount rate as the dataset includes numerical value for product success. In addition we plan to use unsupervised learning to learn the optimal keyword in a product name. Through these approaches, we hope to establish a fundamental understanding of machine learning methods and learn each methods’ strengths and weaknesses through practice.

## Data Preprocessing
In order to obtain meaningful data for our model, we preprocessed our data. Our data is split up into subcategories and we choose to display 3 of them in the preprocessing.ipynb (Although our preprocessing script can be generalized to all categories). 

The first task that we did is to drop null values. After viewing some samples of the data, we found null values in ratings, no_of_ratings, actual_price, and discount_price. We then found out that some instances of ratings and no_of_ratings do not contain float values in string (all the columns were in string format upon import). We handled those edge cases and dropped the instances as well. 

We simultaeneously converted all ratings, no_of_ratings, actual_price, and discount_price from string to float. As for actual_price and discount_price, we converted them to usd, as they were in rupees upon import. We used the exchange rate as of 02/08/24.

We then created to new columns. We calculated the discount_rate of each instance using actual_price and discount_price. We also calculated the aggregate rating, agg_rating, by multiplying ratings and no_of_ratings columns. 

Last but not least, we normalized the data using min-max normalization and standardized the data by using z-scoring.