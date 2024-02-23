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


## First Model: Polynomial Regression

Based on the observation that the correlation matrix did not reveal significant linear relationships, we chose polynomial regression. 
To enhance the flexibility of our model, we converted the sub_category into integer values representing categories. We merged the rating datasets for Amazon Fashion, Televisions, and Luxury Beauty into one dataset, mapping the categories as follows:
Amazon Fashion: 0
Televisions: 1
Luxury Beauty: 2

The polynomial regression model yielded the lowest testing Mean Squared Error (MSE) at degree 3. 
At degree 3, the testing MSE was 0.019547535944113525, and the training MSE was 0.0117217271095025.

Even though polynomial regression yielded satisfactory results, it is likely due to the clustering of ratings between 3 and 4. the next two models we are considering are:

Neural Network: A neural network would be a good choice for modeling the relationship between rating and the features such as no_of_ratings, discount_price_usd, actual_price_usd, discount_rate, and agg_rating, as there may be complex nonlinear relationships. This is because neural networks is good at capturing complex nonlinear patterns, and they offer the flexibility to fine-tune hyperparameters, including various loss functions and activation functions, to adapt to the characteristics of the dataset.

SVM: SVM can also handle nonlinear relationships between ratings and the product information. By categorizing ratings into discrete classes (e.g., 1, 2, 3, 4, 5), we can expect that SVM can effectively classify ratings based on the given features of the product.

### Conclusion and Improvement

The optimal complexity (degree) of the polynomial regression model was determined to be degree 3. While the model's performance(MSE) was good, it may have been influenced by the clustering of ratings around 3 and 4. To address this issue and potentially improve the polynomial model, we can consider the following:

Oversampling the classes with fewer ratings: replicating samples from classes with fewer ratings can help balance the dataset.

Cross-validation (e.g., k-fold): Validate the performance of the polynomial model using techniques like k-fold cross-validation to ensure that degree 3 is indeed the optimal choice. This can provide additional confidence in the chosen complexity of the model and help assess its generalization ability.

