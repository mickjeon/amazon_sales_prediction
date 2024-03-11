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

## Second Model: Aritifical Neural Network
To predict product ratings, we developed an Aritficial Neural Network as the second model. We saw an improvement from the polynomial regression model.

Given that most ratings fall within the 3-4 range, the key to building a good model is in filtering out ratings of 1 and 5. Although our first model, polynomial regression model, performed reasonably well within the 3-4 rating range, it was significantly bad in predicting ratings of 1 and 5. Similarly, our second model, when given normalized data, also struggled with predicting anomalous ratings of 1 and 5.

### Evaluating our data, labels, and loss function
While we found mean squared error as the appropriate loss function for the regression task, we suspected that normalizing the ratings using min-max normalization so that we map data between 0 and 1 can be a problem. Our loss function, MSE, reacts less sensitively to large errors when values are between 0 and 1 compared to the original 1-5 scale.

Therefore, we decided to train our ANN using the original rating scale of 1-5 before normalization. To compare performance, we re-evaluated our first model, polynomial regression model, using the original rating scale. When using the original 1-5 rating scale with the neural network model, we observed a significant improvement in predicting ratings of 1 and 5.

### The baseline ANN Model
After trying out a couple different configuration, we decided that our baseline ANN is a network that uses 4 hidden layers with 32 nodes in each of the layers. We chose relu as the activation function except for the last layer, which had linear activation. <b>This model achieved test MSE of 0.137 and training MSE of 0.097</b>.

### Model's place in a fitting graph
This particular model shows mild signs of overfitting. When comparing the train loss and validation loss, the difference is small except for a few epochs in training. They also reveal a downward trend as epochs increase. When comparing the train mean absolute error and validation mean absolute error, validation error is consistently a little bit higher than the train error. Mean absolute error still reveals a downward trend. Therefore, we believe that our baseline ANN reveals mild signs of overfitting.

### K-Fold Cross Validation
The test MSE for the polynomial model using the 1-5 scale was 0.317, and for the neural network model with the 1-5 scale, it was 0.137.
To validate the model's performance, we conducted repeated k-fold cross-validation and achieved 0.216 as the average validation MSE. We believed that we could do better with an ANN and therefore, performed hyper-parameter tuning.

### Hyperparameter Tuning
Furthermore, to optimize the neural network model's performance, we performed hyperparameter tuning, exploring variations such as:

- Activation function: ["relu", "tanh", "linear"]
- Loss function: ["mse", "mae"]
- Number of nodes in each layer: 16-64, step_size=8

The best model after hyperparameter tuning achieved validation mean square error of 0.03225 and test accuracy of TODO with the following model setup:

- Loss function: mse
- Activation function: used
- Number of nodes in each layer: 32
- Number of layers: TODO


### Conclusion and Improvement
Overall, we were able to achieve a much lower mean squared error by building an artificial neural network regressor to predict ratings, compared to our polynomial regression model. The baseline ANN achieved Test MSE of 0.137, while our first model had Test MSE of 0.316. When comparing the predicted values with the true values, we noticed the ANN had significant improvements at predicting ratings of 1s and 5s. To validate it's capabilities further, we performed repeated Kfold cross validation. We then performed hyperparameter tuning on the activation function, number of nodes in the hidden layer, and loss functions. We were able to achieve a better test loss of TODO.

We believe that there are other ways to improve upon our Artificial Neural Network. We suppose one way is to experiment with different optimizers for gradient descent. Another way is to experiment with the batch size for further hyperparameter tuning.

### Next Model
We plan to build a decision tree regressor as our third model. We have decided not to scale our data for model training in this milestone. We believe that Decision Tree Regressor is beneficial as it does not require feature rescaling to train. Furthermore, we hope that decision tree model will be able to handle the different price range of products well. For example, within the TV category, there are big ranges of TV prices. We believe that the decision tree regressor will be able to separate the different price ranges and predict a price range specific ratings prediction. The same logic applies to the number_of_ratings feature as well.