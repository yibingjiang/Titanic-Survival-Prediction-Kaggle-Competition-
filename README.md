# Titanic Survival Prediction (Kaggle Competition)

## Introduction

The sinking of the Titanic is among the most notorious maritime disasters in history. On its maiden voyage in April 1912, the RMS Titanic collided with an iceberg, leading to a catastrophic sinking that resulted in the loss of over 1,500 lives due to insufficient lifeboats. This tragedy has been studied extensively not only for its historical impact but also for the insights it provides into survival dynamics under disaster conditions. In this project, we will use machine learning to predict survival outcomes based on passenger data, such as age, sex, and socio-economic class, among other factors.

## Data Description

The dataset utilized in this analysis consists of passenger information from the RMS Titanic, which includes several key variables that potentially influence survival rates. Below is a table describing each variable in detail:

| Variable   | Description |
|------------|-------------|
| `survived` | Survival outcome (0 = No, 1 = Yes) |
| `pclass`   | Ticket class as a proxy for socio-economic status (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `sex`      | Gender of the passenger |
| `age`      | Age of the passenger in years |
| `sib_sp`   | Number of siblings or spouses aboard |
| `parch`    | Number of parents or children aboard |
| `ticket`   | Ticket number |
| `fare`     | Fare paid for the ticket |
| `cabin`    | Cabin number |
| `embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Exploratory Data Analysis

### Survival Outcomes: 

The dataset shows a pronounced imbalance in survival outcomes, with approximately 61% of the passengers not surviving. Therefore, when splitting the dataset into training and testing dataset, we should use strata sampling of survival, so that the distribution of survival is same in both training and testing dataset.

![survived_dist](https://github.com/yibingjiang/Titanic-Survival-Prediction-Kaggle-Competition-/assets/141768352/f9ab353b-d3ef-4cdd-b5de-13962cedf606)

### Correlations Among Features: 

The correlation analysis highlights several notable relationships, such as the positive correlation between `sib_sp` and `parch`, indicating that passengers traveling with family were more likely to have either all members survive or perish together. The correlation plots provide a visual confirmation of these relationships, aiding in understanding the dynamics within the data that could influence the modeling approach.

![correlation](https://github.com/yibingjiang/Titanic-Survival-Prediction-Kaggle-Competition-/assets/141768352/06c1d22b-f1d3-4c17-a95f-6dd28d190529)

### Gender and Survival: 

One of the most significant findings from the EDA is the impact of gender on survival rates. The data reveals a marked disparity in survival chances between genders, with females having a significantly higher survival rate. This pattern is vividly illustrated in the bar plots, where the proportion of surviving females substantially exceeds that of males, likely reflecting historical accounts of women and children being prioritized during lifeboat evacuations.

![gender_survival](https://github.com/yibingjiang/Titanic-Survival-Prediction-Kaggle-Competition-/assets/141768352/35a3dd54-d143-426f-80d1-e7a93c2734c6)

### Passenger Class and Survival: 

The passenger class, indicative of socio-economic status, also shows a strong correlation with survival rates. First and Second Class passengers had markedly higher survival rates compared to those in Third Class. This observation is graphically represented through bar plots that clearly depict higher survival rates among the upper classes, suggesting that socio-economic status afforded advantages, possibly in terms of cabin location and access to lifeboats.

![pclass_survival](https://github.com/yibingjiang/Titanic-Survival-Prediction-Kaggle-Competition-/assets/141768352/d0cb4d73-0956-4adf-8c77-ef1de67be39c)

## Model Building

The model building process in this analysis consisted of two primary stages: creating a recipe for preprocessing the data and constructing various predictive models to estimate survival probabilities. This structured approach ensures that the data is appropriately prepared for modeling and that various algorithms are evaluated to find the most effective predictor of survival on the Titanic.

### Creating the Recipe

The preprocessing recipe was crafted using the tidymodels framework, a comprehensive suite of R packages designed for machine learning and statistical modeling. The recipe specifies the transformations and preprocessing steps applied to the features before modeling:

**Imputation**: Linear imputation was used for missing values in the age and fare variables, leveraging all available predictors to estimate the missing values. This approach helps mitigate the impact of missing data on model performance.

**Dummy Variables**: Categorical variables such as pclass and sex were converted into dummy variables to facilitate their use in the various machine learning algorithms that require numerical input.

**Interaction Terms**: Interaction terms were created between age and fare, and sex and fare to explore potential synergistic effects between these variables on the likelihood of survival.

**Scaling and Centering**: All predictors were scaled and centered to have mean zero and unit variance, standardizing the data to improve the numerical stability of the models and the interpretability of the results.

The recipe was then prepared using the training data to finalize the transformations and ensure the data was ready for modeling.

### Model Construction

Several machine learning models were constructed and evaluated for their ability to predict survival:

**Logistic Regression**: This model provides a probabilistic framework for binary classification, making it a natural choice for predicting the binary outcome of survival.

**Linear and Quadratic Discriminant Analysis (LDA and QDA)**: These models were used to assess the data's linear and quadratic boundaries, respectively.

**K-Nearest Neighbors (KNN)**: A non-parametric method that classifies based on the majority vote of the nearest neighbors, providing a flexible and intuitive approach.

**Random Forest**: An ensemble method using multiple decision trees to improve predictive accuracy and control over-fitting, which proved to be highly effective given its ability to handle various data types and complex interaction structures.

**Naive Bayes**: This probabilistic classifier assumes independence between predictors, a useful baseline for comparison due to its simplicity and efficiency.

**Neural Networks**: A deep learning approach was tested to capture non-linear relationships through multiple layers of processing.

## Model Evaluation

### Training Accuracy

![model_train_acc](https://github.com/yibingjiang/Titanic-Survival-Prediction-Kaggle-Competition-/assets/141768352/b71c77e1-e66b-4c9a-a8a4-471b32c33b59)

**Random Forest** demonstrated the highest accuracy, underscoring its robustness and ability to handle complex data relationships and interactions effectively.

**Neural Networks** and **Logistic Regression** also showed strong performance, indicating their capacity to model non-linear relationships and logistic boundaries, respectively.

**K-Nearest Neighbors (KNN)**, **Linear Discriminant Analysis (LDA)**, and **Quadratic Discriminant Analysis (QDA)** offered competitive but slightly lower accuracy levels, reflecting their varying sensitivity to the dataset’s features and structure.

**Naive Bayes** had the lowest accuracy, which might be due to its assumption of independence among predictors—a condition not met in this dataset.

### Testing Accuracy

![test_acc](https://github.com/yibingjiang/Titanic-Survival-Prediction-Kaggle-Competition-/assets/141768352/03d2a514-1226-4d77-8068-dedc09063476)

To ensure that the models not only fit the training data well but also perform well on unseen data, they were evaluated using the separate testing dataset. The accuracies on the testing dataset were generally slightly lower than on the training dataset, which is expected due to the models being optimized for the training data. However, the Random Forest model again stood out, achieving the highest accuracy, followed closely by QDA and Neural Networks. This suggests that these models have better generalization capabilities, which is crucial for practical applications.

Here are some key points from the testing dataset evaluation:

**Random Forest** maintained high accuracy, reinforcing its effectiveness as observed in the training phase.

**QDA** and **Neural Networks** showed good generalization, indicating that these models were able to capture the underlying patterns in the data without overfitting.

## Discussion

This project's exploratory data analysis revealed that socio-economic status, denoted by passenger class, was a decisive factor in survival chances. Higher-class passengers, who typically had access to better resources and were likely located on higher decks closer to lifeboats, showed significantly higher survival rates. This underscores the role of social hierarchies in life-saving situations.

Gender emerged as another critical determinant, with females significantly more likely to survive than males, likely reflecting the historical 'women and children first' protocol observed during the ship's evacuation. Age also played a crucial role, with younger passengers and children having higher survival rates, possibly due to prioritization in rescue efforts. Additionally, the analysis indicated that passengers who paid higher fares, which could also correlate with higher passenger classes, had better survival probabilities. This may be attributed to the location of their cabins, which were possibly closer to the lifeboats, and perhaps also to a greater awareness or faster access to critical information and resources during the crisis.

## Conclusion

This project effectively leveraged a variety of machine learning techniques to predict survival on the Titanic, with the Random Forest model emerging as the most robust and accurate. This model excelled due to its capacity to manage the dataset's complexities, including its inherent feature interactions and class imbalances. The high performance of the Random Forest model, both in terms of training accuracy and its ability to generalize to unseen test data, highlights its suitability for complex predictive tasks where relationships between variables are non-linear and multi-faceted.

Moreover, the project goes beyond mere predictions to offer insights into the human aspects of the Titanic disaster, revealing how socio-economic and demographic factors influenced survival chances. These findings are not only academically intriguing but also provide meaningful lessons on the socio-dynamics during emergencies, which can be valuable for current and future disaster response and preparedness strategies.

## Recommendation

To enhance the predictive models further, several steps could be taken:

**Advanced Model Tuning**: There is scope for further refinement of the models, particularly Neural Networks and KNN, which could benefit from more sophisticated parameter tuning to improve their performance.

**Complex Feature Engineering**: Developing more intricate features through advanced engineering techniques, such as polynomial features and more complex interactions, could help capture the subtleties in the data more effectively.

**Data Augmentation**: Incorporating additional data sources, such as detailed passenger records or historical weather conditions, could provide a richer dataset for analysis. Similarly, techniques like SMOTE could be employed to address class imbalances in the training dataset, potentially leading to more accurate predictions.

In conclusion, this comprehensive analysis not only highlights the predictive power of machine learning in historical datasets but also stresses the importance of understanding the human elements that influence outcomes in catastrophic events. The project serves as a testament to the potential of data science to uncover insights from the past, aiding both historical understanding and future preparedness.
