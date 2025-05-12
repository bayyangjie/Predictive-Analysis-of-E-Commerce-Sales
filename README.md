![aws logo](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/amazon.png)

# Table of Contents
1. [Introduction](#introduction)
2. [About the data](#about-the-data)
3. [Visualisations](#visualisations)
4. [Modelling and Evaluation](#modelling-and-evaluation)
    * [Predictive Modelling](#predictive-modelling)
        * [Encoding](#encoding)
        * [LASSO Modelling](#lasso-modelling)
        * [Predictive Model #1 - KNN](#predictive-model-1---knn)
        * [Predictive Model #2 - Decision Tree with Gradient Boosting](#predictive-model-2---decision-tree-with-gradient-boosting)
        * [Predictive Model #3 - Random Forest](#predictive-model-3---random-forest)
    * [Summary of Findings](#summary-of-findings)
    * [Evaluation of Predictive Models](#evaluation-of-predictive-models)
5. [Recommender System](#recommender-system)
    * [User Item Matrix](#user-item-matrix)
    * [NMF Model](#nmf-model)
    * [Simulation](#simulation)
6. [Conclusion](#conclusion)
7. [Improvements](#improvements)

# Introduction
This research looks into the fast-changing landscape of online shopping, focusing on Amazon to identify and analyze new consumer behaviors. The key goals are to enhance Amazon's marketing strategies and boost user satisfaction by using data analytics and machine learning techniques. The findings highlight crucial behavioral trends that significantly improve customer satisfaction and engagement, particularly through methods like K-Nearest Neighbors and Decision Tree with Gradient Boosting. These insights led to the development of a sophisticated recommendation system that effectively spots opportunities for cross-selling and up-selling. The methods and results discussed have important implications for integrating them into Amazon's operations, aiming to improve user experiences and strengthen the company's market position by better understanding and fulfilling customer needs.

# About the data
This research uses a dataset from Kaggle that includes the interactions of 602 customers on Amazon, covering 23 different variables. These interactions show how often customers browse and buy, their satisfaction levels, and how they respond to recommendations. This information helps to understand customer engagement better. The dataset is detailed enough to analyze user behaviors closely, which supports the predictive insights aimed for in this report. It is assumed that each entry in the dataset is unique, allowing for a strong individual-based analysis.

# Visualisations
## **Count of each Purchase Frequency Category** <br>

The bar plot shows that the overall purchase frequency is on the low end. The total count of the two lowest purchase frequency levels 'Less than once a month' and 'Once a month' tally to 231 which forms the majority count proportion. This gives a clear understanding of how frequently customers shop on Amazon, enabling the business to make strategic decisions to boost sales on the platform. <br>

![Purchase frequency counts](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/purchase%20freq%20counts.png)

## **Count of each Purchase Category** <br>

The category 'Clothing and Fashion' is the most favored among the categories shown, with the highest number of items. This indicates that many transactions are conducted within this specific category. The categories 'Beauty and Personal Care' and 'Home and Kitchen' are also popular but have much lower counts than 'Clothing and Fashion'. This shows a moderate level of interest among customers in these categories. This insight is critical towards guiding strategic marketing efforts at Amazon as it suggests a focus on strengthening the leading purchase categories individually besides grouped purchase categories while exploring growth opportunities in the other categories. <br>

<img src="https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/counts%20of%20purchase%20category.png?raw=true" width="60%">

## **Purchase Frequency vs Browsing Frequency** <br>

As browsing frequency increases, there is a noticeable shift toward more frequent purchases, particularly with higher proportions of weekly and multiple times a week purchases. Leveraging these insights, it is evident that enhancing user interaction through optimized website navigation, targeted promotions, and personalized content are ways that can significantly boost purchase frequencies. <br>

![Purchase freq vs Brows Freq](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/purchase%20vs%20browsing%20frequency.png)

## **Frequency of Purchases from Personalized Recommendations vs Personalized Recommendation Frequencies** <br>

The bar plot above reveals interesting insights into the relationship between the frequency of purchases resulting from personalized recommendations and frequency of personalized recommendations. Notably, customers who never receive recommendations still exhibit the highest purchase proportions, suggesting that factors other than recommendations significantly influence buying decisions. Conversely, customers receiving the most frequent recommendations surprisingly show moderate proportion of purchases. This indicates that an overabundance of recommendations may not effectively convert into sales, potentially due to other reasons such as discounts or vouchers given out. <br>

![purchase freq vs rec freq](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/purchases%20vs%20recommendations.png)

## **Shopping Satisfaction vs Personalized Recommendation Frequencies** <br>

The above plot shows that personalized recommendation frequencies does influence shopping satisfaction levels. There is a visible trend that as personalized recommendation frequency increases, the proportion of satisfaction levels (i.e ‘Satisfied’ , ‘Very satisfied’) increases. And the lower the personalized recommendation frequency, the higher the proportions of dissatisfaction levels (‘Very dissatisfied’, ‘Dissatisfied’). The observation suggests that Amazon's strategic investment in personalized recommendations substantially enhances the customer shopping experience. Customers who frequently receive tailored recommendations feel more valued and are more likely to enjoy their shopping experience. This also underscores the necessity for Amazon to further refine its recommendation algorithms to increase the relevance and frequency of personalized content while providing customers with a more personalized shopping experience. <br>

![shop satisfaction vs rec freq](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/Shopping%20satisfaction%20vs%20Recommendations.png)

## **Proportions of Purchase Frequencies by Age Group** <br>

Senior Citizens: Senior citizens exhibit a relatively high purchase frequency compared to other age groups, especially in the categories of "Multiple times a week" and "Once a week" This indicates a consistent engagement with the platform, suggesting that they rely on it regularly for their purchasing needs. Given their frequent interactions and trust in the platform, there is a significant opportunity for upselling and cross-selling products that cater to the unique needs of senior citizens.

Middle-Aged Adults: This group also shows strong purchase frequencies. Although they have low purchase frequency of multiple times a week, which could be due to time constraints of a working lifestyle, but their purchase frequency is still considerably high in the frequency of few times a month and once a week. This suggests that this group of customers still have a robust engagement with the platform and a routine integration of shopping into their lifestyle.

Young Adults: Young adults exhibit a balanced spread across various shopping frequencies, with a notable propensity for purchasing a few times a month and once a week. This indicates a pattern of steady yet cautious shopping habits.

Students: Students predominantly shop less frequently compared to other age groups. The largest proportion of their purchases falls into the categories of "Few times a month" and "Once a month". This suggests that their shopping habits could be more need-based than habitual or leisure-driven. <br>

![purchase frequency of age groups](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/purchase%20frequency%20age%20groups.png)

# Modelling and Evaluation
This segment of the report delves into the predictive modelling and recommender system approaches to forecast customer behaviors and enhance product recommendations. Employing advanced algorithms such as K-Nearest Neighbors, Decision Tree with Gradient Boosting, and Random Forest, this section evaluates the effectiveness of each model in predicting purchase frequencies and suggests optimal marketing strategies. Simultaneously, the recommender system section explores innovative techniques to personalize user experiences, aiming to improve customer satisfaction and engagement. Through detailed evaluation, the aim is to validate the accuracy and efficiency of these models, providing Amazon with strategic insights to refine and tailor their user interactions.

## Predictive Modelling
In this analysis, the primary goal is to enable strategic decision-making by using predictive modelling to get the predicted purchase frequency among customers. The methodologies chosen for the predictive modelling include K-Nearest Neighbors (KNN), Decision Tree with Gradient Boosting, and Random Forest algorithms, selected for their robustness in handling complex datasets. The analysis rigorously evaluates the performance of these models using several critical metrics to ensure reliability and accuracy in predictions. The insights derived from the modelling could be used to inform and optimize marketing strategies and customer engagement tactics. By understanding purchase frequency habits, Amazon can tailor better approaches to better meet customer needs and drive sales efficiency.

### Encoding
The predictor variables are classified into ordinal and nominal according to their variable types. The respective encoding style is applied accordingly. Ordinal encoding is performed for ordinal variables and one hot encoding is applied for nominal variables. On the other hand, the response variable “PchFrq” has an inherent order, thus ordinal encoding is applied. After performing the encoding, a remapping step was then performed to provide a correlation between the corresponding purchase frequency typs and each unique encoded value.

```
# Accessing the original categories in the response variable
# 'response_encoder_resp' has been fitted to the encoder and now contains the original values under 'PchFrq'
# categories_ is a list of array(s) where each array corresponds to a column variable containing all the original unique values 
# [0] accesses the first array which is 'PchFrq' column in this case
original_categories_resp = response_encoder_resp.categories_[0]

# Retrieving the corresponding names of the categories in the encoded response variable 'PchFrq'
for encoded_value, category in enumerate(original_categories_resp):
    print(f"Encoded Value: {encoded_value} corresponds to original category '{category}'")
```

### LASSO Modelling
LASSO modelling plays a crucial role for feature selection especially after multiple columns are created post encoding. This ensures that only the most impactful features are retained. By applying a penalty that reduces less significant features to zero, the model streamlines the predictive modeling process, shifting the focus to the variables that are more impactful or relevant. This targeted approach not only enhances the efficiency of the overall analysis but also supports more informed, data-driven business strategies. <br>

![lasso](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/LASSO.png)

## Predictive Model #1 - KNN 
KNN model is chosen as one of the methods of predictive modelling here for comparison for a few reasons. Firstly, as the dataset is encoded into numerical inputs, it makes the KNN model suitable. Secondly, this dataset is not very huge making KNN a suitable method. The reduction in features by the LASSO model also helps mitigate the issue of high dimensionality which would make KNN a suitable model since it does not handle high dimensionality datasets well. <br>

![knn](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/KNN.png)

## Predictive Model #2 - Decision Tree with Gradient Boosting
After encoding categorical variables (especially through methods like one-hot encoding), the feature space may become high-dimensional. Gradient Boosting effectively handles high-dimensional spaces by building shallow trees in succession to gradually improve the model’s accuracy, focusing on the most informative features at each step. <br>

![gradient boosting](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/DTGB.png)

## Predictive Model #3 - Random Forest 
One of the key strengths of Random Forest is its robustness against overfitting, especially compared to individual decision trees. This is due to its ensemble method of building many decision trees and averaging their predictions, which generally leads to improved accuracy and stability. <br>

![random forest](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/Random%20Forest.png)

## Summary of Findings
All three models show that customers perform purchases few times a month as shown by the highest count in all 3 bar plots. Moreover, this supports the bar plot results in Figure 1 which shows the purchase frequency of ‘Few times a month’ having the highest count.  This further validates that the prediction outcomes of each model is reliable. Understanding how often customers make purchases such as their purchase frequencies helps in segmenting them into groups based on their buying behavior. This segmentation can be used to tailor marketing strategies to Amazon’s needs and preferences, enhancing the effectiveness of marketing efforts.

### Evaluation of Predictive Models
The KNN model has the lowest MAE, suggesting it has the smallest average error per prediction, making it slightly better in terms of absolute error compared to the other two models. In terms of MSE, the KNN model scores the lowest MSE, indicating it is less prone to large errors compared to the others. This suggests that the KNN model not only makes smaller errors on average but also has fewer large errors in its predictions. Lastly, the KNN model shows the lowest RMSE, indicating that its predictions are, on average, closer to the actual values. The errors in the KNN predictions are smaller than those in the Gradient Boosting and Random Forest models. <br>

![models evaluation](https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/predictive%20models%20evaluation.png)

# Recommender System
The recommender system is designed to enhance the shopping experience by providing tailored product category recommendations based on historical purchase data. In this analysis, the collaborative filtering was employed, specifically the Non-Negative Matrix Factorization (NMF) method. It leverages on patterns found in past user-item interactions. This method is preferred over content-based filtering as the current objectives focus on user behaviour rather than item specifics, which is not the primary focus of this analysis.

## User Item Matrix
Here, the variable purchase category ('PchCat') is the target variable on which recommendations will be generated.

The user item matrix organizes the interactions between users and items in a structured format with rows representing users, columns representing the different combinations of purchase categories, and the cells representing the nature of the interactions (i.e whether a user has interacted with a purchase category combination). The matrix makes it easier to identify relationships between users and items. The index column of the encoded dataframe “df_recommender_encoded“ is used to represent the individual user IDs (rows). The categorical data in the variable ‘PchCat’ are converted into multiple dummy variables with each representing a unique combination of purchased product categories. The binary cell values represent interactions between each user ID and the different combinations of purchase categories.

## NMF Model
The NMF model is a robust technique in collaborative filtering. It is instrumental in deciphering complex, latent patterns within user shopping behaviours, thereby facilitating the personalization of product recommendations. By decomposing the user-item interaction matrix into user and item latent matrices, the NMF model identifies and leverages latent factors to identify unexplored product categories that customers are likely to be interested in.

## Simulation
The recommender function (recommend_items) here takes a specific user (user_id) and retrieves their predicted interaction scores (from **user_predictions_df**) for all items. It then sorts these scores in descending order, recommending the top n grouped categories for the specified user. The simulation below shows an example for the top 5 purchase category recommendations for user IDs 1 to 4. <br>

<img src="https://github.com/bayyangjie/Predictive-Analysis-of-E-Commerce-Sales/blob/main/Images/Simulation.gif?raw=true" width="100%">

# Conclusion
This study has provided deep insights into the shopping behaviours and preferences of Amazon customers, leveraging sophisticated predictive modelling and machine learning techniques. The effective application of models such as K-Nearest Neighbors, Decision Tree with Gradient Boosting, and Random Forest has illuminated key factors that influence customer purchase frequencies. This understanding is vital for Amazon to tailor its marketing strategies, enhancing user engagement and increasing sales efficiency.

Models suggest that by comprehending customer purchasing behaviors, businesses can create better customer segments and personalized communication, ultimately improving the overall shopping experience and customer happiness. Another important discovery highlights the significance of incorporating the NMF model in the recommender system to display its capability in producing precise product suggestions. This system not only enhances the relevance of product suggestions but also boosts user engagement and revenue potential by strategically placing products.

# Improvements
To further enhance the predictive accuracy and breadth of consumer behavior insights, more detailed user metrics could be collected, such as time spent on page and click-through rates. Additionally, incorporating real-time data processing and analytics into the platform will enable instant delivery of recommendations based on user actions, significantly boosting platform responsiveness.

Adidtionally, exploring more complex machine learning models, such as deep learning, could provide deeper insights into consumer behaviors and preferences, offering a more nuanced understanding of user needs and potentially leading to more effective targeting and personalization strategies.
