# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Model Type:** Random Forest Classifier  
**Model Version:** 1.0  
**Date:** 2025  
**Model Architecture:** Ensemble method using 100 decision trees with maximum depth of 10  
**Framework:** scikit-learn  
**Developed by:** [Your Name/Organization]

The model uses a Random Forest classifier to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related features from census data. The model employs one-hot encoding for categorical features and label binarization for the target variable.

## Intended Use

**Primary Use Case:** Predict whether an individual's annual income is above or below $50,000 based on demographic characteristics.

**Intended Users:** 
- Data scientists and researchers studying socioeconomic patterns
- Policy makers analyzing income distribution
- Educational purposes for machine learning classification tasks

**Out-of-Scope Use Cases:**
- Making individual hiring or lending decisions (potential for discrimination)
- Real-time high-stakes financial decisions
- Use with populations significantly different from the 1994 U.S. Census data

## Training Data

**Dataset:** 1994 U.S. Census Bureau database  
**Source:** UCI Machine Learning Repository  
**Size:** Approximately 32,000 records after preprocessing  
**Train/Test Split:** 80% training, 20% testing with stratification on target variable

**Features Used:**
- **Categorical Features:** workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Continuous Features:** age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- **Target Variable:** salary (binary: <=50K, >50K)

**Preprocessing:**
- One-hot encoding applied to categorical features
- Missing values handled through imputation strategies
- Features standardized using scikit-learn preprocessing pipeline

## Evaluation Data

**Test Set:** 20% of the original dataset (approximately 6,400 records)  
**Evaluation Method:** Holdout validation with stratified sampling to maintain class distribution  
**Test Set Characteristics:** Representative sample maintaining the same demographic distributions as the training set

## Metrics

The model performance was evaluated using standard classification metrics:

**Overall Performance:**
- **Precision:** 0.8058 (80.58%)
- **Recall:** 0.5453 (54.53%) 
- **F1-Score:** 0.6504 (65.04%)

**Interpretation:**
- **Precision:** Of all individuals predicted to earn >$50K, approximately 81% actually do
- **Recall:** The model correctly identifies 55% of all individuals who actually earn >$50K  
- **F1-Score:** Balanced measure showing moderate overall performance with room for improvement

**Performance Across Demographic Slices:**
The model's performance varies across different demographic groups (detailed analysis available in `slice_output.txt`). Notable variations observed across:
- Gender groups
- Racial categories  
- Education levels
- Work classifications
- Native countries

## Ethical Considerations

**Bias and Fairness:**
- The model shows performance disparities across protected demographic groups
- Historical biases present in 1994 census data may be perpetuated by the model
- Lower recall (54.53%) means the model may systematically underpredict high income for certain groups

**Potential Harms:**
- Could reinforce existing socioeconomic stereotypes if used inappropriately
- May exhibit discriminatory patterns against minority groups
- Historical data may not reflect current economic conditions or social structures

**Mitigation Strategies:**
- Comprehensive bias testing across demographic slices has been implemented
- Model should not be used for individual decision-making without human oversight
- Regular retraining with more recent data recommended
- Fairness constraints should be considered in future model iterations

## Caveats and Recommendations

**Data Limitations:**
- Training data is from 1994 and may not reflect current economic conditions
- Limited geographic scope (U.S. Census data only)
- Some demographic groups may be underrepresented in the training data

**Model Limitations:**
- Moderate recall (55%) indicates significant false negative rate
- Performance varies substantially across demographic groups
- Model interpretability is limited due to ensemble nature of Random Forest

**Recommendations for Use:**
1. **Do not use for high-stakes individual decisions** without additional validation and human oversight
2. **Monitor for bias** when deploying across diverse populations
3. Retrain regularly with updated census data to maintain relevance and accuracy
4. Use as a screening tool only - not as the sole basis for employment or financial decisions
5. Validate performance on new datasets before deployment in production environments
6. Implement fairness monitoring by tracking performance metrics across demographic groups
7. Combine with domain expertise - use model predictions alongside human judgment and local knowledge
8. Be transparent about limitations when communicating results to stakeholders and end users
9. Consider ensemble approaches with other models to improve overall performance
10. Establish performance thresholds - define minimum acceptable precision/recall before taking action
11. Regular auditing - periodically review model decisions for potential discriminatory patterns
12. Update feature engineering as new relevant data sources become available

**Appropriate Use Cases:**

- Initial screening for large-scale surveys or research
- Resource allocation planning at population level
- Academic research on income prediction factors
- Baseline model for comparative analysis

**Inappropriate Use Cases:**

- Individual loan approval decisions
- Employment screening without additional validation
- Immigration or visa determinations
- Any legally protected decision-making without proper bias testing
