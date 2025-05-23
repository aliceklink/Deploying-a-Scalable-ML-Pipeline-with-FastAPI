# Model Card: Census Income Prediction Model

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Model Type:** Random Forest Classifier  
**Model Version:** 1.0  
**Date:** December 2025  
**Model Architecture:** Ensemble method using 100 decision trees with maximum depth of 10  
**Framework:** scikit-learn 1.5.1  
**Developed by:** Alice Klink

This model implements a Random Forest classifier to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related features from census data. The model employs one-hot encoding for categorical features and label binarization for the target variable. The Random Forest was chosen for its robustness to overfitting and ability to handle mixed data types effectively.

**Model Hyperparameters:**
- Number of estimators: 100
- Maximum depth: 10
- Random state: 42
- Number of jobs: -1 (parallel processing)

## Intended Use

**Primary Use Case:** Predict whether an individual's annual income is above or below $50,000 based on demographic characteristics from census data.

**Intended Users:** 
- Data scientists and researchers studying socioeconomic patterns and income distribution
- Policy makers and government agencies analyzing demographic trends
- Educational institutions for machine learning classification demonstrations
- Researchers conducting studies on income inequality and economic factors

**Intended Applications:**
- Population-level income distribution analysis
- Research on socioeconomic factors affecting income
- Educational demonstrations of binary classification
- Baseline model for comparative machine learning studies

**Out-of-Scope Use Cases:**
- Making individual hiring or lending decisions (high risk of discrimination)
- Real-time high-stakes financial decisions
- Individual credit scoring or loan approval
- Immigration or visa status determinations
- Use with populations significantly different from the 1994 U.S. Census data
- Any legally protected decision-making without proper bias testing

## Training Data

**Dataset:** 1994 U.S. Census Bureau database  
**Source:** UCI Machine Learning Repository  
**Original Size:** Approximately 48,842 records  
**Processed Size:** Approximately 32,000 records after data cleaning  
**Train/Test Split:** 80% training (25,600 records), 20% testing (6,400 records) with stratification on target variable

**Target Variable Distribution:**
- Income ≤$50K: Approximately 76% of samples
- Income >$50K: Approximately 24% of samples

**Features Used:**
- **Categorical Features (8):** workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Continuous Features (6):** age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- **Total Features After Encoding:** 104 features (after one-hot encoding of categorical variables)

**Data Preprocessing Applied:**
- One-hot encoding applied to all categorical features using scikit-learn's OneHotEncoder
- Missing values handled through standard imputation strategies
- Label binarization applied to target variable (<=50K → 0, >50K → 1)
- No feature scaling applied (Random Forest is robust to different scales)
- Stratified sampling used to maintain class distribution in train/test split

**Data Quality Notes:**
- Some records contained missing values denoted by "?" which were handled during preprocessing
- Final weight (fnlwgt) represents the number of people the census entry represents
- Education-num provides numerical encoding of education levels for model compatibility

## Evaluation Data

**Test Set:** 20% of the original dataset (approximately 6,400 records)  
**Evaluation Method:** Holdout validation with stratified sampling to maintain class distribution  
**Test Set Characteristics:** Representative sample maintaining the same demographic distributions as the training set

**Test Set Demographics:**
- Age range: 17-90 years with median around 37 years
- Gender distribution: Approximately 67% male, 33% female
- Education levels: Full range from elementary through doctorate degrees
- Geographic representation: All 50 states plus territories represented

**Evaluation Approach:**
- Overall performance metrics calculated on the entire test set
- Slice-based evaluation performed across all categorical features to identify performance disparities
- No data leakage between training and test sets
- Test set never used for model selection or hyperparameter tuning

## Metrics

The model performance was evaluated using standard binary classification metrics calculated on the holdout test set:

**Overall Performance Metrics:**
- **Precision:** 0.8058 (80.58%)
- **Recall:** 0.5453 (54.53%) 
- **F1-Score:** 0.6504 (65.04%)

**Metric Interpretations:**
- **Precision (80.58%):** Of all individuals predicted to earn >$50K, approximately 81% actually do earn >$50K. This indicates the model has relatively few false positives.
- **Recall (54.53%):** The model correctly identifies 55% of all individuals who actually earn >$50K. This means the model misses 45% of high earners (false negatives).
- **F1-Score (65.04%):** The harmonic mean of precision and recall, indicating moderate overall performance with room for improvement, particularly in recall.

**Performance Analysis:**
The model demonstrates a conservative prediction approach, favoring precision over recall. This means it tends to underpredict high income (>$50K) but when it does predict high income, it's usually correct. The relatively low recall indicates the model may be missing many individuals who actually earn more than $50K.

**Performance Across Demographic Slices:**
Detailed slice-based evaluation reveals significant performance variations across demographic groups:

- **Gender:** Performance differs between male and female predictions
- **Race:** Varying performance across racial categories with some groups showing lower recall
- **Education:** Higher accuracy for individuals with advanced degrees
- **Occupation:** Professional and executive occupations show better prediction accuracy
- **Workclass:** Government workers show different performance patterns than private sector
- **Geography:** Performance varies across different native countries

*Detailed slice performance metrics are available in the slice_output.txt file.*

## Ethical Considerations

**Bias and Fairness Analysis:**
This model exhibits several concerning biases that must be carefully considered:

1. **Historical Bias:** The model is trained on 1994 census data, which reflects historical economic conditions and social structures that may not represent current realities.

2. **Demographic Performance Disparities:** The model shows varying performance across protected demographic groups, with some minorities experiencing lower recall rates, potentially leading to systematic underestimation of their income potential.

3. **Representational Bias:** Certain demographic groups may be underrepresented in the training data, leading to poor performance for these populations.

4. **Algorithmic Bias:** The relatively low recall (54.53%) means the model systematically underpredicts high income for many individuals, which could disproportionately affect certain groups.

**Potential Societal Harms:**
- **Reinforcement of Stereotypes:** The model may perpetuate existing socioeconomic stereotypes present in historical data
- **Discriminatory Outcomes:** Lower performance for certain demographic groups could lead to unfair treatment if used inappropriately
- **Economic Impact:** Systematic underprediction could affect resource allocation or opportunity assessment
- **Privacy Concerns:** The model uses sensitive demographic information that could be misused

**Fairness Interventions Implemented:**
- Comprehensive bias testing across demographic slices has been conducted
- Performance metrics are reported for all major demographic groups
- Clear documentation of limitations and appropriate use cases
- Explicit warnings against high-stakes individual decision-making

**Recommendations for Ethical Use:**
1. **Human Oversight Required:** Never use for individual decisions without human review
2. **Regular Bias Monitoring:** Continuously monitor for discriminatory patterns in deployment
3. **Transparent Communication:** Always disclose model limitations to stakeholders
4. **Periodic Retraining:** Update with more recent data to reflect current economic conditions

## Caveats and Recommendations

**Model Limitations:**

1. **Temporal Limitations:** Training data is from 1994 and may not reflect current economic conditions, wage structures, or demographic patterns.

2. **Geographic Scope:** Limited to U.S. Census data and may not generalize to other countries or economic systems.

3. **Performance Limitations:** Moderate recall (55%) indicates significant false negative rate, missing many actual high earners.

4. **Demographic Representation:** Some demographic groups may be underrepresented, leading to poor performance for these populations.

5. **Feature Limitations:** Model does not include many factors that influence income such as industry trends, economic cycles, or individual career trajectories.

**Technical Recommendations:**

**For Deployment:**
1. **Validation Required:** Always validate performance on new datasets before deployment
2. **Monitoring Essential:** Implement continuous monitoring for performance degradation and bias
3. **Threshold Tuning:** Consider adjusting decision thresholds based on specific use case requirements
4. **Ensemble Approaches:** Consider combining with other models to improve overall performance

**For Improvement:**
1. **Data Updates:** Retrain with more recent census data when available
2. **Feature Engineering:** Explore additional relevant features that might improve prediction accuracy
3. **Algorithm Exploration:** Test other algorithms or ensemble methods to improve recall
4. **Bias Mitigation:** Implement fairness constraints or bias correction techniques

**Appropriate Use Cases:**
- **Population-level analysis:** Suitable for understanding broad demographic income patterns
- **Research applications:** Useful for academic studies on socioeconomic factors
- **Educational purposes:** Good for demonstrating machine learning classification concepts
- **Baseline comparisons:** Can serve as a benchmark for more sophisticated models
- **Policy research:** May inform population-level policy discussions with appropriate caveats

**Inappropriate Use Cases:**
- **Individual employment decisions:** Never use for hiring, promotion, or salary determination
- **Financial services:** Unsuitable for loan approval, credit scoring, or insurance decisions
- **Legal proceedings:** Should not inform legal decisions or sentencing
- **Immigration decisions:** Inappropriate for visa or residency determinations
- **Real-time applications:** Not suitable for time-sensitive financial decisions

**Best Practices for Use:**
1. **Always combine with domain expertise** and human judgment
2. **Establish minimum performance thresholds** before taking any action based on predictions
3. **Regular auditing** of model decisions for potential discriminatory patterns
4. **Transparent communication** about model limitations to all stakeholders
5. **Continuous learning** - update model as new data and better methods become available
6. **Stakeholder engagement** with affected communities when deploying in sensitive contexts

**Monitoring and Maintenance:**
- **Performance Monitoring:** Track precision, recall, and F1-score on new data quarterly
- **Bias Monitoring:** Evaluate performance across demographic groups monthly
- **Data Drift Detection:** Monitor for changes in input data distribution
- **Feedback Integration:** Collect and incorporate user feedback on model performance
- **Regular Retraining:** Schedule annual model updates with new data

**Contact Information:**
For questions about this model, its appropriate use, or to report issues, please contact the development team through the project repository.

**Version History:**
- v1.0 (December 2025): Initial model trained on 1994 Census data with Random Forest classifier
