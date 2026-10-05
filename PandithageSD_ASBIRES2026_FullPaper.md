# **Machine Learning for Personalized Food Allergen Risk Prediction** 

<u>Pandithage SD1, Hansika Gunasekara</u><sup>2</sup> 

_1,2Department of Computing and Information Systems, Faculty of Applied Sciences, Wayamba University of Sri Lanka_ _<u>senuripandithage31@gmail.com</u>_ 1 _<u><mark>hansikag@wyb.ac.lk</mark></u> 2_ 

## **ABSTRACT** 

**Food allergy risk assessment remains challenging because responses to food ingredients differ between individuals, and relevant datasets are often highly imbalanced. In this study, a machine learning approach has been developed to estimate personalized food allergen risk using self-reported clinical, demographic, dietary, and family history information collected from 90 Sri Lankan participants. A positive target has represented a participant-reported allergic reaction, sensitivity, or perceived risk to an ingredient and has not represented a clinically confirmed allergy diagnosis. 31 candidate allergen targets have been identified, and 24 with at least 2 positive observations have been retained. Candidate features have been screened using Spearman's correlation, with feature selection and scaling performed strictly inside training folds to prevent test-set information leakage. Logistic Regression, Random Forest, and XGBoost have been compared using 5-fold stratified cross-validation with imbalance-aware training. Accuracy, precision, recall, F1-score, and PR-AUC have been used for evaluation. Random Forest has achieved the highest mean accuracy (88.15%), whereas Logistic Regression has achieved the highest mean recall (26.82%) and F1-score (20.31%), with 0.3562 PR-AUC. Logistic Regression has therefore been selected as the most suitable model for personalized risk prediction because positive-risk detection has been prioritized over accuracy alone. Apriori association rule mining has also been applied to identify strong co-occurrence patterns among self-reported allergen responses. The strongest observed rule, Mutton → Beef, has achieved 94.4% confidence and 3.15 lift. These associations have been treated as supplementary statistical evidence rather than clinically confirmed biological cross-reactivity. The findings have demonstrated the feasibility of personalized allergen-risk prediction and highlighted the need for larger, clinically validated datasets.** 

### **KEYWORDS: Food Allergy, Machine Learning, Personalized Risk Prediction, Logistic Regression, Association Rule Mining** 

## **1. INTRODUCTION** 

Food allergies continue to be a significant health issue since reactions to food components can vary from mild symptoms to life-threatening anaphylaxis (Sicherer & Sampson, 2009). Tailoring risk assessments has proven difficult due to the variability in individual reactions and the frequent imbalances in food allergy datasets. 

Machine learning has increasingly been investigated for patient-level allergy-risk prediction. Ahmad _et al._ (2025) compared several classifiers for foodallergy detection and reported higher accuracy for tree-based models than Logistic Regression. However, accuracy can be misleading when positive cases are less frequent than negative cases. The present study has therefore focused on self-reported allergen-risk prediction using data from 90 Sri Lankan participants, with recall and F1-score considered alongside accuracy. 

The study has compared Logistic Regression, Random Forest, and XGBoost under 5-fold stratified cross-validation and has selected the most suitable model for personalized risk prediction. Apriori association rule mining has also been used to identify strong co-occurrence patterns among self-reported allergen responses. 

## **2. LITERATURE REVIEW** 

Ahmad _et al._ (2025) compared Decision Tree, Random Forest, K- Nearest Neighbors, Support Vector Classifier, Naive Bayes, and Logistic Regression for food-allergy detection. Their results favored tree-based classifiers in terms of accuracy, demonstrating that algorithm performance can vary across datasets. Landau _et al._ (2024) also demonstrated the use of patient-level electronic medical record data for food-allergy risk stratification and compared Logistic Regression with Random Forest. Their work supports the use of interpretable machine learning models for individualized risk assessment. 

Association-rule mining provides a complementary approach to supervised classification. Apriori, introduced by Agrawal and Srikant (1994), identifies frequent item sets and association rules from transactional data. In the present study, it has been applied to allergenresponse data to identify recurring cooccurrence patterns that can supplement the predictive model. 

## **3. METHODOLOGY** 

_3.1 Dataset and Target Definition_  
A structured, anonymized questionnaire has been used to collect demographic, dietary, personal-allergy, medical-condition, and family-history information from 90 Sri Lankan participants. The data represent selfreported experiences rather than clinically confirmed allergy diagnoses. Thirty-one candidate food or food-related targets have been identified. A value greater than zero has been encoded as 1 to represent a participant-reported allergic reaction, sensitivity, or perceived risk to that ingredient, while zero has been encoded as 0. Targets with fewer than two positive observations have been excluded, leaving 24 eligible targets for modeling. 

_3.2 Feature Preparation and Selection_  
Candidate predictors have included age, gender, personal allergy history, frequency of outside food, lactose intolerance, medical conditions, family history, and family-reported allergy patterns. One-hot-encoded variables for province, blood type, dietary pattern, and work environment have also been included. Categorical variables have been converted using binary, ordinal, and onehot encoding. Multi-response variables have been transformed into binary indicators, and continuous variables have been normalized using MinMaxScaler. The candidate feature space initially contained 51 features. Feature screening has been performed using Spearman correlation. For each feature, the maximum absolute correlation with the allergen targets has been calculated. Features with a maximum absolute correlation of at least 0.15 have been retained, reducing the feature space to 43 informative features (8 zero-variance or uninformative features dropped). Feature selection and scaling have been performed strictly inside each training fold to prevent test-set information leakage. 

_3.3 Model Development and Evaluation_  
For each eligible target, 5-fold stratified cross-validation has been performed (random state 42). Logistic Regression, Random Forest, and XGBoost have been compared. Logistic Regression has been configured with balanced class weights, a maximum of 500 iterations, and random state 42. Random Forest has been configured with 100 estimators, a maximum depth of 5, balanced class weights, and a random state of 42. XGBoost has been configured with 100 estimators, a maximum depth of 3, a learning rate of 0.05, and a random state of 42. SMOTE has been applied to XGBoost when at least four positive cases are available; otherwise, scale_pos_weight has been used. Accuracy, precision, recall, F1-score, and precision-recall AUC (PR-AUC) have been calculated independently for each target-specific classification task across all folds, alongside out-of-fold confusion matrices. The mean value of each metric across the eligible targets has been reported for each algorithm. Recall has been emphasized because positive-risk cases have been less frequent than negative cases. 

## _3.4 Association Rule Mining Using Apriori_ 

Apriori association rule mining has been applied to the Boolean allergenresponse matrix, following Agrawal and Srikant (1994). A minimum support of 0.10 has been used, requiring an itemset to occur in at least 10% of participants. Rule length has been restricted to two items, and rules with lift of at least 2.0 have been retained. Support, confidence, and lift have been used to interpret the rules. The resulting associations have been treated as statistical co-occurrence patterns within the study population. They have not been interpreted as proof of biological cross-reactivity or clinical allergy. 

## **4. RESULTS** 

## _4.1 Dataset Characteristics_ 

90 Sri Lankan participants have been included in the dataset, and 21 have reported a pre-existing food allergy. The most frequently reported positive foodrelated responses have been pork, beef, crab, cuttlefish/squid, mutton, spicy oily foods, prawns, pineapple, milk, and tuna. 

**Figure 1. Most Frequently Reported Positive Food-Related Responses** 

These values describe the study sample and have not been treated as population prevalence estimates. 

## _4.2 Model Performance_ 

**Table 1. Macro-Averaged 5-Fold Stratified Cross-Validation Performance Across 24 Allergen Targets** 

|**Model**|**Accuracy**|**Precision**|**Recall**|**F1-score**|**PR-AUC**|
|---|---|---|---|---|---|
|Logistic Regression (Balanced)|82.08 ± 7.42%|17.97 ± 10.63%|26.82 ± 23.36%|20.31 ± 14.88%|0.3562|
|Random Forest (Balanced)|88.15 ± 5.17%|16.75 ± 14.28%|10.43 ± 14.39%|12.04 ± 15.34%|0.3530|
|XGBoost (SMOTE / Weighted)|83.52 ± 6.94%|21.19 ± 14.34%|27.48 ± 23.00%|22.08 ± 16.48%|0.3670|

Random Forest has achieved the highest mean accuracy (88.15%). Logistic Regression has achieved the highest mean recall (26.82%) and F1-score (20.31%), with 82.08% accuracy and 0.3562 PR-AUC. XGBoost has produced 83.52% accuracy, 21.19% precision, 27.48% recall, 22.08% F1-score, and 0.3670 PR-AUC. Aggregated confusion matrices showed that Logistic Regression captured 72 True Positives across validation folds compared to 28 for Random Forest, directly reducing hazardous false-negative predictions. The results have shown that the highest overall accuracy has not corresponded to the strongest positive-case detection. Logistic Regression has therefore been selected as the most suitable model for the intended personalized risk-prediction task. 

## _4.3 Apriori Association Rules_ 

**Table 2. Strongest Pairwise Associations Identified Using Apriori** 

|**Antecedent**|**Consequent**|**Confidence**|**Lift**|**Support**|
|---|---|---|---|---|
|Mutton|Beef|94.4%|3.15|18.9%|
|Mutton|Pork|94.4%|2.74|18.9%|
|Beef|Pork|85.2%|2.47|25.6%|
|Cuttlefish / Squid|Crab|83.3%|4.17|16.7%|
|Prawns|Cuttlefish / Squid|76.5%|3.82|14.4%|

Mutton → Beef and Mutton → Pork have achieved the highest confidence at 94.4%. Cuttlefish/Squid → Crab has produced the highest lift among the reported rules at 4.17. These results have indicated strong associations within the sample, but they have not established biological cross-reactivity. 

## **5. DISCUSSION** 

The results have shown that the most accurate classifier is not necessarily the most suitable for personalized risk detection. Random Forest has achieved the highest mean accuracy, whereas Logistic Regression has achieved the highest mean recall and F1-score. This distinction has been important because the positive class has been comparatively rare. Accuracy alone could therefore give an overly favorable impression of performance. 

The findings have differed from the accuracy-focused results reported by Ahmad _et al._ (2025), where tree-based classifiers achieved strong accuracy. The present findings have not demonstrated that tree-based methods are generally inferior. Instead, they have shown that model suitability has depended on the dataset, class distribution, and intended evaluation objective. In this dataset, class-weighted Logistic Regression has provided the best balance of recall and F1-score and has therefore been selected. 

The result has also been consistent with the value of interpretable models identified by Landau _et al._ (2024). Logistic Regression has provided a comparatively simple modeling structure while achieving the strongest positivecase detection among the evaluated approaches. 

Apriori has provided information that differs from supervised prediction. Logistic Regression has used participant characteristics to estimate allergenspecific risk, whereas Apriori has identified recurring relationships between self-reported allergen responses. The Mutton → Beef rule has shown a strong statistical relationship in the survey data, but it has not been sufficient to establish immunological cross-reactivity. The Apriori component has therefore been treated as supplementary evidence rather than a diagnostic mechanism. 

## **6. LIMITATIONS AND FUTURE WORK** 

The study has several limitations. The sample size has been limited to 90 participants, while multiple features and allergen targets have been modeled. Several positive classes have also contained few observations, which has limited model stability. The target labels have been self-reported and have not represented clinically confirmed allergy diagnoses. 

While 5-fold stratified cross-validation has been implemented, external validation on an independent Sri Lankan cohort should be conducted in future work. Per-allergen confidence intervals, calibration, precision-recall analysis, and targetspecific error analysis should also be reported. 

The Apriori rules have been derived from the same small survey dataset. Larger datasets and independent clinical or immunological evidence will be required before individual associations can be interpreted as biological crossreactivity. 

## **7. CONCLUSION** 

A personalized food allergen-risk prediction approach has been developed using self-reported data from 90 Sri Lankan participants. Logistic Regression, Random Forest, and XGBoost have been compared under 5-fold stratified cross-validation, and Logistic Regression has achieved the highest mean recall (26.82%) and F1-score (20.31%) with 82.08% accuracy and 0.3562 PR-AUC. It has therefore been selected as the most suitable model for the present riskprediction task. Apriori has additionally identified strong co-occurrence patterns among self-reported allergen responses, providing supplementary evidence rather than clinically confirmed crossreactivity. Larger datasets, clinically validated labels, and independent validation will be required before clinical use. 

## **REFERENCES** 

- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules in large databases. In _Proceedings of the 20th International Conference on Very Large Data Bases_ (pp. 487–499). Morgan Kaufmann. https://www.vldb.org/conf/1994/P487.PDF 

- Ahmad, Z., Farooq, A., Fuzail, M., Aziz, Y., & Aslam, N. (2025). Food allergy detection using a machine learning approach. _Kashf Journal of Multidisciplinary Research, 2_ (4), 116– 127. https://doi.org/10.71146/kjmr402 

- Landau, T., Gamrasni, K., Barlev, Y., Elizur, A., Benor, S., Mimouni, F., & Brandwein, M. (2024). A machine learning approach for stratifying risk for food allergies utilizing electronic medical record data. _Allergy, 79_ (2), 499– 502. https://doi.org/10.1111/all.15839 

- Sicherer, S. H., & Sampson, H. A. (2009). Food allergy: Recent advances in pathophysiology and treatment. _Annual Review of Medicine, 60_ (1), 261–277. https://doi.org/10.1146/annurev.med.60.042407.205711 
