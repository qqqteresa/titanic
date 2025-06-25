# titanic predictor
A machine learning project using the Titanic dataset from Kaggle to predict passenger survival based on selected features and an XGBoost classification model.
## 🔷 Model Training
1. Data Preprocessing (Cleaning and Feature Engineering)
* Download the original train.csv from kaggle,below is the link：
  > https://www.kaggle.com/competitions/titanic
* Handle missing values ：fill missing Age and Fare with median, Embarked with 'S'
* Feature Engineering：FamilySize = SibSp + Parch + 1 
* Create new features: extract honorifics from the Name column
* Apply One-Hot Encoding to categorical features for Sex, Embarked, Title

2. Selected Features Used for Training
* Sex_female, Pclass, Title_Mr, Fare, Age, Title_Master, FamilySize,
Title_Mrs, Embarked_S, SibSp, TicketFreq, Embarked_C, Parch, Title_Rare

3. Model Training and Saving
* Split the dataset into training and testing sets using train_test_split(80/20)
* Train an XGBoostClassifier with chosen hyperparameters
* xgb_model.pkl: trained XGBoost model
* xgb_features.pkl: list of selected feature column names
* titanic_selected_features.csv: processed training dataset with final features

4. Evaluation and Saving Processed Data
* Accuracy is calculated on the 20% held-out test set using accuracy_score

## 🔷 Prediction and Submission
1. Test Data Preprocessing
* Load the test.csv
* Apply the same preprocessing as the training set

2. Load Model and Make Predictions
* Load xgb_model.pkl and xgb_features.pkl
* Apply the model to the test data

3. Generate Submission File
* columns for PassengerId, Survived
* Save predictions to submission.csv
