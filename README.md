# titanic predictor
## 🔷 Model Training
1. Data Preprocessing (Cleaning and Feature Engineering)
* Download the original train.csv from kaggle,below is the link：
  > https://www.kaggle.com/competitions/titanic
* Handle missing values ：fill missing Age and Fare with median, Embarked with 'S'
* Simplify the Cabin column by extracting the first letter (e.g., 'B57' → 'B')
* Create new features: FamilySize, IsAlone, FareGroup
* Apply One-Hot Encoding to categorical features for Sex, Embarked, Cabin, FareGroup

2. Feature Selection and Standardization
* Select meaningful features (e.g., Pclass, Age, Sex_, etc.) for training
* Use StandardScaler to normalize Age and Fare 

3. Model Training and Saving
* Split the dataset into training and testing sets using train_test_split(80/20)
* Train an XGBoostClassifier with chosen hyperparameters
* Save the trained model to xgboost_model.json

4. Evaluation and Saving Processed Data
* Evaluate model accuracy on the test set
* Save the processed features as train_processed.csv

##　🔷 Prediction and Submission
1. Test Data Preprocessing
* Load the test.csv
* Fill missing values in Age, Fare, Embarked, and simplify Cabin
* Create new features ：FamilySize, IsAlone, FareGroup
* Apply One-Hot Encoding to match training preprocessing

2. Ensure Matching Columns with Training Data
* Load train_processed.csv to get the list of columns
* Reindex the test dataset to ensure it has the same features 

3. Load Model and Make Predictions
* Load the previously saved xgboost_model.json model
* Use it to predict survival on the test dataset

4. Generate Kaggle Submission File
* Create a new DataFrame with PassengerId and predicted Survived values
* Save the results as submission.csv 
