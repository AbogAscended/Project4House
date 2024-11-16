import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, OneHotEncoder, StandardScaler
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from xgboost import XGBRegressor
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
import cupy as cp

def studier(trials):
    study = optuna.create_study(
        study_name='We the best house',
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=10,
            interval_steps=3
        ),
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=trials,
        show_progress_bar=True,
    )
    return study.best_params, study.best_value
# Load datasets
train = pd.read_csv('data/train.csv')
train = train.drop('Id', axis=1)
test = pd.read_csv('data/test.csv')
test_ids = test.pop('Id')

# Fill missing values for specific columns
train.fillna({'LotFrontage': 0, 'MiscVal': 0}, inplace=True)

# Feature engineering
train['year_qual'] = train['YearBuilt'] * train['OverallQual']
train['year_r_qual'] = train['YearRemodAdd'] * train['OverallQual']
train['qual_bsmt'] = train['OverallQual'] * train['TotalBsmtSF']
train['qual_fl'] = train['OverallQual'] * train['1stFlrSF']
train['qual_gr'] = train['OverallQual'] * train['GrLivArea']
train['qual_gar_area'] = train['OverallQual'] * train['GarageArea']
train['qual_gar_cars'] = train['OverallQual'] * train['GarageCars']
train['qual_bath'] = train['OverallQual'] * train['FullBath']
train['qual_bed'] = train['OverallQual'] * train['BedroomAbvGr']
train['qual_kit'] = train['OverallQual'] * train['KitchenAbvGr']
train['qual_fire'] = train['OverallQual'] * train['Fireplaces']
train['qual_wd'] = train['OverallQual'] * train['WoodDeckSF']
train['qual_op'] = train['OverallQual'] * train['OpenPorchSF']
train['qual_en'] = train['OverallQual'] * train['EnclosedPorch']
train['qual_3s'] = train['OverallQual'] * train['3SsnPorch']
train['qual_scr'] = train['OverallQual'] * train['ScreenPorch']
train['qual_pool'] = train['OverallQual'] * train['PoolArea']
train['qual_mo'] = train['OverallQual'] * train['MoSold']
train['qual_yr'] = train['OverallQual'] * train['YrSold']
train['total_sqft'] = train['GrLivArea'] + train['TotalBsmtSF']
train['total_bathrooms'] = train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (
            0.5 * train['BsmtHalfBath'])
train['house_age'] = train['YrSold'] - train['YearBuilt']
train['remod_age'] = train['YrSold'] - train['YearRemodAdd']
train['price_per_sqft'] = train['total_sqft'] * train['OverallQual']
train['garage_age'] = train['GarageYrBlt'] - train['YearBuilt']
train['total_porch'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
train['has_pool'] = (train['PoolArea'] > 0).astype(int)
train['has_garage'] = (train['GarageArea'] > 0).astype(int)
train['has_basement'] = (train['TotalBsmtSF'] > 0).astype(int)
train['total_area'] = train['total_sqft'] + train['total_porch']
train['quality_score'] = train['OverallQual'] * train['OverallCond']

test['year_qual'] = test['YearBuilt'] * test['OverallQual']
test['year_r_qual'] = test['YearRemodAdd'] * test['OverallQual']
test['qual_bsmt'] = test['OverallQual'] * test['TotalBsmtSF']
test['qual_fl'] = test['OverallQual'] * test['1stFlrSF']
test['qual_gr'] = test['OverallQual'] * test['GrLivArea']
test['qual_gar_area'] = test['OverallQual'] * test['GarageArea']
test['qual_gar_cars'] = test['OverallQual'] * test['GarageCars']
test['qual_bath'] = test['OverallQual'] * test['FullBath']
test['qual_bed'] = test['OverallQual'] * test['BedroomAbvGr']
test['qual_kit'] = test['OverallQual'] * test['KitchenAbvGr']
test['qual_fire'] = test['OverallQual'] * test['Fireplaces']
test['qual_wd'] = test['OverallQual'] * test['WoodDeckSF']
test['qual_op'] = test['OverallQual'] * test['OpenPorchSF']
test['qual_en'] = test['OverallQual'] * test['EnclosedPorch']
test['qual_3s'] = test['OverallQual'] * test['3SsnPorch']
test['qual_scr'] = test['OverallQual'] * test['ScreenPorch']
test['qual_pool'] = test['OverallQual'] * test['PoolArea']
test['qual_mo'] = test['OverallQual'] * test['MoSold']
test['qual_yr'] = test['OverallQual'] * test['YrSold']
test['total_sqft'] = test['GrLivArea'] + test['TotalBsmtSF']
test['total_bathrooms'] = test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (
            0.5 * test['BsmtHalfBath'])
test['house_age'] = test['YrSold'] - test['YearBuilt']
test['remod_age'] = test['YrSold'] - test['YearRemodAdd']
test['price_per_sqft'] = test['total_sqft'] * test['OverallQual']
test['garage_age'] = test['GarageYrBlt'] - test['YearBuilt']
test['total_porch'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']
test['has_pool'] = (test['PoolArea'] > 0).astype(int)
test['has_garage'] = (test['GarageArea'] > 0).astype(int)
test['has_basement'] = (test['TotalBsmtSF'] > 0).astype(int)
test['total_area'] = test['total_sqft'] + test['total_porch']
test['quality_score'] = test['OverallQual'] * test['OverallCond']

# Separate target variable and concatenate datasets for imputation
train_target = train.pop('SalePrice')
train_target = np.log1p(train_target)
combinedSet = pd.concat([train, test], axis=0, ignore_index=True)

# Separate transformed data back into train and test
train = combinedSet.iloc[:len(train_target)].copy()
test = combinedSet.iloc[len(train_target):].copy()

# Select categorical and numerical columns
categorical_cols = combinedSet.select_dtypes(include=['object']).columns.tolist()
numerical_cols = combinedSet.select_dtypes(include=['number']).columns.tolist()

# Define the transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Apply the preprocessor to the data
train_processed = preprocessor.fit_transform(train)
test_processed = preprocessor.transform(test)

# Split processed training data for validation
X_train, X_valid, y_train, y_valid = train_test_split(train_processed, train_target, test_size=0.2,
                                                      random_state=42)
# Tree-based Parameters
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 7),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        # fixed parameters
        'n_estimators': 10000,
        'early_stopping_rounds': 30,
        'tree_method':'hist',
        'random_state': 42,
        'device': 'cuda',
    }
    model = XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cpXtrain = cp.array(X_train)
    cpYtrain = cp.array(y_train)
    cpXvalid = cp.array(X_valid)
    cpYvalid = cp.array(y_valid)
    scores = []
    for train_idx, val_idx in kf.split(train_processed):
        X_fold_train = train_processed[train_idx]
        y_fold_train = train_target[train_idx]
        X_fold_val = train_processed[val_idx]
        y_fold_val = train_target[val_idx]

        cpX_fold_train = cp.array(X_fold_train)
        cpy_fold_train = cp.array(y_fold_train)
        cpX_fold_val = cp.array(X_fold_val)

        model.fit(
            cpX_fold_train,
            cpy_fold_train,
            eval_set=[(cpX_fold_val, cp.array(y_fold_val))],
            verbose=False
        )

        predictions = model.predict(cpX_fold_val)
        fold_score = np.sqrt(mean_squared_error(y_fold_val, predictions))
        scores.append(fold_score)
    return np.mean(scores)