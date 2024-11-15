import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error


def studier(trials, jobs):
    study = optuna.create_study(
        study_name='We the best house',
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=50,
            interval_steps=10
        )
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=trials,
        n_jobs=jobs
    )
    return study.best_params, study.best_value

def objective(trial):
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

    # Extract numerical features and calculate correlation with SalePrice
    numerical_features = train.select_dtypes(include=[np.number])
    correlation_matrix = numerical_features.corr()
    high_corr_features = correlation_matrix.index[abs(correlation_matrix["SalePrice"]) > 0.8].tolist()
    high_corr_data = train[high_corr_features]

    # Separate target variable and concatenate datasets for imputation
    train_target = train.pop('SalePrice')
    combinedSet = pd.concat([train, test], axis=0, ignore_index=True)

    # Separate transformed data back into train and test
    train = combinedSet.iloc[:len(train_target)].copy()
    test = combinedSet.iloc[len(train_target):].copy()

    # Select categorical and numerical columns
    categorical_cols = combinedSet.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = combinedSet.select_dtypes(include=['number']).columns.tolist()

    # Define the transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=0))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
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
                                                          random_state=723894)
    # Tree-based Parameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),

        # Number of trees (usually fixed, but can be optimized)
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),

        # Learning Parameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),

        # Regularization Parameters
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),

        # Other Parameters
        'random_state': 56486875,
        'device': 'cuda'
    }
    if params['booster'] == 'dart':
        params.update({
            'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
            'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
            'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
            'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5)
        })
    model = XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=50,
        verbose=False
    )
    predictions = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))
    return rmse
