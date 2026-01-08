from scipy.stats import uniform, randint


LIGHTGBM_PARAMS = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.3),
    'num_leaves': randint(20, 150),
    'max_depth': randint(5, 50),
    'boosting_type': ['gbdt', 'dart', 'goss']
}


RANDON_SEARCH_PARAMS = {
    'n_iter': 50,
    'cv': 5,
    'verbose': 1,
    'n_jobs': -1,
    'random_state': 42,
    'scoring': 'accuracy'
}

