import os
import sys
import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluation_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting input features and target variable")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'Gradient Boosting Regression': GradientBoostingRegressor(),
                'Random Forest Regression': RandomForestRegressor()
            }

            params = {
                "Linear Regression": {},
                "Ridge Regression": {
                    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ['auto', 'svd', 'cholesky', 'sparse_cg', 'saga'],
                    "max_iter": [100, 200]
                },
                "Lasso Regression": {
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                    "max_iter": [1000, 2000, 5000],
                    "selection": ['cyclic', 'random']
                },
                "Gradient Boosting Regression": {
                    "learning_rate": [0.1, 0.01],
                    "max_depth": [3, 5, 7],
                    "min_samples_leaf": [1, 2],
                    "min_samples_split": [2, 4],
                    "n_estimators": [100, 200]
                },
                "Random Forest Regression": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            }

            model_report = {}

            for model_name, model in models.items():
                logging.info(f"Tuning: {model_name}")

                param_grid = params.get(model_name, {})
                if param_grid:
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        scoring='r2',
                        cv=5,
                        n_jobs=-1,
                        verbose=1
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    logging.info(f"Best params for {model_name}: {grid_search.best_params_}")
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                evaluation_model(model_name, mse, r2)

                model_report[model_name] = {
                    "model": best_model,
                    "mse": mse,
                    "r2_score": r2
                }

            best_model_name = max(model_report, key=lambda k: model_report[k]["r2_score"])
            best_model_info = model_report[best_model_name]
            best_model = best_model_info["model"]

            logging.info(f"Best model is {best_model_name} with R2: {best_model_info['r2_score']:.4f}")

            save_object(
                file_path=os.path.join("artifacts", f"{best_model_name}_model.pkl"),
                obj=best_model
            )
            logging.info(f"Saved best model {best_model_name} successfully")

            return {
                "best_model_name": best_model_name,
                "best_model_object": best_model,
                "best_model_r2": best_model_info["r2_score"],
                "best_model_mse": best_model_info["mse"],
                "preprocessor_path": preprocessor_path
            }

        except Exception as e:
            logging.error("Exception occurred in ModelTrainer")
            raise CustomException(e, sys)
