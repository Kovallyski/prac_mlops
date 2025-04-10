import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import json

class ModelTrainer:
    def __init__(self, config_path='config.json'):
        self.model = None
        self.best_model = None
        self.name_of_best = None
        self.history = []
        self.n_splits = 5
        self.load_config(config_path)
        self.init_storage()
        
    def load_config(self, config_path):
        """Загрузка конфигурации из JSON файла"""
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.model_type = self.config.get('model_type', 'logistic')
        self.hyperparams = self.config.get('hyperparams', {})
        self.metrics_threshold = self.config.get('metrics_threshold', 0.7)
        
    def init_storage(self):
        """Инициализация хранилища моделей и метрик"""
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
    def train_initial_model(self, X_train, y_train):
        """Первоначальное обучение модели"""
        if self.model_type == 'logistic':
            model = LogisticRegression(**self.hyperparams)
        elif self.model_type == 'tree':
            model = DecisionTreeClassifier(**self.hyperparams)
        elif self.model_type == 'knn':
            model = KNeighborsClassifier(**self.hyperparams)
        else:
            raise ValueError("Unsupported model type")
            
        model.fit(X_train, y_train)
        self.model = model

        if self.best_model is None:
            self.best_model = self.model 

        y_pred = self.model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        metrics = {
            'accuracy' : accuracy,
            'f1_score' : f1
        }

        # self.models['initial'] = model
        return self.model, metrics
    
    def update_model(self, X_new, y_new):
        """Дообучение модели на новых данных"""
        if self.model_type == 'logistic' and hasattr(self.model, "warm_start") and self.model.warm_start:
            self.model.fit(X_new, y_new, classes=np.unique(y_new))
        elif self.model_type == 'knn':
            X = np.vstack([self.model.knn_model._fit_X, X_new])
            y = np.concatenate([self.model._y, y_new])
            self.model.fit(X, y)
        else:
            raise ValueError("Unsupported")

        y_pred = self.model.predict(X_new)
        accuracy = accuracy_score(y_new, y_pred)
        f1 = f1_score(y_new, y_pred)

        metrics = {
            'accuracy' : accuracy,
            'f1_score' : f1
        }
            
        return self.model, metrics
    
    def validate_model(self, X_test, y_test):
        """Валидация модели и сохранение результатов"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        y_pred_b = self.best_model.predict(X_test)
        accuracy_b = accuracy_score(y_test, y_pred)
        f1_b = f1_score(y_test, y_pred)

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'f1_score': f1,
        }
        
        self.history.append(metrics)
        
        # Сохраняем модель если она улучшила метрики
        if accuracy > accuracy_b:
            self.best_model = self.model
            self.save_model(metrics, best=True)
            
        return metrics
    
    def save_model(self, metrics, best=False):
        """Сохранение модели и метрик"""
        model_name = f"model_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = os.path.join('models', model_name)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metrics': metrics,
                'model_type': self.model_type,
                'hyperparams': self.hyperparams
            }, f)
            
        # Сохраняем историю метрик
        history_path = os.path.join('reports', 'metrics_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

        if best:
            model_name = f"best_{self.model_type}.pkl"
            model_path = os.path.join('models', model_name)
        
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'metrics': metrics,
                    'model_type': self.model_type,
                    'hyperparams': self.hyperparams
                }, f)

    def load_model(self, name=None):
        if name is None:
            model_path = os.path.join('models', f"best_{self.model_type}.pkl")
        else:
            model_path = os.path.join('models', name)

        with open(model_path, 'rb') as f:
            tmp = pickle.load(f)
            self.model = tmp['model']
            self.hyperparams = tmp['hyperparams']
            self.model_type = tmp['model_type']
    
    def hyperparameter_tuning(self, X, y, val='CV'):
        """Подбор гиперпараметров с временной кросс-валидацией"""

        if val == 'CV':
            cv = self.n_splits
        else:
            cv = TimSeriesSplit(n_splits=self.n_splits)

        if self.model_type == 'logistic':
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression()

        elif self.model_type == 'tree':
            param_grid = {
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
            model = DecisionTreeClassifier()

        elif self.model_type == 'knn':
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            model = KNeighborsClassifier()
            
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Обновляем лучшие параметры
        self.hyperparams = grid_search.best_params_
        self.model = grid_search.best_estimator_
        accuracy = grid_search.best_score_

        accuracy_b = cross_val_score(self.best_model, X, y, cv=self.n_splits, scoring='accuracy').mean()

        if accuracy > accuracy_b:
            self.best_model = self.model

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy
            }

            self.save_model(metrics, best=True)
        
        # Сохраняем результаты подбора
        self.save_hyperparam_report(grid_search.cv_results_)

        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        metrics = {
            'accuracy' : accuracy,
            'f1_score' : f1
        }
        
        return grid_search.best_params_, metrics
    
    def hyperparameter_tuning_with_preproc(self, X, y, preproc, grid=None, val='CV'):
        """Подбор гиперпараметров с временной кросс-валидацией"""

        if val == 'CV':
            cv = self.n_splits
        else:
            cv = TimSeriesSplit(n_splits=self.n_splits)

        if self.model_type == 'logistic':
            param_grid = {
                'model__C': [0.1, 1, 10],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear']
            }
            model = LogisticRegression()

        elif self.model_type == 'tree':
            param_grid = {
                'model__max_depth': [3, 5, 7],
                'model__min_samples_split': [2, 5, 10],
                'model__criterion': ['gini', 'entropy']
            }
            model = DecisionTreeClassifier()

        elif self.model_type == 'knn':
            param_grid = {
                'model__n_neighbors': [3, 5, 7],
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['euclidean', 'manhattan']
            }
            model = KNeighborsClassifier()

        if grid is None:
            grid = param_grid
        else:
            grid.update(param_grid)

        
        grid_search = GridSearchCV(
                    make_pipeline(preproc, model),
                    param_grid=grid,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1)
        grid_search.fit(X, y)
        
        # Обновляем лучшие параметры
        self.hyperparams = grid_search.best_params_
        self.model = grid_search.best_estimator_
        accuracy = grid_search.best_score_

        accuracy_b = cross_val_score(
            make_pipeline(preproc, self.best_model), X, y,
            cv=self.n_splits, scoring='accuracy').mean()

        if accuracy > accuracy_b:
            self.best_model = self.model

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy
            }

            self.save_model(metrics, best=True)
        
        # Сохраняем результаты подбора
        self.save_hyperparam_report(grid_search.cv_results_)

        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        metrics = {
            'accuracy' : accuracy,
            'f1_score' : f1
        }
        
        return grid_search.best_params_, metrics

    def save_hyperparam_report(self, cv_results):
        """Сохранение отчета по подбору гиперпараметров"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'best_score': self.best_score,
            'best_params': self.hyperparams,
            'all_results': cv_results
        }
        
        report_path = os.path.join('reports', 'hyperparam_tuning.json')
        with open(report_path, 'w') as f:
            json.dump(report, f)
        
        # Визуализация
        self.plot_hyperparam_results(cv_results)
    
    def generate_summary(self):
        """Генерация отчета о работе модели"""
        summary = {
            'best_model': {
                'type': self.model_type,
                'hyperparams': self.hyperparams,
                
            },
            'history': self.history,
            'last_update': datetime.now().isoformat()
        }
        
        summary_path = os.path.join('reports', 'model_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f)
            
        return summary_path