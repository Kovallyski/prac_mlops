import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import json
import lime
import lime.lime_tabular
from sklearn.tree import DecisionTreeClassifier

class ModelTrainer:
    def __init__(self, config_path='config.json'):
        self.model = None
        self.best_model = None
        self.history = []
        self.n_splits = 3
        self.load_config(config_path)
        self.init_storage()
        self.metrics_for_drift_det = []
        self.feature_names = None
        self.class_names = None
        self.load_best_model()
        
    def set_feature_info(self, feature_names, class_names):
        """Установка имен признаков и классов для интерпретации"""
        self.feature_names = feature_names
        self.class_names = class_names
        
    def load_config(self, config_path):
        """Загрузка конфигурации из JSON файла"""
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.model_type = self.config.get('model_type', 'logistic')
        self.hyperparams = self.config.get('hyperparams', {})
        self.metrics_threshold = self.config.get('metrics_threshold', 0.7)
        
    def init_storage(self):
        """Инициализация хранилища моделей и метрик"""
        os.makedirs(f'models_{self.model_type}', exist_ok=True)
        os.makedirs(f'reports_{self.model_type}', exist_ok=True)
        
    def train_initial_model(self, X_train, y_train):
        """Первоначальное обучение модели"""
        if self.model_type == 'logistic':
            model = LogisticRegression(**self.hyperparams, warm_start=True)
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(**self.hyperparams, warm_start=True)
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
        balanced_accuracy = balanced_accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        metrics = {
            'accuracy' : accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score' : f1
        }

        # self.models['initial'] = model
        return self.model, metrics
    
    def update_model(self, X_new, y_new):
        """Дообучение модели на новых данных"""
        print(self.model.warm_start)
        if self.model_type in ['logistic', 'random_forest'] and hasattr(self.model, "warm_start") and self.model.warm_start:
            self.model.fit(X_new, y_new)
        elif self.model_type == 'knn':
            X = np.vstack([self.model.knn_model._fit_X, X_new])
            y = np.concatenate([self.model._y, y_new])
            self.model.fit(X, y)
        else:
            raise ValueError("Unsupported")

        y_pred = self.model.predict(X_new)
        accuracy = accuracy_score(y_new, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_new, y_pred)
        f1 = f1_score(y_new, y_pred)

        metrics = {
            'accuracy' : accuracy,
            'balanced_accuracy': balanced_accuracy,
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
        model_name = None
        # Сохраняем модель если она улучшила метрики
        if accuracy >= accuracy_b:
            self.best_model = self.model
            model_name = self.save_model(metrics, best=True)
            
        return metrics, model_name
    
    def predict(self, X_test):
        """Применение лучшей модели"""
        return (self.best_model if self.best_model else self.model).predict(X_test)

    def save_model(self, metrics, best=False):
        """Сохранение модели и метрик"""
        model_name = f"model_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = os.path.join(f'models_{self.model_type}', model_name)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metrics': metrics,
                'model_type': self.model_type,
                'hyperparams': self.hyperparams
            }, f)
            
        # Сохраняем историю метрик
        history_path = os.path.join(f'reports_{self.model_type}', 'metrics_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

        if best:
            model_name = f"best_{self.model_type}.pkl"
            model_path = os.path.join(f'models_{self.model_type}', model_name)
        
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'metrics': metrics,
                    'model_type': self.model_type,
                    'hyperparams': self.hyperparams
                }, f)

        return model_name

    def load_model(self, name=None):
        if name is None:
            model_path = os.path.join(f'models_{self.model_type}', f"best_{self.model_type}.pkl")
        else:
            model_path = os.path.join(f'models_{self.model_type}', name)

        with open(model_path, 'rb') as f:
            tmp = pickle.load(f)
            self.model = tmp['model']
            self.hyperparams = tmp['hyperparams']
            self.model_type = tmp['model_type']
    
    def load_best_model(self):
        model_path = os.path.join(f'models_{self.model_type}', f"best_{self.model_type}.pkl")

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                tmp = pickle.load(f)
                self.model = tmp['model']
                self.best_model = self.model
                self.hyperparams = tmp['hyperparams']
                self.model_type = tmp['model_type']
        
        return f"best_{self.model_type}.pkl"

    def hyperparameter_tuning(self, X, y, val='CV', force_save=True):
        """Подбор гиперпараметров с кросс-валидацией"""

        if val == 'CV':
            cv = self.n_splits
        else:
            cv = TimeSeriesSplit(n_splits=self.n_splits)

        if self.model_type == 'logistic':
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(warm_start=True)

        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 150, 250],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(warm_start=True)

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
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Обновляем лучшие параметры
        self.hyperparams = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.best_score = accuracy = grid_search.best_score_
        

        
        if self.best_model or not force_save:
            accuracy_b = cross_val_score(self.best_model, X, y, cv=self.n_splits, scoring='balanced_accuracy', n_jobs=-1, verbose=1).mean()
        else:
            accuracy_b = accuracy - 1

        model_name = None

        if accuracy >= accuracy_b:
            self.best_model = self.model

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'balanced_accuracy': accuracy
            }

            model_name = self.save_model(metrics, best=True)
        
        # Сохраняем результаты подбора
        self.save_hyperparam_report(grid_search.cv_results_)

        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        metrics = {
            'accuracy' : accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score' : f1
        }
        
        return grid_search.best_params_, metrics, model_name
    
    def hyperparameter_tuning_with_preproc(self, X, y, preproc, grid=None, val='CV', force_save=True):
        """Подбор гиперпараметров и препроцессинга с кросс-валидацией"""

        if val == 'CV':
            cv = self.n_splits
        else:
            cv = TimeSeriesSplit(n_splits=self.n_splits)

        if self.model_type == 'logistic':
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(warm_start=True)

        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 150, 250],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(warm_start=True)

        elif self.model_type == 'knn':
            param_grid = {
                'n_neighbors': [3, 5, 7],
                # 'weights': ['uniform', 'distance'],
                # 'metric': ['euclidean', 'manhattan']
            }
            model = KNeighborsClassifier()

        param_grid = {'model__' + k: v for k, v in param_grid.items()}

        if grid is None:
            grid = param_grid
        else:
            grid.update(param_grid)

        
        grid_search = GridSearchCV(
                    Pipeline(steps=preproc.steps + [('model', model)]),
                    param_grid=grid,
                    cv=cv,
                    scoring='balanced_accuracy',
                    n_jobs=-1,
                    verbose=1,
                    refit=True)
        grid_search.fit(X, y)

        # Обновляем лучшие параметры
        self.hyperparams = grid_search.best_params_
        self.model = grid_search.best_estimator_.steps[-1][1]
        self.best_score = accuracy = grid_search.best_score_

        if self.best_model or not force_save:
            accuracy_b = cross_val_score(
                make_pipeline(preproc, self.best_model), X, y,
                cv=self.n_splits, scoring='balanced_accuracy', n_jobs=-1).mean()
        else:
            accuracy_b = accuracy - 1

        model_name = None

        if accuracy >= accuracy_b:
            self.best_model = self.model

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'balanced_accuracy': accuracy
            }

            model_name = self.save_model(metrics, best=True)
        
        # Сохраняем результаты подбора
        self.save_hyperparam_report(grid_search.cv_results_)

        
        y_pred = grid_search.best_estimator_.predict(X)
        accuracy = accuracy_score(y, y_pred)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        metrics = {
            'accuracy' : accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score' : f1
        }
        
        return grid_search.best_params_, metrics, grid_search.best_estimator_.steps[:-1], model_name

    def save_hyperparam_report(self, cv_results):
        """Сохранение отчета по подбору гиперпараметров"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'best_score': self.best_score,
            'best_params': self.hyperparams,
            # 'all_results': cv_results #TODO: rewrite using some other json library (orjson?) or convert numpy arrays to lists
        }
        
        report_path = os.path.join(f'reports_{self.model_type}', 'hyperparam_tuning.json')
        with open(report_path, 'w') as f:
            json.dump(report, f)
        
        # Визуализация (и она тут не существует?)
        # self.plot_hyperparam_results(cv_results)
    
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
        
        summary_path = os.path.join(f'reports_{self.model_type}', 'model_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f)
            
        return summary_path

    def detect_model_drift(self, X_new, y_new):
        """Обнаружение дрифта модели по изменению метрик"""
        y_pred = self.model.predict(X_new)
        current_accuracy = accuracy_score(y_new, y_pred)
        
        if len(self.metrics_for_drift_det) == 0:
            self.metrics_for_drift_det.append(current_accuracy)
            return False
        
        median_accuracy = np.median(self.metric_history)
        self.metric_history.append(current_accuracy)
        
        return current_accuracy < median_accuracy * 0.90

    def interpret_model(self, X, y=None, sample_size=1000):
        """Интерпретация модели в зависимости от типа"""
        if sample_size < len(X):
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
            
        if self.model_type == 'random_forest':
            self._plot_feature_importance()
        elif self.model_type == 'logistic':
            self._plot_logistic_coefficients()
        elif self.model_type == 'knn':
            self._plot_nearest_neighbors(X_sample, X_train=X)
            
        # Общие методы интерпретации
        # self.shap_analysis(X_sample)
        self.lime_analysis(X_sample[0], X)

    def lime_analysis(self, instance, X_train=None):
        """LIME анализ для интерпретации предсказаний"""
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
               # training_data=np.zeros((2, len(instance))) if self.best_model.X_train is None else self.best_model.X_train,
                training_data=X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification'
            )
            
            exp = explainer.explain_instance(
                instance, 
                self.best_model.predict_proba, 
                num_features=5
            )
            
            plot_path = os.path.join(f'reports_{self.model_type}', 'lime_explanation.html')
            exp.save_to_file(plot_path)
            print(f"LIME analysis saved to {plot_path}")
        
        except Exception as e:
            print(f"LIME analysis failed: {str(e)}")

    def _plot_feature_importance(self):
        """Визуализация важности признаков для Random Forest"""
        if self.model_type != 'random_forest':
            return
            
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance - Random Forest")
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        
        plot_path = os.path.join(f'reports_{self.model_type}', 'feature_importance.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {plot_path}")

    def _plot_logistic_coefficients(self, threshold=0):
        """Визуализация коэффициентов логистической регрессии"""
        if self.model_type != 'logistic':
            return

        coefs = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.best_model.coef_[0]
        }).sort_values('coefficient', ascending=False)

        coefs = coefs[coefs['coefficient'].abs() >= threshold]

        plt.figure(figsize=(10, 6))
        plt.barh(coefs['feature'], coefs['coefficient'])
        plt.title('Logistic Regression Coefficients')
        plt.xlabel('Coefficient value')
        plot_path = os.path.join(f'reports_{self.model_type}', 'logistic_coefficients.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Logistic coefficients visualization saved to {plot_path}")
    
    def _plot_nearest_neighbors(self, X_sample, X_train=None):
        """Визуализация ближайших соседей для KNN"""
        if self.model_type != 'knn':
            return
            
        # Получаем индексы ближайших соседей для первого примера
        distances, indices = self.best_model.kneighbors(X_sample[:1])
        
        # Создаем DataFrame для визуализации
        neighbors_df = pd.DataFrame(X_train[indices[0]], 
                                 columns=self.feature_names)
        neighbors_df['distance'] = distances[0]
        
        # Сохраняем в файл
        neighbors_path = os.path.join(f'reports_{self.model_type}', 'nearest_neighbors.csv')
        neighbors_df.to_csv(neighbors_path, index=False)
        print(f"Nearest neighbors saved to {neighbors_path}")
