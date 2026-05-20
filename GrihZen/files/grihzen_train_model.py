"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           GrihZen ML Training Framework - grihzen_train_model.py            ║
║           Author: [Your Name] | MS Data Science                              ║
║           Purpose: Train custom models on household & shopping data          ║
╚══════════════════════════════════════════════════════════════════════════════╝

MODELS TO TRAIN:
    1. Demand Forecasting  → Predict when items run out
    2. Route Efficiency    → Learn optimal routes from history
    3. Nutrition Optimizer → Recommend nutritionally balanced recipes
    4. Visit Pattern       → Predict shopping behavior by day/time
    5. Shopping Recommender → Collaborative filtering for items

USAGE:
    python grihzen_train_model.py --model demand --user_id 1
    python grihzen_train_model.py --model route --evaluate
    python grihzen_train_model.py --model all --export
"""

import os
import json
import sqlite3
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any

# ── OPTIONAL IMPORTS (install as needed) ─────────────────────────────────────
# TODO: Uncomment the imports you plan to use

# sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, f1_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# TODO: Install xgboost → pip install xgboost
# import xgboost as xgb

# TODO: Install lightgbm → pip install lightgbm
# import lightgbm as lgb

# TODO: For time series
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from prophet import Prophet  # pip install prophet

# TODO: For deep learning → pip install torch
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset

# TODO: For visualization
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# TODO: Set your database path
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "grihzen_data",
    "grihzen.db"
)

# TODO: Set your model save directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grihzen_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# TODO: Set your experiment tracking (optional)
EXPERIMENT_NAME = "grihzen_v1"
LOG_EXPERIMENTS = True  # Set True to log all model results

# ── DATA LOADING ──────────────────────────────────────────────────────────────
class GrihZenDataLoader:
    """
    Load and preprocess GrihZen data for ML training.
    
    TODO: Add your own data sources here:
    - External grocery datasets (Kaggle, UCI)
    - USDA nutrition database
    - Indian food datasets
    - Synthetic data generators
    """
    
    def __init__(self, db_path: str = DB_PATH, user_id: Optional[int] = None):
        self.db_path = db_path
        self.user_id = user_id
        self.conn = None
        
    def connect(self):
        """Connect to GrihZen SQLite database"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        return self
    
    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None
    
    # ── INVENTORY DATA ────────────────────────────────────────────────────────
    def load_inventory(self) -> pd.DataFrame:
        """
        Load personal inventory data.
        
        Columns: user_id, item_id, name, quantity, unit, threshold,
                 category, calories_per_100g, protein_per_100g, etc.
        
        TODO: Feature engineering ideas:
        - days_until_empty = quantity / avg_daily_consumption
        - stock_ratio = quantity / threshold
        - category_encoding (one-hot or label)
        - nutrition_score per item
        """
        query = "SELECT * FROM inventory"
        if self.user_id:
            query += f" WHERE user_id = {self.user_id}"
        return pd.read_sql_query(query, self.conn)
    
    # ── VISIT DATA ────────────────────────────────────────────────────────────
    def load_visits(self, days: int = 365) -> pd.DataFrame:
        """
        Load personal visit history.
        
        Columns: user_id, location_id, location_type, location_name, visited_at
        
        TODO: Feature engineering ideas:
        - day_of_week (0-6)
        - hour_of_day (0-23)
        - week_of_year
        - is_weekend (bool)
        - days_since_last_visit (per location)
        - visit_frequency (visits per month)
        - visit_sequence (position in typical route)
        """
        query = f"""
        SELECT * FROM visits
        WHERE visited_at >= datetime('now', '-{days} days')
        """
        if self.user_id:
            query += f" AND user_id = {self.user_id}"
        df = pd.read_sql_query(query, self.conn)
        if not df.empty:
            df['visited_at'] = pd.to_datetime(df['visited_at'])
            # TODO: Add your feature engineering here
            df['day_of_week'] = df['visited_at'].dt.dayofweek
            df['hour'] = df['visited_at'].dt.hour
            df['week'] = df['visited_at'].dt.isocalendar().week
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['month'] = df['visited_at'].dt.month
        return df
    
    # ── RECIPE DATA ───────────────────────────────────────────────────────────
    def load_recipes(self, household_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load household recipe data.
        
        TODO: Feature engineering ideas:
        - ingredient_count
        - nutrition_density (calories per gram)
        - category_encoding
        - cuisine_encoding
        - cook_time_bucket (quick/medium/long)
        - ingredient_variety (unique ingredient types)
        """
        if household_id:
            query = f"SELECT * FROM household_recipes WHERE household_id = {household_id}"
        else:
            query = "SELECT * FROM household_recipes"
        df = pd.read_sql_query(query, self.conn)
        if not df.empty:
            # Parse JSON columns
            df['ingredients_dict'] = df['ingredients'].apply(
                lambda x: json.loads(x) if x else {}
            )
            df['nutrition_dict'] = df['nutrition'].apply(
                lambda x: json.loads(x) if x else {}
            )
            # TODO: Extract nutrition features
            df['calories'] = df['nutrition_dict'].apply(lambda x: x.get('cal', 0))
            df['protein'] = df['nutrition_dict'].apply(lambda x: x.get('pro', 0))
            df['ingredient_count'] = df['ingredients_dict'].apply(len)
        return df
    
    # ── SYNTHETIC DATA GENERATORS ─────────────────────────────────────────────
    def generate_synthetic_visits(self, n_users: int = 100, 
                                   n_days: int = 365) -> pd.DataFrame:
        """
        Generate synthetic visit data for training when real data is limited.
        
        TODO: Customize patterns to match:
        - Indian household shopping patterns
        - Weekly grocery cycles
        - Festival/seasonal spikes
        - Different user archetypes
        """
        locations = ['Desi Brother', 'Walmart', 'H-Mart', 'Chipotle', 'Taj Chaat']
        data = []
        
        for user_id in range(1, n_users + 1):
            # TODO: Add different user archetypes
            # Heavy shopper: visits 3-4 times/week
            # Light shopper: visits 1-2 times/week
            # Weekend shopper: mostly Saturday/Sunday
            
            shopping_frequency = np.random.choice(['heavy', 'medium', 'light'],
                                                    p=[0.2, 0.5, 0.3])
            visits_per_week = {'heavy': 4, 'medium': 2, 'light': 1}[shopping_frequency]
            
            for day in range(n_days):
                date = datetime.now() - timedelta(days=n_days - day)
                
                # Simulate shopping behavior
                if np.random.random() < visits_per_week / 7:
                    # TODO: Add Indian festival boosts
                    # if is_festival_week(date): multiply by 1.5
                    
                    n_locations = np.random.randint(1, 4)
                    visited = np.random.choice(locations, n_locations, replace=False)
                    
                    for loc in visited:
                        data.append({
                            'user_id': user_id,
                            'location_name': loc,
                            'visited_at': date + timedelta(
                                hours=np.random.randint(9, 20)
                            ),
                            'day_of_week': date.weekday(),
                            'is_weekend': int(date.weekday() >= 5),
                        })
        
        return pd.DataFrame(data)
    
    def generate_synthetic_inventory(self, n_users: int = 100) -> pd.DataFrame:
        """
        Generate synthetic inventory consumption data.
        
        TODO: Add realistic Indian household consumption rates:
        - Rice: 2-3 kg/week for family of 4
        - Dal: 500g-1kg/week
        - Vegetables: varies by season
        """
        items = {
            'basmati_rice': {'avg_weekly_consumption': 2.5, 'unit': 'kg', 'threshold': 1},
            'urad_dal': {'avg_weekly_consumption': 0.5, 'unit': 'kg', 'threshold': 0.2},
            'chicken': {'avg_weekly_consumption': 1.5, 'unit': 'lbs', 'threshold': 0.5},
            'milk': {'avg_weekly_consumption': 3.5, 'unit': 'liters', 'threshold': 1},
            'spinach': {'avg_weekly_consumption': 300, 'unit': 'g', 'threshold': 100},
        }
        
        data = []
        for user_id in range(1, n_users + 1):
            for item_id, props in items.items():
                # TODO: Add household size factor
                # TODO: Add seasonal variation
                # TODO: Add preference variation per user
                
                consumption_rate = np.random.normal(
                    props['avg_weekly_consumption'],
                    props['avg_weekly_consumption'] * 0.2  # 20% variance
                )
                data.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'weekly_consumption': max(0, consumption_rate),
                    'threshold': props['threshold'],
                    'unit': props['unit'],
                })
        
        return pd.DataFrame(data)


# ── MODEL 1: DEMAND FORECASTING ───────────────────────────────────────────────
class DemandForecastingModel:
    """
    Predict when inventory items will run out.
    
    Problem Type: Regression / Time Series
    
    Target: days_until_empty
    Features: current_quantity, threshold, avg_consumption, day_of_week,
              month, week_of_year, item_category, user_shopping_frequency
    
    TODO: Algorithms to compare:
        - Linear Regression (baseline)
        - Random Forest Regressor
        - Gradient Boosting Regressor  
        - XGBoost
        - ARIMA (time series)
        - Prophet (Facebook's model)
        - LSTM (deep learning - best for sequences)
    
    Evaluation Metrics:
        - MAE (Mean Absolute Error) - primary metric
        - RMSE (Root Mean Squared Error)
        - R² Score
        - MAPE (Mean Absolute Percentage Error)
    
    Time Complexity:
        - Linear Regression: O(n·p) train, O(p) predict
        - Random Forest: O(n·log(n)·trees) train, O(trees·log(n)) predict
        - XGBoost: O(n·log(n)·trees·depth) train
        - LSTM: O(n·seq_len·hidden²) train
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = []
        self.label_encoders = {}
    
    def prepare_features(self, inventory_df: pd.DataFrame, 
                          visits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering for demand forecasting.
        
        TODO: Add more features:
        - Rolling average consumption (7-day, 30-day)
        - Seasonal indicators (festival weeks)
        - Shopping frequency per user
        - Item substitutes (if A runs low, B usage increases)
        - Weather correlation (optional)
        """
        features = []
        
        for _, item in inventory_df.iterrows():
            # Basic features
            row = {
                'quantity': item['quantity'],
                'threshold': item['threshold'],
                'stock_ratio': item['quantity'] / max(item['threshold'], 0.001),
                'calories': item.get('calories_per_100g', 0),
                'protein': item.get('protein_per_100g', 0),
            }
            
            # Category encoding
            # TODO: Use one-hot encoding for categories
            row['category'] = item.get('category', 'other')
            
            # TODO: Calculate actual consumption from visit patterns
            # row['avg_daily_consumption'] = calculate_consumption(item, visits_df)
            
            # TODO: Add temporal features
            now = datetime.now()
            row['day_of_week'] = now.weekday()
            row['month'] = now.month
            row['week_of_year'] = now.isocalendar()[1]
            
            # TODO: Target variable
            # row['days_until_empty'] = estimate_days_until_empty(item)
            
            features.append(row)
        
        return pd.DataFrame(features)
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train and compare multiple models.
        
        TODO: Add more models:
        - XGBoost: best performance usually
        - LightGBM: fastest training
        - Prophet: best for time series with seasonality
        - LSTM: best for long-term patterns
        """
        # Define models to compare
        self.models = {
            # Baseline
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            
            # Tree-based (usually best)
            'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            
            # Other
            'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
            'SVR': SVR(kernel='rbf', C=1.0),
            
            # TODO: Uncomment when xgboost installed
            # 'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        }
        
        results = {}
        print("\n" + "="*70)
        print(f"{'MODEL':<30} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'TIME':>10}")
        print("="*70)
        
        for name, model in self.models.items():
            try:
                start = datetime.now()
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[name] = scaler
                
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Evaluate
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                elapsed = (datetime.now() - start).total_seconds() * 1000
                
                results[name] = {
                    'mae': mae, 'rmse': rmse, 'r2': r2,
                    'time_ms': elapsed, 'model': model,
                    'y_pred': y_pred
                }
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                            cv=5, scoring='neg_mean_absolute_error')
                results[name]['cv_mae'] = -cv_scores.mean()
                results[name]['cv_std'] = cv_scores.std()
                
                print(f"{name:<30} {mae:>10.3f} {rmse:>10.3f} {r2:>10.3f} {elapsed:>8.1f}ms")
                
            except Exception as e:
                print(f"{name:<30} ERROR: {e}")
        
        print("="*70)
        
        # Find best model (lowest MAE)
        if results:
            self.best_model_name = min(results, key=lambda x: results[x]['mae'])
            self.best_model = results[self.best_model_name]['model']
            print(f"\n🏆 Best Model: {self.best_model_name}")
            print(f"   MAE: {results[self.best_model_name]['mae']:.3f}")
            print(f"   R²: {results[self.best_model_name]['r2']:.3f}")
        
        self.results = results
        return results
    
    def hyperparameter_tuning(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """
        TODO: Add hyperparameter grids for each model type.
        
        Random Forest:
            n_estimators: [50, 100, 200, 300]
            max_depth: [None, 5, 10, 20]
            min_samples_split: [2, 5, 10]
            min_samples_leaf: [1, 2, 4]
        
        Gradient Boosting:
            n_estimators: [50, 100, 200]
            learning_rate: [0.01, 0.1, 0.2]
            max_depth: [3, 5, 7]
            subsample: [0.8, 1.0]
        """
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5],
            },
            # TODO: Add XGBoost params
            # 'XGBoost': {
            #     'n_estimators': [100, 200, 300],
            #     'learning_rate': [0.01, 0.1, 0.3],
            #     'max_depth': [3, 6, 9],
            #     'subsample': [0.8, 1.0],
            #     'colsample_bytree': [0.8, 1.0],
            # }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid for {model_name}")
            return None
        
        # TODO: Add your implementation here
        # grid_search = GridSearchCV(
        #     self.models[model_name],
        #     param_grids[model_name],
        #     cv=5, scoring='neg_mean_absolute_error',
        #     n_jobs=-1, verbose=1
        # )
        # grid_search.fit(X, y)
        # return grid_search.best_params_
        pass
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        TODO: Add SHAP values for better interpretability
        """
        if not self.best_model or not hasattr(self.best_model, 'feature_importances_'):
            print("Best model doesn't support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # TODO: Plot feature importance
        # fig = px.bar(importance_df, x='importance', y='feature', orientation='h')
        # fig.show()
        
        return importance_df
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict days until empty for given inventory"""
        if not self.best_model:
            raise ValueError("Model not trained yet!")
        scaler = self.scalers.get(self.best_model_name)
        X_scaled = scaler.transform(X) if scaler else X
        return self.best_model.predict(X_scaled)
    
    def save(self, path: Optional[str] = None):
        """Save best model to disk"""
        import pickle
        path = path or os.path.join(MODEL_DIR, 'demand_forecast.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'scaler': self.scalers.get(self.best_model_name),
                'model_name': self.best_model_name,
                'feature_columns': self.feature_columns,
                'results': {k: {m: v for m, v in r.items() if m not in ['model','y_pred']}
                           for k, r in self.results.items()},
            }, f)
        print(f"✅ Model saved: {path}")
    
    def load(self, path: Optional[str] = None):
        """Load saved model"""
        import pickle
        path = path or os.path.join(MODEL_DIR, 'demand_forecast.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.best_model = data['model']
        self.scalers[data['model_name']] = data['scaler']
        self.best_model_name = data['model_name']
        self.feature_columns = data['feature_columns']
        print(f"✅ Model loaded: {data['model_name']}")


# ── MODEL 2: VISIT PATTERN CLASSIFIER ────────────────────────────────────────
class VisitPatternModel:
    """
    Predict which day/time user is likely to shop.
    
    Problem Type: Multi-class Classification
    
    Target: location_name (which store will they visit)
    Features: day_of_week, hour, month, is_weekend, 
              days_since_last_visit, inventory_low_count
    
    TODO: Algorithms:
        - Logistic Regression (baseline)
        - Random Forest Classifier
        - Gradient Boosting Classifier
        - XGBoost Classifier
        - Neural Network (MLP)
    
    Time Complexity:
        - Logistic Regression: O(n·p·iterations) train
        - Random Forest: O(n·log(n)·trees) train
        - XGBoost: best accuracy/speed tradeoff
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.results = {}
    
    def prepare_features(self, visits_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO: Add more features:
        - Previous visit location (Markov chain features)
        - Low stock items count (drives shopping urgency)
        - Time since last visit to each location
        - Payday proximity (if spending increases end of month)
        """
        if visits_df.empty:
            return np.array([]), np.array([])
        
        # Temporal features
        feature_cols = ['day_of_week', 'hour', 'month', 'is_weekend', 'week']
        
        # TODO: Add more features
        # visits_df['days_since_last'] = calculate_days_since_last(visits_df)
        # visits_df['low_stock_count'] = get_low_stock_count(visits_df['user_id'])
        
        self.feature_columns = [c for c in feature_cols if c in visits_df.columns]
        X = visits_df[self.feature_columns].fillna(0).values
        y = self.label_encoder.fit_transform(visits_df['location_name'].values)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train and compare classification models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            # TODO: Add XGBoost, LightGBM, MLP
        }
        
        results = {}
        print("\n" + "="*70)
        print(f"{'MODEL':<30} {'ACC':>8} {'F1':>8} {'AUC':>8} {'TIME':>10}")
        print("="*70)
        
        for name, model in models.items():
            try:
                start = datetime.now()
                X_train_s = self.scaler.fit_transform(X_train)
                X_test_s = self.scaler.transform(X_test)
                
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                y_prob = model.predict_proba(X_test_s) if hasattr(model, 'predict_proba') else None
                
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                elapsed = (datetime.now() - start).total_seconds() * 1000
                
                # AUC for multiclass
                auc = "-"
                if y_prob is not None and len(np.unique(y)) > 1:
                    try:
                        auc = f"{roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'):.3f}"
                    except: pass
                
                results[name] = {'accuracy': acc, 'f1': f1, 'time_ms': elapsed, 'model': model}
                print(f"{name:<30} {acc:>8.3f} {f1:>8.3f} {str(auc):>8} {elapsed:>8.1f}ms")
                
            except Exception as e:
                print(f"{name:<30} ERROR: {e}")
        
        print("="*70)
        
        if results:
            best = max(results, key=lambda x: results[x]['f1'])
            self.model = results[best]['model']
            print(f"\n🏆 Best: {best} (F1={results[best]['f1']:.3f})")
        
        self.results = results
        return results


# ── MODEL 3: ROUTE EFFICIENCY LEARNER ────────────────────────────────────────
class RouteEfficiencyModel:
    """
    Learn from historical routes to predict optimal routes.
    
    Problem Type: Regression + Optimization
    
    This is a novel contribution - no existing model does this for
    household-level multi-stop route optimization!
    
    TODO: Approach options:
        1. Learn route quality from past trips (supervised)
        2. Cluster common routes (unsupervised)
        3. Reinforcement learning (optimal routes over time)
        4. Graph Neural Network (location relationships)
    
    Features:
        - Number of stops
        - Total distance (haversine)
        - Time of day started
        - Day of week
        - Items needed (from inventory)
        - Historical success (did user finish all stops?)
    
    Target:
        - Route efficiency score (time saved vs naive route)
        - Predicted total time
    """
    
    def __init__(self):
        # TODO: Define your model architecture here
        self.model = None
        self.results = {}
        self.cluster_model = None
    
    def cluster_routes(self, visits_df: pd.DataFrame, n_clusters: int = 5):
        """
        Cluster visit sequences to find common route patterns.
        
        TODO: Implement sequence clustering:
        - Convert visit sequences to feature vectors
        - Use K-Means or DBSCAN
        - Identify "typical" shopping routes
        - Use as basis for recommendations
        """
        # TODO: Your implementation here
        # Example: cluster by number of stops + day of week
        if visits_df.empty:
            return None
        
        # Group by user + date to get route sequences
        # visits_df['date'] = visits_df['visited_at'].dt.date
        # routes = visits_df.groupby(['user_id', 'date'])['location_name'].apply(list).reset_index()
        
        # TODO: Convert routes to feature vectors (bag of words style)
        # vectorizer = CountVectorizer()
        # X = vectorizer.fit_transform(routes['location_name'].apply(' '.join))
        
        # TODO: Cluster
        # self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        # routes['cluster'] = self.cluster_model.fit_predict(X.toarray())
        
        print("TODO: Implement route clustering")
        return None
    
    def predict_route_quality(self, route_features: Dict) -> float:
        """
        Predict if a given route will be efficient.
        
        TODO: Features to use:
        - n_stops: number of destinations
        - total_distance_estimate: sum of haversine distances
        - start_time: hour of day
        - day_of_week: 0-6
        - historical_avg_time: user's typical time for similar routes
        
        Returns:
            efficiency_score: 0-100 (100 = perfect route)
        """
        # TODO: Your implementation here
        pass


# ── MODEL 4: NUTRITION OPTIMIZER ─────────────────────────────────────────────
class NutritionOptimizer:
    """
    Recommend nutritionally balanced meals based on inventory.
    
    Problem Type: Constrained Optimization + Recommendation
    
    This is your most novel contribution - cultural nutrition-aware
    recommendation system for Indian households!
    
    TODO: Approach:
        1. Collaborative filtering (user × recipe matrix)
        2. Content-based filtering (recipe nutrition features)
        3. Hybrid approach (best of both)
        4. Reinforcement learning (reward = good nutrition)
    
    Novel Elements:
        - Cultural constraints (vegetarian days, fasting, festivals)
        - Regional preferences (South Indian vs North Indian)
        - Family size scaling
        - Seasonal ingredient availability
        - Budget constraints
    """
    
    def __init__(self):
        self.model = None
        self.recipe_matrix = None
        self.results = {}
    
    def build_user_recipe_matrix(self, cook_history: pd.DataFrame) -> np.ndarray:
        """
        TODO: Build user-recipe interaction matrix for collaborative filtering.
        
        Rows: users
        Columns: recipes
        Values: cook frequency (0, 1, 2, 3...)
        
        Then use Matrix Factorization (SVD) or Neural Collaborative Filtering
        """
        # TODO: Your implementation here
        pass
    
    def optimize_weekly_menu(self, inventory: pd.DataFrame, 
                              recipes: pd.DataFrame,
                              nutritional_goals: Dict) -> List[Dict]:
        """
        Find optimal weekly meal plan given constraints.
        
        TODO: Use Linear Programming (scipy.optimize) or
              Genetic Algorithm to solve:
        
        minimize: nutritional_gap (difference from goals)
        subject to:
            - Use ingredients in inventory (reduce waste)
            - Budget constraint
            - Variety constraint (no same recipe twice in 3 days)
            - Cultural constraints (vegetarian days etc.)
        
        Returns:
            List of {day: 'Monday', meal_type: 'dinner', recipe_id: '...'}
        """
        # TODO: Your implementation here
        print("TODO: Implement weekly menu optimization")
        return []
    
    def collaborative_filter(self, user_id: int, n_recommendations: int = 5) -> List[str]:
        """
        TODO: Recommend recipes based on similar users' cooking patterns.
        
        Steps:
        1. Build user-recipe matrix
        2. Apply SVD or NMF decomposition
        3. Compute user similarity
        4. Recommend recipes that similar users cook
        
        Can also use:
        - Neural Collaborative Filtering (NCF)
        - DeepFM
        - Factorization Machines
        """
        # TODO: Your implementation here
        return []


# ── MODEL 5: SHOPPING RECOMMENDER ────────────────────────────────────────────
class ShoppingRecommender:
    """
    Recommend items to buy based on patterns, inventory, and household behavior.
    
    Problem Type: Recommendation System
    
    Features:
        - Current inventory levels
        - Historical shopping frequency per item
        - Seasonal patterns
        - Recipe requirements
        - Other household members' usage patterns
    
    TODO: Algorithms:
        - Association Rules (Apriori/FP-Growth) - "Users who buy X also buy Y"
        - Matrix Factorization
        - Graph-based recommendations
        - Deep Learning (session-based)
    """
    
    def __init__(self):
        self.association_rules = None
        self.model = None
    
    def mine_association_rules(self, purchase_history: pd.DataFrame, 
                                min_support: float = 0.1,
                                min_confidence: float = 0.5):
        """
        TODO: Find items frequently bought together.
        
        pip install mlxtend
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
        
        This tells you: "People who buy Basmati Rice also buy Urad Dal 80% of the time"
        Useful for creating smart shopping list suggestions!
        """
        # TODO: Your implementation here
        print("TODO: Implement association rule mining")
        print("Install: pip install mlxtend")
    
    def predict_next_purchase(self, user_id: int, 
                               inventory: pd.DataFrame) -> List[Dict]:
        """
        TODO: Predict what items user needs to buy next.
        
        Based on:
        - Days since last purchase of each item
        - Average consumption rate
        - Current stock level
        - Seasonal trends
        
        Returns:
            List of {item_id, urgency_score, predicted_days_until_empty}
        """
        # TODO: Your implementation here
        return []


# ── EXPERIMENT TRACKING ───────────────────────────────────────────────────────
class ExperimentTracker:
    """
    Track all model experiments, hyperparameters, and results.
    
    TODO: Consider using MLflow for professional tracking:
        import mlflow
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
    """
    
    def __init__(self, experiment_name: str = EXPERIMENT_NAME):
        self.experiment_name = experiment_name
        self.experiments = []
        self.log_path = os.path.join(MODEL_DIR, 'experiments.json')
    
    def log(self, model_name: str, model_type: str, 
            params: Dict, metrics: Dict, notes: str = ""):
        """Log an experiment result"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment': self.experiment_name,
            'model_name': model_name,
            'model_type': model_type,
            'params': params,
            'metrics': metrics,
            'notes': notes,
        }
        self.experiments.append(entry)
        
        # Save to disk
        existing = []
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                existing = json.load(f)
        existing.append(entry)
        with open(self.log_path, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print(f"📊 Logged: {model_name} | {metrics}")
    
    def print_leaderboard(self):
        """Print all experiments sorted by performance"""
        if not self.experiments:
            print("No experiments logged yet")
            return
        
        print("\n" + "="*80)
        print(f"{'EXPERIMENT LEADERBOARD':^80}")
        print("="*80)
        df = pd.DataFrame(self.experiments)
        # TODO: Sort by primary metric
        print(df[['model_name','model_type','metrics','timestamp']].to_string())


# ── MAIN TRAINING PIPELINE ───────────────────────────────────────────────────
def run_training_pipeline(args):
    """
    Main training pipeline.
    
    TODO: Add your training logic here.
    This is where you orchestrate all models.
    """
    print("\n" + "🧠"*35)
    print("  GrihZen ML Training Pipeline")
    print("🧠"*35)
    
    tracker = ExperimentTracker()
    
    # ── Load Data ──
    print("\n📁 Loading data...")
    loader = GrihZenDataLoader(user_id=args.user_id if hasattr(args,'user_id') else None)
    
    try:
        loader.connect()
        inventory_df = loader.load_inventory()
        visits_df = loader.load_visits(days=365)
        print(f"  ✅ Inventory: {len(inventory_df)} items")
        print(f"  ✅ Visits: {len(visits_df)} records")
    except FileNotFoundError:
        print("  ⚠️  Database not found. Generating synthetic data...")
        inventory_df = loader.generate_synthetic_inventory(n_users=50)
        visits_df = loader.generate_synthetic_visits(n_users=50, n_days=180)
        print(f"  ✅ Synthetic inventory: {len(inventory_df)} records")
        print(f"  ✅ Synthetic visits: {len(visits_df)} records")
    finally:
        loader.disconnect()
    
    # ── Model 1: Demand Forecasting ──
    if args.model in ['demand', 'all']:
        print("\n" + "─"*60)
        print("📈 MODEL 1: Demand Forecasting")
        print("─"*60)
        
        demand_model = DemandForecastingModel()
        
        # TODO: Prepare your features here
        # X, y = demand_model.prepare_features(inventory_df, visits_df)
        
        # TODO: Uncomment when features are ready
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # results = demand_model.train_all_models(X_train, y_train, X_test, y_test)
        # demand_model.save()
        
        print("  TODO: Implement feature engineering in prepare_features()")
        print("  See: DemandForecastingModel.prepare_features()")
    
    # ── Model 2: Visit Patterns ──
    if args.model in ['pattern', 'all']:
        print("\n" + "─"*60)
        print("🗓️  MODEL 2: Visit Pattern Classifier")
        print("─"*60)
        
        if not visits_df.empty and len(visits_df) > 10:
            pattern_model = VisitPatternModel()
            X, y = pattern_model.prepare_features(visits_df)
            
            if len(X) > 0 and len(np.unique(y)) > 1:
                results = pattern_model.train(X, y)
                # Log best result
                best = max(results, key=lambda x: results[x].get('f1', 0))
                tracker.log(
                    model_name=best, model_type='classifier',
                    params={'model': best},
                    metrics={'f1': results[best]['f1'], 'accuracy': results[best]['accuracy']},
                    notes="Visit pattern classification"
                )
            else:
                print("  ⚠️  Not enough data for classification")
        else:
            print("  ⚠️  Need more visit history for pattern analysis")
    
    # ── Model 3: Route Efficiency ──
    if args.model in ['route', 'all']:
        print("\n" + "─"*60)
        print("🗺️  MODEL 3: Route Efficiency")
        print("─"*60)
        route_model = RouteEfficiencyModel()
        route_model.cluster_routes(visits_df)
        # TODO: Implement route efficiency prediction
    
    # ── Model 4: Nutrition Optimizer ──
    if args.model in ['nutrition', 'all']:
        print("\n" + "─"*60)
        print("🥗 MODEL 4: Nutrition Optimizer")
        print("─"*60)
        # TODO: Implement nutrition optimization
        print("  TODO: Implement nutrition optimization in NutritionOptimizer")
    
    # ── Model 5: Shopping Recommender ──
    if args.model in ['recommend', 'all']:
        print("\n" + "─"*60)
        print("🛒 MODEL 5: Shopping Recommender")
        print("─"*60)
        recommender = ShoppingRecommender()
        recommender.mine_association_rules(visits_df)
        # TODO: Implement shopping recommendations
    
    print("\n" + "✅"*35)
    print("  Training Complete!")
    print("✅"*35)
    tracker.print_leaderboard()


# ── CLI INTERFACE ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='GrihZen ML Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python grihzen_train_model.py --model all
  python grihzen_train_model.py --model demand --user_id 1
  python grihzen_train_model.py --model pattern
  python grihzen_train_model.py --model route --evaluate
  python grihzen_train_model.py --model all --export

Models:
  demand    → Demand forecasting (when items run out)
  pattern   → Visit pattern classifier (which day/store)
  route     → Route efficiency learner
  nutrition → Nutrition optimizer (meal planning)
  recommend → Shopping recommender
  all       → Train all models
        """
    )
    parser.add_argument('--model', type=str, default='all',
                        choices=['demand','pattern','route','nutrition','recommend','all'],
                        help='Which model to train')
    parser.add_argument('--user_id', type=int, default=None,
                        help='Train on specific user data (default: all users)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation only (no training)')
    parser.add_argument('--export', action='store_true',
                        help='Export trained models to files')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data even if real data exists')
    
    args = parser.parse_args()
    run_training_pipeline(args)
