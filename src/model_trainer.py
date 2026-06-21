import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

RANDOM_SEED = 42
TFIDF_MAX_FEATURES = 3000


class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrap MLPClassifier to encode string labels."""

    def __init__(self):
        self.encoder = LabelEncoder()
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1,
        )

    def fit(self, X, y):
        y_enc = self.encoder.fit_transform(y)
        self.model.fit(X, y_enc)
        self.classes_ = self.encoder.classes_
        return self

    def predict(self, X):
        return self.encoder.inverse_transform(self.model.predict(X))


class MLPRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            random_state=RANDOM_SEED,
            early_stopping=False,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.model.predict(X)


def build_feature_transformer(X_train, include_text=True):
    """Build ColumnTransformer with optional TF-IDF on issue_description."""
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    if include_text and 'issue_description' in categorical_features:
        categorical_features.remove('issue_description')
    if include_text and 'issue_description' in numeric_features:
        numeric_features.remove('issue_description')

    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num', numeric_pipe, numeric_features))

    if categorical_features:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=30)),
        ])
        transformers.append(('cat', cat_pipe, categorical_features))

    if include_text and 'issue_description' in X_train.columns:
        text_pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')),
        ])
        transformers.append(('text', text_pipe, 'issue_description'))

    remainder = 'drop' if transformers else 'passthrough'
    return ColumnTransformer(transformers=transformers, remainder=remainder)


def get_classification_pipelines(X_train):
    """Return dict of named sklearn pipelines for priority classification."""
    preprocessor = build_feature_transformer(X_train, include_text=True)
    return {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)),
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', build_feature_transformer(X_train, include_text=True)),
            ('classifier', RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
            )),
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', build_feature_transformer(X_train, include_text=True)),
            ('classifier', GradientBoostingClassifier(random_state=RANDOM_SEED)),
        ]),
        'Neural Network (MLP)': Pipeline([
            ('preprocessor', build_feature_transformer(X_train, include_text=True)),
            ('classifier', MLPClassifierWrapper()),
        ]),
    }


def get_regression_pipelines(X_train):
    """Return dict of named sklearn pipelines for resolution time regression."""
    return {
        'Linear Regression': Pipeline([
            ('preprocessor', build_feature_transformer(X_train, include_text=True)),
            ('regressor', LinearRegression()),
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', build_feature_transformer(X_train, include_text=True)),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        'Neural Network (MLP)': Pipeline([
            ('preprocessor', build_feature_transformer(X_train, include_text=True)),
            ('regressor', MLPRegressorWrapper()),
        ]),
    }


def get_satisfaction_pipelines(X_train):
    """Return dict of named sklearn pipelines for satisfaction classification."""
    return {
        'Logistic Regression': Pipeline([
            ('preprocessor', build_feature_transformer(X_train, include_text=True)),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)),
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', build_feature_transformer(X_train, include_text=True)),
            ('classifier', RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
            )),
        ]),
        'Neural Network (MLP)': Pipeline([
            ('preprocessor', build_feature_transformer(X_train, include_text=True)),
            ('classifier', MLPClassifierWrapper()),
        ]),
    }
