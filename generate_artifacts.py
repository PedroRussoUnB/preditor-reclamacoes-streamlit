# Este é um script para ser executado UMA VEZ via CMD para gerar os arquivos .pkl.
# Ele NÃO deve conter nenhum código de interface do Streamlit (st.algo).

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve
import pickle
import warnings

warnings.filterwarnings('ignore')

# --- Definições e Classes do Projeto ---
class ProjectConfig:
    TARGET_VARIABLE = 'Complain'
    TEST_SIZE_RATIO = 0.3
    RANDOM_STATE_SEED = 42

class ManualSelector:
    def __init__(self, selected_indices):
        self.support_ = None
        self.selected_indices_ = selected_indices
    def fit(self, X):
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[self.selected_indices_] = True
    def transform(self, X):
        if self.support_ is None: raise RuntimeError("O método 'fit' deve ser chamado antes do 'transform'.")
        return X[:, self.support_]

# --- Funções de Preparação de Dados ---
def execute_feature_engineering(_df):
    df = _df.copy()
    if 'Income' in df.columns and df['Income'].isnull().any():
        df['Income'].fillna(df['Income'].median(), inplace=True)
    current_year = datetime.now().year
    df['Age'] = current_year - df['Year_Birth']
    df['Customer_Lifetime_Days'] = (datetime.now() - pd.to_datetime(df['Dt_Customer'], dayfirst=True)).dt.days
    df['Children_Total'] = df['Kidhome'] + df['Teenhome']
    mnt_cols = [col for col in df.columns if 'Mnt' in col]
    df['Total_Spent'] = df[mnt_cols].sum(axis=1)
    purchase_cols = [col for col in df.columns if 'Num' in col and 'Purchases' in col]
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    df['Luxury_Purchase_Ratio'] = (df['MntWines'] + df['MntGoldProds']) / (df['Total_Spent'] + 1)
    cmp_cols = [col for col in df.columns if 'AcceptedCmp' in col] + ['Response']
    df['Marketing_Engagements'] = df[cmp_cols].sum(axis=1)
    df['Marital_Status'] = df['Marital_Status'].replace({'Married': 'In_Relationship', 'Together': 'In_Relationship', 'Alone': 'Single', 'Single': 'Single', 'Divorced': 'Single', 'Widow': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'})
    df['Education'] = df['Education'].replace({'2n Cycle': 'Master'})
    cols_to_drop = ['ID', 'Year_Birth', 'Dt_Customer', 'Kidhome', 'Teenhome', 'Z_CostContact', 'Z_Revenue'] + mnt_cols + cmp_cols
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.fillna(0, inplace=True)
    return df

def prepare_data_for_modeling(_df, target, test_size, random_state):
    X = _df.drop(columns=[target])
    y = _df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features), ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)], remainder='passthrough')
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    processed_feature_names = numeric_features.tolist() + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    return {'X_train_orig': X_train_processed, 'y_train_orig': y_train, 'X_train_resampled': X_train_resampled, 'y_train_resampled': y_train_resampled, 'X_test': X_test_processed, 'y_test': y_test, 'preprocessor': preprocessor, 'processed_feature_names': processed_feature_names}

# --- Funções de Geração de Artefatos ---
def generate_selection_artifacts(modeling_data):
    print("Gerando artefatos da seleção de features...")
    X_train, y_train, feature_names = modeling_data['X_train_orig'], modeling_data['y_train_orig'], modeling_data['processed_feature_names']
    estimator = LGBMClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED, n_estimators=50, n_jobs=-1, verbose=-1)
    estimator.fit(X_train, y_train)
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': estimator.feature_importances_}).sort_values(by='Importance', ascending=False)
    total_importance = importances_df['Importance'].sum()
    importances_df['Cumulative_Importance'] = importances_df['Importance'].cumsum()
    importances_df['Cumulative_Percentage'] = importances_df['Cumulative_Importance'] / total_importance
    optimal_n_features = (importances_df['Cumulative_Percentage'] <= 0.95).sum() + 1
    optimal_n_features = min(optimal_n_features, len(importances_df))
    top_features = importances_df.head(optimal_n_features)
    selected_features_names = top_features['Feature'].tolist()
    selected_indices = [feature_names.index(f) for f in selected_features_names]
    selector_object = ManualSelector(selected_indices)
    selector_object.fit(X_train)
    artifacts = {'selector_object': selector_object, 'optimal_n_features': optimal_n_features, 'selected_feature_names': selected_features_names, 'feature_ranking_df': importances_df}
    with open('selection_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("-> 'selection_artifacts.pkl' gerado com sucesso!")
    return artifacts

def generate_baseline_artifacts(modeling_data, selection_artifacts):
    print("\nGerando artefatos do baseline (isso pode demorar alguns minutos)...")
    X_train, y_train, X_test, y_test = modeling_data['X_train_resampled'], modeling_data['y_train_resampled'], modeling_data['X_test'], modeling_data['y_test']
    selector = selection_artifacts['selector_object']
    X_train_final = selector.transform(X_train)
    X_test_final = selector.transform(X_test)
    models_to_test = {
        "LightGBM": LGBMClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED, verbose=-1),
        "XGBoost": XGBClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED, use_label_encoder=False, eval_metric='logloss'),
        "Random Forest": RandomForestClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED),
        "SVC": SVC(probability=True, random_state=ProjectConfig.RANDOM_STATE_SEED),
        "AdaBoost": AdaBoostClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED),
        "KNN": KNeighborsClassifier()
    }
    baseline_results = {}
    for name, model in models_to_test.items():
        print(f"Treinando {name}...")
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        y_proba = model.predict_proba(X_test_final)[:, 1]
        baseline_results[name] = {
            'model_object': model,
            'metrics': {'AUC': roc_auc_score(y_test, y_proba), 'F1-Score': f1_score(y_test, y_pred), 'Recall': recall_score(y_test, y_pred), 'Precisão': precision_score(y_test, y_pred)},
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'full_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_curve_data': roc_curve(y_test, y_proba)
        }
    with open('baseline_results.pkl', 'wb') as f:
        pickle.dump(baseline_results, f)
    print("-> 'baseline_results.pkl' gerado com sucesso!")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    print("Iniciando processo de geração de artefatos pré-calculados...")
    df_raw = pd.read_csv('marketing_campaign.csv', sep='\t')
    df_processed = execute_feature_engineering(df_raw)
    modeling_data = prepare_data_for_modeling(df_processed, ProjectConfig.TARGET_VARIABLE, ProjectConfig.TEST_SIZE_RATIO, ProjectConfig.RANDOM_STATE_SEED)
    selection_artifacts = generate_selection_artifacts(modeling_data)
    generate_baseline_artifacts(modeling_data, selection_artifacts)
    print("\nProcesso concluído! Todos os artefatos foram gerados.")