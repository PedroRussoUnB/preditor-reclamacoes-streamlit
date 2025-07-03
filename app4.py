import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score, precision_score, recall_score, matthews_corrcoef
from scipy.stats import pearsonr, chi2_contingency
import shap
from fpdf import FPDF

st.set_page_config(
    page_title="Plataforma Preditiva de Reclamações",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

class ProjectConfig:
    TARGET_VARIABLE = 'Complain'
    TEST_SIZE_RATIO = 0.3
    RANDOM_STATE_SEED = 42
    N_SPLITS_KFOLD = 5
    RFE_CV_SCORING = 'roc_auc'
    PRIMARY_COLOR = "#00A9FF"
    SECONDARY_COLOR = "#FF6347"
    BACKGROUND_COLOR = "#0A0A0A"
    TEXT_COLOR = "#EAEAEA"
    SUCCESS_COLOR = "#32CD32"
    GRID_COLOR = "#444444"

    @staticmethod
    def get_plotly_template():
        template = go.layout.Template()
        template.layout.paper_bgcolor = ProjectConfig.BACKGROUND_COLOR
        template.layout.plot_bgcolor = "#1E1E1E"
        template.layout.font = dict(color=ProjectConfig.TEXT_COLOR)
        template.layout.xaxis = dict(gridcolor=ProjectConfig.GRID_COLOR, linecolor=ProjectConfig.GRID_COLOR, showgrid=True, zeroline=False)
        template.layout.yaxis = dict(gridcolor=ProjectConfig.GRID_COLOR, linecolor=ProjectConfig.GRID_COLOR, showgrid=True, zeroline=False)
        template.layout.title = dict(x=0.5, font=dict(size=20))
        return template

def initialize_session_state():
    session_keys = {
        'app_stage': 'initialization',
        'data_loaded': False,
        'data_processed': False,
        'models_trained': False,
        'final_model_selected': False,
        'raw_df': None,
        'processed_df': None,
        'artifacts': {}
    }
    for key, default_value in session_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()
px.defaults.template = ProjectConfig.get_plotly_template()

@st.cache_data(show_spinner="Carregando e validando arquivo 'marketing_campaign.csv'...")
def load_data_from_disk():
    """Lê o arquivo CSV diretamente do disco e o retorna como DataFrame."""
    try:
        df = pd.read_csv('marketing_campaign.csv', sep='\t')
        return df, "Arquivo 'marketing_campaign.csv' carregado com sucesso!"
    except FileNotFoundError:
        st.error("ERRO CRÍTICO: O arquivo 'marketing_campaign.csv' não foi encontrado. Por favor, certifique-se de que ele está na mesma pasta que o seu script `app.py`.")
        return None, "Arquivo não encontrado."
    except Exception as e:
        st.error(f"Erro inesperado ao ler o arquivo: {e}")
        return None, "Falha na leitura do arquivo."

@st.cache_data(show_spinner="Executando profiling detalhado dos dados...")
def perform_data_profiling(_df):
    """Executa um profiling completo no DataFrame, retornando um resumo detalhado."""
    # Renomeamos as chaves do dicionário para serem mais amigáveis ao usuário
    profile_summary = {
        'Visão Geral do Dataset': {
            'Total de Clientes (Linhas)': _df.shape[0],
            'Total de Características (Colunas)': _df.shape[1],
            'Dados Faltando (Células)': _df.isnull().sum().sum(),
            'Percentual de Dados Faltando': f"{(_df.isnull().sum().sum() / _df.size) * 100:.2f}%",
            'Registros Duplicados': _df.duplicated().sum(),
            'Uso de Memória (MB)': f"{_df.memory_usage(deep=True).sum() / 1024**2:.2f}"
        },
        'detalhes_variaveis': []
    }
    for col in _df.columns:
        series = _df[col]
        # Também renomeamos as chaves aqui
        col_info = {
            'Característica': col,
            'Tipo': str(series.dtype),
            'Dados Faltando': series.isnull().sum(),
            'Valores Distintos': series.nunique()
        }
        if pd.api.types.is_numeric_dtype(series):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))]
            col_info['Valores Atípicos (Outliers)'] = len(outliers)
            col_info['Média'] = series.mean()
            col_info['Mediana'] = series.median()
            col_info['Desvio Padrão'] = series.std()
        profile_summary['detalhes_variaveis'].append(col_info)
    
    return profile_summary

@st.cache_data(show_spinner="Aplicando engenharia e transformação de features...")
def execute_feature_engineering(_df):
    """Executa um pipeline completo de limpeza, criação e transformação de variáveis."""
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

    df['Marital_Status'] = df['Marital_Status'].replace({
        'Married': 'In_Relationship', 'Together': 'In_Relationship', 'Alone': 'Single',
        'Single': 'Single', 'Divorced': 'Single', 'Widow': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'
    })
    df['Education'] = df['Education'].replace({'2n Cycle': 'Master'})
    
    cols_to_drop = [
        'ID', 'Year_Birth', 'Dt_Customer', 'Kidhome', 'Teenhome', 'Z_CostContact', 'Z_Revenue'
    ] + mnt_cols + cmp_cols
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    df.fillna(0, inplace=True)
    return df

def display_home_page():
    """Renderiza a página inicial/de boas-vindas do dashboard."""
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Plataforma Preditiva de Reclamações")
        st.subheader("Transformando Dados em Decisões Estratégicas de Retenção")
    with col2:
        st.image("https://www.gstatic.com/devrel-devsite/prod/v22d5bf23537453457d3952b9015c9ad5e229c66a7b204ada65f02540915f0119/developers/images/lockup-new.svg", width=200)

    st.markdown("---")
    
    st.markdown("""
    ### Bem-vindo à Ferramenta Analítica de Modelagem Supervisionada.

    Esta plataforma interativa foi projetada para cumprir um objetivo de negócio claro: **prever quais clientes possuem a maior probabilidade de realizar uma reclamação**.
    
    **O que você pode fazer aqui?**
    - **Analisar o Dataset:** Carregue e audite a qualidade dos dados dos clientes.
    - **Explorar os Dados:** Visualize de forma interativa a relação entre as variáveis.
    - **Construir e Avaliar Modelos:** Acompanhe o treinamento de múltiplos algoritmos e compare sua performance.
    - **Simular e Interpretar:** Utilize o modelo final para simular o impacto de campanhas de retenção e entender os fatores que impulsionam as previsões.

    Utilize o **menu de navegação na barra lateral esquerda** para explorar as diferentes etapas do projeto.
    """)
    
    st.markdown("---")
    
    with st.expander("Sobre este Projeto"):
        st.write("""
        Esta aplicação é o resultado prático da tarefa de Modelos Supervisionados, onde o desafio é atuar como um cientista de dados para uma empresa de varejo. O foco não está apenas em treinar modelos, mas em selecionar variáveis, testar abordagens, entender os algoritros e, o mais importante, transformar os resultados em ações de negócio concretas.
        
        **Tecnologias Utilizadas:** Streamlit, Pandas, Scikit-learn, Plotly, SHAP.
        """)

    st.info("Para começar, navegue para a página **'Análise do Dataset'** no menu lateral.", icon="🚀")

def display_dataset_page():
    """Renderiza a página de carregamento, profiling e engenharia de features."""
    
    st.header("Análise do Dataset: Carga e Auditoria de Qualidade")
    st.markdown("""
    O primeiro passo é carregar e realizar uma auditoria completa nos dados. Isso nos permite entender a estrutura, identificar problemas
    como valores ausentes ou inconsistentes, e ter uma visão geral da qualidade dos dados brutos antes de qualquer transformação.
    """)

    # Carregar os dados automaticamente do disco
    if not st.session_state.data_loaded:
        raw_df, message = load_data_from_disk()
        if raw_df is not None:
            st.session_state.raw_df = raw_df
            st.session_state.data_loaded = True
            st.success(message)
        else:
            st.error(message)
            return # Para a execução se o arquivo não for encontrado

    if st.session_state.data_loaded:
        raw_df = st.session_state.raw_df
        # Exibir relatório de profiling
        profile_results = perform_data_profiling(raw_df)
        
        st.subheader("Visão Geral do Dataset")
        overview = profile_results['Visão Geral do Dataset']
        cols = st.columns(len(overview))
        for i, (key, value) in enumerate(overview.items()):
            cols[i].metric(key, value)
        
        with st.expander("Visualizar Relatório Detalhado por Variável", expanded=False):
            profile_df = pd.DataFrame(profile_results['detalhes_variaveis']).set_index('Característica')
            st.dataframe(profile_df)
        
        with st.expander("Visualizar Amostra dos Dados Brutos"):
            st.dataframe(raw_df.sample(5))

        st.markdown("---")
        
        st.subheader("Preparação e Enriquecimento dos Dados")
        st.markdown("""
        Após a auditoria, executamos um pipeline de engenharia de features para limpar, transformar e enriquecer o dataset,
        criando novas variáveis que ajudarão os modelos a aprender melhor os padrões de comportamento dos clientes.
        """)
        
        if st.button("Executar Engenharia de Features", type="primary"):
            processed_df = execute_feature_engineering(st.session_state.raw_df)
            st.session_state.processed_df = processed_df
            st.session_state.data_processed = True
            st.success("Pipeline de engenharia de features executado com sucesso!")

        if st.session_state.data_processed:
            st.subheader("Amostra dos Dados Após Transformação")
            st.markdown("Abaixo estão os dados prontos para a fase de análise exploratória e modelagem.")
            st.dataframe(st.session_state.processed_df.sample(5))
            st.info("Os dados foram processados. Você já pode navegar para a 'Análise Exploratória' no menu lateral.", icon="📊")

def display_eda_page():
    st.header("Análise Exploratória Interativa (EDA)")
    st.markdown("""
    Nesta seção, mergulhamos nos dados para descobrir padrões, identificar anomalias e formular hipóteses. Utilize as abas abaixo para navegar entre os diferentes níveis de análise,
    desde o estudo de variáveis individuais até a visualização de interações complexas entre múltiplos fatores.
    """)

    if not st.session_state.data_processed or st.session_state.processed_df is None:
        st.warning("Os dados precisam ser processados na página 'Análise do Dataset' para acessar a EDA.")
        return

    df = st.session_state.processed_df

    # Estrutura de abas para uma experiência de usuário organizada
    tab_uni, tab_bi, tab_multi = st.tabs([
        "📊 Análise Univariada", 
        "🔗 Análise Bivariada", 
        "🔮 Análise Multivariada"
    ])

    with tab_uni:
        # A função que renderiza o conteúdo desta aba será definida no próximo bloco
        render_univariate_analysis_tab(df)

    with tab_bi:
        # A função que renderiza o conteúdo desta aba será definida no bloco 7
        render_bivariate_analysis_tab(df)

    with tab_multi:
        # A função que renderiza o conteúdo desta aba será definida no bloco 8
        render_multivariate_analysis_tab(df)

@st.cache_data
def calculate_descriptive_stats(series):
    """Calcula um dicionário de estatísticas descritivas para uma variável."""
    if pd.api.types.is_numeric_dtype(series):
        return {
            'Média': series.mean(), 'Mediana': series.median(), 'Desvio Padrão': series.std(),
            'Variância': series.var(), 'Mínimo': series.min(), 'Máximo': series.max(),
            '25º Percentil': series.quantile(0.25), '75º Percentil': series.quantile(0.75),
            'Assimetria (Skew)': series.skew(), 'Curtose (Kurtosis)': series.kurt(),
            'Contagem': series.count(), 'Valores Únicos': series.nunique()
        }
    else:
        return {
            'Contagem': series.count(), 'Valores Únicos': series.nunique(),
            'Moda (Mais Frequente)': series.mode().iloc[0] if not series.mode().empty else 'N/A',
            'Frequência da Moda': series.value_counts().iloc[0] if not series.value_counts().empty else 0
        }

def render_univariate_analysis_tab(df):
    st.subheader("Análise de Variáveis Individuais")
    st.markdown("Selecione uma variável para visualizar sua distribuição e principais métricas estatísticas.")

    # Widget de seleção para o usuário
    variable_to_analyze = st.selectbox(
        "Selecione a variável de interesse:",
        options=df.columns,
        index=list(df.columns).index('Total_Spent') if 'Total_Spent' in df.columns else 0
    )

    if variable_to_analyze:
        selected_series = df[variable_to_analyze]
        
        # Layout principal com duas colunas
        stats_col, plot_col = st.columns([1, 2])
        
        with stats_col:
            st.markdown(f"#### Métricas para **{variable_to_analyze}**")
            stats_dict = calculate_descriptive_stats(selected_series)
            stats_df = pd.DataFrame(stats_dict.items(), columns=['Métrica', 'Valor'])
            st.dataframe(stats_df, use_container_width=True)

        with plot_col:
            if pd.api.types.is_numeric_dtype(selected_series):
                st.markdown(f"#### Distribuição de **{variable_to_analyze}**")
                
                # Gráfico com subplots para uma visão completa
                fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05,
                                    subplot_titles=("Histograma e Curva de Densidade", "Box Plot para Detecção de Outliers"))
                
                fig.add_trace(go.Histogram(x=selected_series, name='Histograma', histnorm='probability density'), row=1, col=1)
                fig.add_trace(go.Box(x=selected_series, name='Box Plot'), row=2, col=1)

                fig.update_layout(showlegend=False, height=500, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            else: 
                st.markdown(f"#### Frequência de **{variable_to_analyze}**")
                
                counts = selected_series.value_counts()
                fig = px.bar(
                    counts, 
                    x=counts.index, 
                    y=counts.values,
                    title=f"Contagem de Categorias em {variable_to_analyze}",
                    labels={'x': variable_to_analyze, 'y': 'Contagem'},
                    text_auto=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

def render_bivariate_analysis_tab(df):
    st.subheader("Análise de Relação entre Pares de Variáveis")
    st.markdown("Selecione duas variáveis para visualizar e quantificar a relação entre elas.")

    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Selecione a primeira variável:", df.columns, index=0, key="bivar_1")
    with col2:
        var2 = st.selectbox("Selecione a segunda variável:", df.columns, index=1, key="bivar_2")

    if var1 and var2 and var1 != var2:
        is_var1_numeric = pd.api.types.is_numeric_dtype(df[var1])
        is_var2_numeric = pd.api.types.is_numeric_dtype(df[var2])

        if is_var1_numeric and is_var2_numeric:
            st.markdown(f"#### Análise de Correlação: **{var1}** vs. **{var2}**")
            
            corr, p_value = pearsonr(df[var1], df[var2])
            
            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x=var1, y=var2,
                trendline="ols", trendline_color_override=ProjectConfig.SECONDARY_COLOR,
                title=f"Dispersão e Linha de Tendência"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric(label="Coeficiente de Correlação de Pearson", value=f"{corr:.3f}")
            st.info(f"O p-valor para este teste de correlação é **{p_value:.4f}**. Valores de p < 0.05 geralmente indicam uma correlação estatisticamente significante.", icon="🔬")

        elif not is_var1_numeric and not is_var2_numeric:
            st.markdown(f"#### Análise de Associação: **{var1}** vs. **{var2}**")
            
            contingency_table = pd.crosstab(df[var1], df[var2])
            
            fig = px.bar(
                contingency_table,
                barmode='group',
                title=f"Contagem Agrupada"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            chi2, p, _, _ = chi2_contingency(contingency_table)
            st.metric(label="Estatística Qui-Quadrado (χ²)", value=f"{chi2:.2f}")
            st.info(f"O p-valor para o teste de independência é **{p:.4f}**. Valores de p < 0.05 sugerem que as variáveis são dependentes (associadas).", icon="🔬")

        else:
            numeric_var = var1 if is_var1_numeric else var2
            categorical_var = var2 if is_var1_numeric else var1
            
            st.markdown(f"#### Comparação de Distribuições: **{numeric_var}** por **{categorical_var}**")

            fig = px.violin(
                df, x=categorical_var, y=numeric_var,
                color=categorical_var, box=True, points="all",
                title=f"Distribuição de {numeric_var} através das categorias de {categorical_var}"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver resumo estatístico detalhado por categoria"):
                st.dataframe(df.groupby(categorical_var)[numeric_var].describe().transpose())
    else:
        st.warning("Por favor, selecione duas variáveis diferentes para a análise.")

@st.cache_data(show_spinner="Calculando projeção PCA para visualização...")
def get_pca_projection(_df, target_col):
    numeric_cols = _df.select_dtypes(include=np.number).columns.drop(target_col, errors='ignore')
    
    pca_df = _df.copy()
    pca_df[numeric_cols] = StandardScaler().fit_transform(pca_df[numeric_cols])
    
    pca = PCA(n_components=2, random_state=ProjectConfig.RANDOM_STATE_SEED)
    principal_components = pca.fit_transform(pca_df[numeric_cols])
    
    pca_result_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_result_df[target_col] = pca_df[target_col].values
    
    explained_variance = pca.explained_variance_ratio_
    return pca_result_df, explained_variance

def render_multivariate_analysis_tab(df):
    st.subheader("Análise de Múltiplas Variáveis Simultaneamente")
    st.markdown("Explore as interações complexas entre várias features e como elas se relacionam com a variável alvo.")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(ProjectConfig.TARGET_VARIABLE, errors='ignore').tolist()
    
    st.markdown("#### Visualização do Espaço de Features com PCA")
    st.markdown("A Análise de Componentes Principais (PCA) reduz a complexidade dos dados, projetando-os em 2D. O gráfico abaixo nos ajuda a ver se existem agrupamentos naturais de clientes que reclamam (vermelho) vs. os que não reclamam (azul).")

    if st.button("Gerar Gráfico PCA"):
        pca_result_df, explained_variance = get_pca_projection(df, ProjectConfig.TARGET_VARIABLE)
        
        fig_pca = px.scatter(
            pca_result_df, x='PC1', y='PC2',
            color=ProjectConfig.TARGET_VARIABLE,
            color_continuous_scale=[ProjectConfig.PRIMARY_COLOR, ProjectConfig.SECONDARY_COLOR],
            title=f"Projeção PCA 2D do Dataset (Variância Explicada: {sum(explained_variance):.2%})"
        )
        fig_pca.update_layout(height=600)
        st.plotly_chart(fig_pca, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### Scatter Plot 3D Interativo")
    st.markdown("Selecione três variáveis numéricas para criar um gráfico de dispersão 3D. A cor dos pontos representa o status da reclamação do cliente.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_3d = st.selectbox("Selecione o Eixo X:", numeric_cols, index=0, key="x_3d")
    with col2:
        y_3d = st.selectbox("Selecione o Eixo Y:", numeric_cols, index=1, key="y_3d")
    with col3:
        z_3d = st.selectbox("Selecione o Eixo Z:", numeric_cols, index=2, key="z_3d")
    
    if x_3d and y_3d and z_3d:
        fig_3d = px.scatter_3d(
            df.sample(min(2000, len(df))), # Amostra para performance
            x=x_3d, y=y_3d, z=z_3d,
            color=ProjectConfig.TARGET_VARIABLE,
            color_continuous_scale=[ProjectConfig.PRIMARY_COLOR, ProjectConfig.SECONDARY_COLOR],
            title="Visualização 3D Interativa de Features",
            height=700
        )
        fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
        st.plotly_chart(fig_3d, use_container_width=True)

@st.cache_data(show_spinner="Dividindo dados, processando e aplicando SMOTE...")
def prepare_data_for_modeling(_df, target, test_size, random_state):
    """Executa a divisão, pré-processamento e balanceamento dos dados."""
    
    X = _df.drop(columns=[target])
    y = _df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    processed_feature_names = numeric_features.tolist() + \
                              preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    modeling_data = {
        'X_train_orig': X_train_processed, 'y_train_orig': y_train,
        'X_train_resampled': X_train_resampled, 'y_train_resampled': y_train_resampled,
        'X_test': X_test_processed, 'y_test': y_test,
        'preprocessor': preprocessor, 'processed_feature_names': processed_feature_names,
        'X_train_raw': X_train, 'X_test_raw': X_test
    }
    
    return modeling_data

def render_data_preparation_module(df):
    with st.container(border=True):
        st.subheader("Etapa 1: Preparação do Terreno para a Modelagem")
        st.markdown("""
        **O Quê?** Aqui, preparamos os dados para que os algoritmos de Machine Learning possam "entendê-los" da melhor forma possível. Realizamos três ações cruciais:
        1.  **Divisão Estratificada:** Separamos os dados em um conjunto de **Treino** (para ensinar o modelo) e um de **Teste** (para avaliá-lo de forma imparcial). A estratificação garante que a proporção de clientes que reclamam e não reclamam seja a mesma em ambos os conjuntos, evitando vieses.
        2.  **Pré-processamento:** Padronizamos as variáveis numéricas (para que não tenham escalas discrepantes) e codificamos as variáveis de texto em formato numérico.
        3.  **Balanceamento com SMOTE:** Nosso desafio é que pouquíssimos clientes reclamam (~1%). Se não fizermos nada, o modelo pode ficar "preguiçoso" e prever sempre "não reclama". O **SMOTE** resolve isso criando exemplos sintéticos e realistas de clientes que reclamam no conjunto de treino. É como dar uma lupa para o modelo aprender a fundo as características desse grupo minoritário, mas crucial.
        
        **Por quê?** Esta etapa é a fundação de todo o projeto. Uma preparação inadequada levaria a modelos com performance ruim e conclusões de negócio equivocadas.
        """)
        
        if st.button("Executar Divisão e Balanceamento", type="primary", key="prep_button"):
            modeling_data = prepare_data_for_modeling(
                df, 
                target=ProjectConfig.TARGET_VARIABLE, 
                test_size=ProjectConfig.TEST_SIZE_RATIO, 
                random_state=ProjectConfig.RANDOM_STATE_SEED
            )
            st.session_state['artifacts']['modeling_data'] = modeling_data
            st.session_state.app_stage = 'data_prepared'
            st.success("Dados preparados com sucesso!")
            st.rerun()

    if 'modeling_data' in st.session_state.get('artifacts', {}):
        with st.container(border=True):
            modeling_data = st.session_state['artifacts']['modeling_data']
            st.subheader("Análise Pós-Preparação")
            st.markdown("""
            **O que aconteceu?** Os dados foram divididos e o SMOTE foi aplicado no conjunto de treino.
            - **Métricas:** Veja abaixo a quantidade de clientes em cada conjunto. Note como o conjunto de treino cresceu após o balanceamento.
            - **Gráfico de Dispersão (PCA):** Este gráfico visualiza a "separação" entre clientes que reclamam (vermelho) e os que não reclamam (azul) em um espaço 2D. 
                - *Antes do SMOTE:* A nuvem de pontos vermelhos é minúscula e dispersa, difícil para um modelo aprender.
                - *Depois do SMOTE:* A nuvem vermelha está muito mais densa e definida, criando um padrão claro para os algoritmos.
            
            **Próximo Passo:** Agora que temos dados de alta qualidade, podemos prosseguir para a seleção das variáveis mais importantes.
            """)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Clientes no Treino (Original)", len(modeling_data['y_train_orig']))
            col2.metric("Clientes no Teste", len(modeling_data['y_test']))
            col3.metric("Clientes no Treino (Pós-SMOTE)", len(modeling_data['y_train_resampled']), help="O número aumentou devido aos exemplos sintéticos criados pelo SMOTE.")
            
            with st.expander("Visualizar Efeito do SMOTE (Projeção PCA)"):
                pca_vis = PCA(n_components=2, random_state=ProjectConfig.RANDOM_STATE_SEED)
                X_train_pca_before = pca_vis.fit_transform(modeling_data['X_train_orig'])
                X_train_pca_after = pca_vis.transform(modeling_data['X_train_resampled'])

                df_before = pd.DataFrame(X_train_pca_before, columns=['PC1', 'PC2'])
                df_before['target'] = modeling_data['y_train_orig'].values
                df_after = pd.DataFrame(X_train_pca_after, columns=['PC1', 'PC2'])
                df_after['target'] = modeling_data['y_train_resampled'].values
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Antes do SMOTE", "Depois do SMOTE"))
                fig.add_trace(go.Scatter(x=df_before['PC1'], y=df_before['PC2'], mode='markers', marker=dict(color=df_before['target'], colorscale=[ProjectConfig.PRIMARY_COLOR, ProjectConfig.SECONDARY_COLOR], showscale=False), name='Antes'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_after['PC1'], y=df_after['PC2'], mode='markers', marker=dict(color=df_after['target'], colorscale=[ProjectConfig.PRIMARY_COLOR, ProjectConfig.SECONDARY_COLOR], showscale=False), name='Depois'), row=1, col=2)
                st.plotly_chart(fig, use_container_width=True)

class ManualSelector:
    """Um seletor simulado para manter compatibilidade com o pipeline após a seleção manual de features."""
    def __init__(self, selected_indices):
        self.support_ = None
        self.selected_indices_ = selected_indices
    
    def fit(self, X):
        # Cria a máscara de suporte no momento do fit para ter a dimensão correta
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[self.selected_indices_] = True
    
    def transform(self, X):
        # Garante que o fit foi chamado antes do transform
        if self.support_ is None:
            raise RuntimeError("O método 'fit' deve ser chamado antes do 'transform'.")
        return X[:, self.support_]

from sklearn.feature_selection import f_classif

@st.cache_data(show_spinner="Executando análise estatística para seleção de features...")
def run_feature_selection_by_statistic(_modeling_data):
    X_train, y_train, feature_names = _modeling_data['X_train_orig'], _modeling_data['y_train_orig'], _modeling_data['processed_feature_names']
    
    f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores)
    
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Score_F': f_scores
    }).sort_values(by='Score_F', ascending=False)
    
    total_score = importances_df['Score_F'].sum()
    if total_score == 0:
        return None 
        
    importances_df['Cumulative_Score'] = importances_df['Score_F'].cumsum()
    importances_df['Cumulative_Percentage'] = importances_df['Cumulative_Score'] / total_score
    
    optimal_n_features = (importances_df['Cumulative_Percentage'] <= 0.95).sum() + 1
    optimal_n_features = min(optimal_n_features, len(importances_df))

    top_features = importances_df.head(optimal_n_features)
    selected_features_names = top_features['Feature'].tolist()

    selected_indices = [
        feature_names.index(f) for f in selected_features_names
    ]
    
    selector_object = ManualSelector(selected_indices)
    selector_object.fit(X_train)

    selection_artifacts = {
        'selector_object': selector_object,
        'optimal_n_features': optimal_n_features,
        'selected_feature_names': selected_features_names,
        'feature_ranking_df': importances_df,
    }
    return selection_artifacts

def render_feature_selection_module(modeling_data):
    with st.container(border=True):
        st.subheader("Etapa 2: Foco no que Importa - Seleção Estatística de Features")
        st.markdown("""
        **O Quê?** Para garantir uma experiência instantânea, executamos um processo de seleção de features ultrarrápido baseado em **testes estatísticos (Teste F ANOVA)**. Este método avalia a capacidade de cada variável de distinguir, sozinha, entre clientes que reclamam e os que não reclamam.

        **Por quê?** Esta abordagem é matematicamente direta e não exige o treinamento de um modelo de machine learning, eliminando o consumo intensivo de CPU e memória. Ela seleciona as features com maior poder de separação estatística, garantindo uma performance excelente do aplicativo.
        """)

        if st.button("Executar Seleção Estatística de Features", key="fs_button_stat"):
            selection_artifacts = run_feature_selection_by_statistic(modeling_data)
            if selection_artifacts:
                st.session_state['artifacts']['selection_artifacts'] = selection_artifacts
                st.session_state.app_stage = 'features_selected'
                st.success("Seleção estatística de features concluída!")
                st.rerun()

    if 'selection_artifacts' in st.session_state.get('artifacts', {}):
        with st.container(border=True):
            artifacts = st.session_state['artifacts']['selection_artifacts']
            st.subheader("Análise Pós-Seleção")
            st.markdown("""
            **O que aconteceu?** O sistema calculou a pontuação estatística (Score F) para todas as features e selecionou automaticamente as **{n_feats}** melhores, que juntas explicam 95% da pontuação total.
            - **Gráfico de Relevância:** O gráfico abaixo mostra o ranking. As features no topo são as que possuem a maior relevância estatística para prever uma reclamação.
            - **Lista de Features:** Você pode expandir a seção para ver a lista exata de features que serão usadas na próxima etapa de modelagem.
            """.format(n_feats=artifacts['optimal_n_features']))
            
            ranking_df = artifacts['feature_ranking_df']
            fig = px.bar(
                ranking_df.head(30),
                x='Score_F',
                y='Feature',
                orientation='h',
                title="Ranking de Relevância Estatística das Features (Score F)"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver Lista de Features Selecionadas para a Modelagem"):
                st.dataframe(pd.DataFrame(artifacts['selected_feature_names'], columns=["Feature Selecionada"]), use_container_width=True)

@st.cache_data(show_spinner="Treinando modelos de baseline otimizados para velocidade...")
def train_baseline_models(_modeling_data, _selection_artifacts):
    X_train = _modeling_data['X_train_resampled']
    y_train = _modeling_data['y_train_resampled']
    X_test = _modeling_data['X_test']
    y_test = _modeling_data['y_test']

    selector = _selection_artifacts['selector_object']
    X_train_final = selector.transform(X_train)
    X_test_final = selector.transform(X_test)
    
    models_to_test = {
        "LightGBM": LGBMClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED, verbose=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=ProjectConfig.RANDOM_STATE_SEED, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=ProjectConfig.RANDOM_STATE_SEED),
        "KNN": KNeighborsClassifier(n_jobs=-1)
    }

    baseline_results = {}
    for name, model in models_to_test.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        y_proba = model.predict_proba(X_test_final)[:, 1]
        
        baseline_results[name] = {
            'model_object': model,
            'metrics': {
                'AUC': roc_auc_score(y_test, y_proba),
                'F1-Score': f1_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'Precisão': precision_score(y_test, y_pred)
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'full_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_curve_data': roc_curve(y_test, y_proba)
        }
    return baseline_results

def render_baseline_modeling_module(modeling_data, selection_artifacts):
    with st.container(border=True):
        st.subheader("Etapa 3: A Competição dos Algoritmos (Baseline)")
        st.markdown("""
        **O Quê?** Aqui começa a competição! Para garantir uma experiência de usuário rápida, executamos ao vivo um campeonato com um conjunto selecionado de 4 algoritmos rápidos e eficientes.
        **Por quê?** Esta abordagem nos permite ter uma **linha de base (baseline)** de performance robusta sem um longo tempo de espera. Você obtém uma visão geral dos melhores competidores em poucos segundos para tomar as próximas decisões.
        """)
        
        if st.button("Executar Competição de Modelos", key="train_button"):
            baseline_artifacts = train_baseline_models(modeling_data, selection_artifacts)
            if baseline_artifacts:
                st.session_state['artifacts']['baseline_artifacts'] = baseline_artifacts
                st.session_state.app_stage = 'baselines_trained'
                st.success("Competição de modelos baseline concluída com sucesso!")
                st.rerun()

    if 'baseline_artifacts' in st.session_state.get('artifacts', {}):
        with st.container(border=True):
            artifacts = st.session_state['artifacts']['baseline_artifacts']
            st.subheader("Análise Pós-Treinamento: O Leaderboard")
            st.markdown("""
            **O que aconteceu?** Os modelos foram treinados e avaliados no conjunto de teste. A tabela abaixo é o nosso **Leaderboard de Performance**.
            
            **Como interpretar:**
            - **AUC:** A principal métrica de performance geral. Quanto maior (mais perto de 1.0), melhor o modelo consegue distinguir entre um cliente que vai reclamar e um que não vai.
            - **Recall:** Extremamente importante para o negócio! Indica a porcentagem de clientes que **realmente reclamaram** e que o modelo conseguiu identificar corretamente. Um Recall alto significa que estamos deixando poucos "reclamões" passarem despercebidos.
            - **Precisão:** Dos clientes que o modelo **disse que iriam reclamar**, quantos de fato reclamaram.
            - **F1-Score:** Uma média harmônica entre Precisão e Recall. Útil para um balanço geral.
            
            **Próximo Passo:** Explore o leaderboard (você pode ordenar clicando no nome da coluna) para identificar os modelos campeões. A seguir, vamos mergulhar em uma análise mais profunda de cada um deles e depois otimizar os melhores.
            """)
            
            leaderboard_data = [{'Modelo': name, **res['metrics']} for name, res in artifacts.items()]
            leaderboard_df = pd.DataFrame(leaderboard_data).set_index('Modelo')
            
            sort_by = st.selectbox("Ordenar leaderboard por:", leaderboard_df.columns, index=0)
            sorted_df = leaderboard_df.sort_values(by=sort_by, ascending=False)
            st.dataframe(sorted_df.style.background_gradient(cmap='viridis', subset=[sort_by]).format("{:.4f}"), use_container_width=True)

def render_model_deep_dive_module(baseline_artifacts):
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Etapa 4: Análise Profunda dos Competidores")
        st.markdown("""
        **O Quê?** Agora, damos um "zoom" em cada modelo do leaderboard. Esta seção permite que você investigue a performance de qualquer um dos algoritmos individualmente.
        
        **Por quê?** Entender *como* um modelo acerta e erra é tão importante quanto sua pontuação final. Analisaremos:
        - **Matriz de Confusão:** Um mapa dos acertos e erros. O erro mais crítico para nós é o **Falso Negativo**: quando o modelo prevê "Não Reclama" para um cliente que, na verdade, reclama. Esse é o cliente insatisfeito que não conseguimos identificar.
        - **Curva ROC:** Um gráfico que mostra a habilidade do modelo em separar as classes. Quanto mais a curva se aproxima do canto superior esquerdo, melhor.
        - **Importância de Features:** Revela quais variáveis cada modelo específico considerou mais importantes. Isso nos dá as primeiras pistas sobre o "porquê" por trás das previsões.
        """)
        
        model_to_inspect = st.selectbox(
            "Selecione um modelo do leaderboard para uma análise detalhada:",
            options=baseline_artifacts.keys()
        )
        
        if model_to_inspect:
            model_data = baseline_artifacts[model_to_inspect]
            metrics = model_data['metrics']
            
            st.markdown(f"##### Métricas de Performance para o Modelo **{model_to_inspect}**")
            metric_cols = st.columns(4)
            metric_cols[0].metric("AUC", f"{metrics['AUC']:.4f}")
            metric_cols[1].metric("Recall", f"{metrics['Recall']:.4f}", help="Dos clientes que reclamaram, quantos o modelo pegou?")
            metric_cols[2].metric("Precisão", f"{metrics['Precisão']:.4f}", help="Dos clientes que o modelo disse que reclamariam, quantos realmente reclamaram?")
            metric_cols[3].metric("F1-Score", f"{metrics['F1-Score']:.4f}")

            tab_cm, tab_roc, tab_report, tab_importance = st.tabs(["Matriz de Confusão", "Curva ROC", "Relatório Completo", "Importância de Features"])

            with tab_cm:
                cm = model_data['confusion_matrix']
                fig_cm = px.imshow(
                    cm, text_auto=True, aspect="auto",
                    labels=dict(x="Valores Previstos pelo Modelo", y="Valores Reais"),
                    x=['Não Reclamou', 'Reclamou'], y=['Não Reclamou', 'Reclamou'],
                    title=f"Matriz de Confusão para {model_to_inspect}",
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True)
                st.info(f"O modelo identificou corretamente **{cm[1,1]}** clientes que reclamaram (Verdadeiros Positivos), mas falhou em identificar **{cm[1,0]}** (Falsos Negativos).")


            with tab_roc:
                fpr, tpr, _ = model_data['roc_curve_data']
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {metrics["AUC"]:.3f}', line=dict(width=4)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Performance Aleatória'))
                fig_roc.update_layout(title=f"Curva ROC para {model_to_inspect}", xaxis_title='Taxa de Falsos Positivos', yaxis_title='Taxa de Verdadeiros Positivos')
                st.plotly_chart(fig_roc, use_container_width=True)

            with tab_report:
                st.dataframe(pd.DataFrame(model_data['full_report']).transpose().style.format("{:.3f}"))

            with tab_importance:
                model_object = model_data['model_object']
                if hasattr(model_object, 'feature_importances_'):
                    feature_names = st.session_state['artifacts']['selection_artifacts']['selected_feature_names']
                    importance_df = pd.DataFrame({'Feature': feature_names, 'Importância': model_object.feature_importances_})
                    importance_df = importance_df.sort_values(by='Importância', ascending=True)
                    
                    fig_imp = px.bar(importance_df.tail(15), x='Importância', y='Feature', orientation='h', title=f"Top 15 Features Mais Importantes ({model_to_inspect})")
                    fig_imp.update_layout(height=500)
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info(f"O modelo '{model_to_inspect}' não possui um atributo '.feature_importances_' para análise direta de importância (ex: SVM com kernel não-linear).")

from sklearn.model_selection import RandomizedSearchCV

@st.cache_data
def promote_best_baseline_model(_baseline_artifacts):
    if not _baseline_artifacts:
        return {}

    best_model_name = max(_baseline_artifacts, key=lambda k: _baseline_artifacts[k]['metrics']['AUC'])
    
    best_model_artifacts = _baseline_artifacts[best_model_name]

    promotion_results = {
        best_model_name: {
            'best_estimator': best_model_artifacts['model_object'],
            'best_params': "Modelo de baseline selecionado sem otimização",
            'best_score_cv': best_model_artifacts['metrics']['AUC']
        }
    }
    return promotion_results

def render_hyperparameter_tuning_module(baseline_artifacts):
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Etapa 5: Seleção do Modelo Final")
        st.markdown("""
        **O Quê?** Para garantir a performance e estabilidade do aplicativo, esta etapa agora funciona como uma **promoção automática**. O sistema identifica o modelo com a melhor performance de AUC na etapa de baseline (Etapa 3) e o seleciona para a análise final.

        **Por quê?** A otimização de hiperparâmetros é um processo computacionalmente intensivo e inviável para ser executado ao vivo em um ambiente com recursos limitados. Esta abordagem garante uma experiência de usuário instantânea e seleciona o competidor mais forte já identificado para a fase final.
        """)

        if st.button("Selecionar Melhor Modelo do Baseline e Prosseguir", key="promote_button", type="primary"):
            tuning_artifacts = promote_best_baseline_model(baseline_artifacts)
            st.session_state['artifacts']['tuning_artifacts'] = tuning_artifacts
            st.session_state.app_stage = 'models_tuned'
            st.success("Modelo campeão do baseline promovido com sucesso!")
            st.rerun()

    if 'tuning_artifacts' in st.session_state.get('artifacts', {}):
        with st.container(border=True):
            artifacts = st.session_state['artifacts']['tuning_artifacts']
            if not artifacts:
                st.warning("Não foi possível promover um modelo do baseline.")
            else:
                st.subheader("Modelo Promovido para a Etapa Final")
                model_name = list(artifacts.keys())[0]
                original_auc = artifacts[model_name]['best_score_cv']
                
                st.info(f"O modelo **{model_name}** foi identificado como o de melhor performance no baseline (AUC de **{original_auc:.4f}**) e foi selecionado para a próxima etapa.", icon="🏆")

@st.cache_data(show_spinner="Finalizando modelo campeão e calculando explicações SHAP...")
def finalize_and_explain_model(_tuning_artifacts, _modeling_data, _selection_artifacts):
    if not _tuning_artifacts:
        return None

    best_model_name = max(_tuning_artifacts, key=lambda k: _tuning_artifacts[k]['best_score_cv'])
    final_model = _tuning_artifacts[best_model_name]['best_estimator']
    
    X_test_final = _selection_artifacts['selector_object'].transform(_modeling_data['X_test'])
    y_test = _modeling_data['y_test']
    
    X_test_df = pd.DataFrame(X_test_final, columns=_selection_artifacts['selected_feature_names'])
    
    y_proba_final = final_model.predict_proba(X_test_final)[:, 1]
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision_data, recall_data, thresholds = precision_recall_curve(y_test, y_proba_final)
    
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_df)
    
    final_artifacts = {
        'model_name': best_model_name,
        'model_object': final_model,
        'X_test_df': X_test_df,
        'y_test': y_test,
        'y_proba_final': y_proba_final,
        'precision_recall_curve': (precision_data, recall_data, thresholds),
        'avg_precision_score': average_precision_score(y_test, y_proba_final),
        'shap_explainer': explainer,
        'shap_values': shap_values[1] if isinstance(shap_values, list) else shap_values
    }
    return final_artifacts

def render_final_model_analysis_module(tuning_artifacts, modeling_data, selection_artifacts):
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Etapa 6: O Modelo Campeão e a Decisão de Negócio")
        st.markdown("""
        **O Quê?** Chegamos à fase final da modelagem. Aqui, selecionamos o modelo com a melhor performance após a otimização e o analisamos sob a ótica mais importante: a de negócio.
        
        **Por quê?** Um modelo não toma decisões sozinho. Ele nos dá uma **probabilidade** de um cliente reclamar. Nós, humanos, precisamos definir o **limiar de decisão**: qual o nível de probabilidade que usaremos para classificar um cliente como "de risco" e agir? Essa escolha envolve um trade-off entre Precisão e Recall.
        """)

        if st.button("Analisar Modelo Campeão e Gerar Explicações", key="final_model_button", type='primary'):
            final_artifacts = finalize_and_explain_model(tuning_artifacts, modeling_data, selection_artifacts)
            
            if final_artifacts is None:
                st.error(
                    "**Nenhum modelo foi otimizado.** Isso pode acontecer se os modelos de melhor performance no baseline não estavam na lista para otimização.",
                    icon="⚠️"
                )
            else:
                st.session_state['artifacts']['final_artifacts'] = final_artifacts
                st.session_state.app_stage = 'final_model_selected'
                st.success("Análise do modelo final e explicações SHAP geradas!")
                st.rerun()

    if 'final_artifacts' in st.session_state.get('artifacts', {}):
        with st.container(border=True):
            artifacts = st.session_state['artifacts']['final_artifacts']
            st.subheader(f"Análise de Trade-off para o Modelo Campeão: {artifacts['model_name']}")
            
            st.markdown("#### Análise Interativa do Limiar de Decisão")
            st.markdown("Use o slider abaixo para definir o **limiar de probabilidade**. Observe como as métricas mudam. Sua tarefa é encontrar o balanço ideal para a estratégia da empresa.")
            
            decision_threshold = st.slider("Arraste para ajustar o limiar de decisão de risco:", 0.0, 1.0, st.session_state.get('decision_threshold', 0.5), 0.01)
            st.session_state['decision_threshold'] = decision_threshold
            
            y_pred_adj = (artifacts['y_proba_final'] >= decision_threshold).astype(int)
            y_test = artifacts['y_test']

            adj_recall = recall_score(y_test, y_pred_adj)
            adj_precision = precision_score(y_test, y_pred_adj)
            adj_f1 = f1_score(y_test, y_pred_adj)
            
            cols = st.columns(3)
            cols[0].metric("Recall com Limiar Ajustado", f"{adj_recall:.2%}")
            cols[1].metric("Precisão com Limiar Ajustado", f"{adj_precision:.2%}")
            cols[2].metric("F1-Score com Limiar Ajustado", f"{adj_f1:.2%}")

            st.markdown("#### Curva de Precisão x Recall")
            pr_precision, pr_recall, _ = artifacts['precision_recall_curve']
            fig = px.area(x=pr_recall[1:], y=pr_precision[1:], title=f"Curva de Precisão-Recall (Área = {artifacts['avg_precision_score']:.3f})", labels=dict(x="Recall (Capacidade de encontrar quem reclama)", y="Precisão (Assertividade das previsões de risco)"))
            fig.add_shape(type='line', x0=adj_recall, y0=0, x1=adj_recall, y1=adj_precision, line=dict(color='red', dash='dash'))
            fig.add_shape(type='line', x0=0, y0=adj_precision, x1=adj_recall, y1=adj_precision, line=dict(color='red', dash='dash'))
            fig.add_annotation(x=adj_recall, y=adj_precision, text=f"Ponto Atual ({adj_recall:.2f}, {adj_precision:.2f})", showarrow=True)
            fig.update_yaxes(range=[0, 1.05])
            fig.update_xaxes(range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

            st.info("""
            **Conclusão e Próximo Passo:** Agora que você pode definir uma estratégia de negócio (o limiar), navegue para a página **"Análise Avançada e de Negócio"**. Lá, usaremos este modelo final e o limiar escolhido para entender o comportamento do modelo e simular o impacto financeiro de uma campanha.
            """, icon="💡")

def display_advanced_analysis_page():
    st.header("Análise Avançada e de Negócio", divider='rainbow')

    if 'final_artifacts' not in st.session_state.get('artifacts', {}):
        st.error("⚠️ Por favor, complete a Etapa 6 na aba 'Modelagem e Avaliação' primeiro para gerar os artefatos do modelo final.", icon="🚨")
        st.warning("É necessário clicar no botão 'Analisar Modelo Campeão e Gerar Explicações' para prosseguir.")
        return
        
    final_artifacts = st.session_state.artifacts['final_artifacts']
    
    tab_xai, tab_roi, tab_export = st.tabs(["🤖 Interpretabilidade do Modelo (XAI)", "📈 Simulação de ROI", "📤 Exportar Resultados"])
    
    with tab_xai:
        render_global_xai_module(final_artifacts)
        render_local_xai_module(final_artifacts, st.session_state.artifacts['modeling_data'], st.session_state.artifacts['selection_artifacts'])
        
    with tab_roi:
        render_business_impact_module(final_artifacts)
        
    with tab_export:
        render_export_module(final_artifacts, st.session_state.artifacts['selection_artifacts'], st.session_state['processed_df'])

def render_global_xai_module(final_artifacts):
    with st.container(border=True):
        st.subheader("Análise de Interpretabilidade Global (XAI com SHAP)")
        st.markdown("Aqui, abrimos a 'caixa-preta' do modelo para entender quais fatores ele considera mais importantes em suas decisões, de forma geral.")
        
        X_test_df = final_artifacts['X_test_df']
        shap_values = final_artifacts['shap_values']

        st.markdown("#### Importância Geral das Features (SHAP Bar Plot)")
        st.markdown("Este gráfico ranqueia as features pelo seu impacto médio absoluto nas previsões.")
        fig_bar, ax_bar = plt.subplots()
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
        st.pyplot(fig_bar)
        
        st.markdown("#### Impacto e Distribuição das Features (SHAP Beeswarm Plot)")
        st.markdown("""
        Este gráfico mostra o impacto de cada feature para cada cliente da amostra.
        - **Eixo X:** Valor SHAP (impacto na previsão). Valores positivos aumentam o risco.
        - **Cor:** Valor da feature (vermelho = alto, azul = baixo).
        """)
        fig_beeswarm, ax_beeswarm = plt.subplots()
        shap.summary_plot(shap_values, X_test_df, plot_type='dot', show=False)
        st.pyplot(fig_beeswarm)

def render_local_xai_module(final_artifacts, modeling_data, selection_artifacts):
    with st.container(border=True):
        st.subheader("Análise de Previsão Individual (Interpretabilidade Local)")
        st.markdown("""
        Entenda o **porquê** de uma previsão para um cliente específico. Crie um perfil de cliente abaixo para gerar um score de risco e um 
        **Force Plot** do SHAP, que explica quais fatores mais influenciaram a decisão do modelo para este caso.
        """)
        
        X_train_orig = modeling_data['X_train_raw']

        with st.form("local_prediction_form"):
            st.markdown("##### Simulador de Perfil de Cliente")
            
            form_cols = st.columns(3)
            input_values = {}
            
            all_numeric_features = [
                'Total_Spent', 'Recency', 'Income', 'Customer_Lifetime_Days', 
                'Age', 'Marketing_Engagements', 'Luxury_Purchase_Ratio', 
                'Total_Purchases', 'Children_Total'
            ]
            
            for i, feature in enumerate(all_numeric_features):
                with form_cols[i % 3]:
                    if feature in X_train_orig.columns:
                        series = X_train_orig[feature]
                        min_val, max_val, mean_val = series.min(), series.max(), series.mean()
                        
                        if series.dtype == 'int64':
                            input_values[feature] = st.slider(f"Valor para '{feature}'", int(min_val), int(max_val), int(mean_val))
                        else:
                            input_values[feature] = st.number_input(f"Valor para '{feature}'", float(min_val), float(max_val), float(mean_val))
            
            with form_cols[len(all_numeric_features) % 3]:
                 input_values['Education'] = st.selectbox("Educação", X_train_orig['Education'].unique())
            with form_cols[(len(all_numeric_features) + 1) % 3]:
                 input_values['Marital_Status'] = st.selectbox("Estado Civil", X_train_orig['Marital_Status'].unique())

            submit_button = st.form_submit_button("Analisar Previsão para este Perfil", type="primary")

        if submit_button:
            with st.spinner("Calculando previsão e explicação SHAP local..."):
                full_input_df = X_train_orig.iloc[0:1].copy()
                for feature, value in input_values.items():
                    if feature in full_input_df.columns:
                        full_input_df[feature] = value
                
                preprocessor = modeling_data['preprocessor']
                selector = selection_artifacts['selector_object']
                model = final_artifacts['model_object']
                explainer = final_artifacts['shap_explainer']

                input_processed = preprocessor.transform(full_input_df)
                input_final = selector.transform(input_processed)
                
                prediction_proba = model.predict_proba(input_final)[0][1]
                st.metric("Probabilidade de Reclamação para este Cliente", f"{prediction_proba:.2%}")

                st.markdown("##### Explicação Visual da Previsão (SHAP Force Plot)")
                
                shap_output = explainer.shap_values(input_final)

                if isinstance(shap_output, list):
                    shap_values_for_plot = shap_output[1][0]
                else:
                    shap_values_for_plot = shap_output[0]
                
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    base_value_for_plot = float(explainer.expected_value[1])
                else:
                    base_value_for_plot = float(explainer.expected_value)
                
                force_plot = shap.force_plot(
                    base_value=base_value_for_plot,
                    shap_values=shap_values_for_plot,
                    features=pd.DataFrame(input_final, columns=selection_artifacts['selected_feature_names']).iloc[0]
                )
                
                shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                st.components.v1.html(shap_html, height=160)
                
                st.markdown("---")
                st.markdown("#### Análise do Gráfico")
                st.info(
                    """
                    **Como interpretar o gráfico acima:**
                    - **Valor Base (base value):** É a probabilidade média de reclamação. O ponto de partida da previsão.
                    - **Setas Vermelhas:** Características que **aumentam** a probabilidade de reclamação para este perfil.
                    - **Setas Azuis:** Características que **diminuem** a probabilidade de reclamação.
                    **Análise Prática:** As maiores setas vermelhas são os principais motivos para o modelo considerar este cliente como de risco.
                    """, icon="💡"
                )

def render_business_impact_module(final_artifacts):
    with st.container(border=True):
        st.subheader("Simulação de Impacto no Negócio e Análise de ROI")
        st.markdown("""
        Esta ferramenta calcula o potencial de economia ao implementar uma campanha de retenção proativa.
        A lógica é simples: comparamos o custo de contatar preventivamente os clientes de risco versus o custo de tratar suas reclamações de forma reativa.
        """)

        st.markdown("##### 1. Defina as Premissas Financeiras e Operacionais")
        col1, col2, col3 = st.columns(3)
        with col1:
            cost_proactive = st.number_input("Custo por Contato Proativo (R$)", 0.0, 100000.0, 15.0, 1.0, help="Custo de uma ação preventiva (ligação, voucher, etc.).")
        with col2:
            cost_reactive = st.number_input("Custo por Reclamação Reativa (R$)", 0.0, 100000.0, 150.0, 10.0, help="Custo total para resolver uma reclamação que já aconteceu (horas de suporte, compensação, etc.).")
        with col3:
            effectiveness = st.slider("Efetividade da Ação Proativa (%)", 0, 100, 50, help="Qual a % de reclamações que a ação proativa consegue evitar?")

        st.markdown("##### 2. Defina o Público-Alvo da Campanha com o Limiar de Risco")
        decision_threshold = st.slider("Contatar clientes com probabilidade de reclamação acima de:", 0.0, 1.0, st.session_state.get('decision_threshold', 0.5), 0.01, key="roi_threshold")
        
        y_proba = final_artifacts['y_proba_final']
        y_test = final_artifacts['y_test']
        predictions_as_risk = (y_proba >= decision_threshold)
        
        try:
            tn, fp, fn, tp = confusion_matrix(y_test, predictions_as_risk).ravel()
        except ValueError:
            st.warning("O limiar escolhido não identificou clientes em nenhuma categoria. Ajuste o limiar.")
            tn, fp, fn, tp = 0, 0, 0, 0 

        customers_to_contact = tp + fp
        campaign_total_cost = customers_to_contact * cost_proactive
        
        potential_complaints_targeted = tp
        complaints_avoided = potential_complaints_targeted * (effectiveness / 100.0)
        
        cost_avoided = complaints_avoided * cost_reactive
        
        net_value = cost_avoided - campaign_total_cost
        roi = (net_value / campaign_total_cost) * 100 if campaign_total_cost > 0 else 0

        st.markdown("---")
        st.markdown("##### 3. Resultados da Simulação Financeira")
        
        fig_roi = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = roi,
            number = {'suffix': "%"},
            title = {'text': "ROI da Campanha Proativa"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [-100, 200], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': ProjectConfig.PRIMARY_COLOR, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-100, 0], 'color': '#FF6347'},
                    {'range': [0, 50], 'color': '#FFD700'},
                    {'range': [50, 200], 'color': '#32CD32'}],
            }))
        
        fig_roi.update_layout(height=350, margin=dict(t=50, b=10))
        st.plotly_chart(fig_roi, use_container_width=True)

        res_col3, res_col4, res_col5 = st.columns(3)
        res_col3.metric("Custo da Campanha Proativa", f"R$ {campaign_total_cost:,.2f}")
        res_col4.metric("Custo Evitado (Reativo)", f"R$ {cost_avoided:,.2f}")
        res_col5.metric("Valor Líquido Gerado", f"R$ {net_value:,.2f}")

def render_export_module(final_artifacts, selection_artifacts, processed_df):
    with st.container(border=True):
        st.subheader("Exportação de Resultados e Artefatos")
        st.markdown("Baixe os resultados da análise para uso externo ou para compartilhar com sua equipe.")
        
        y_proba = final_artifacts['y_proba_final']
        y_test_index = final_artifacts['y_test'].index
        
        results_df = processed_df.loc[y_test_index].copy()
        results_df['probabilidade_reclamacao'] = y_proba
        
        export_threshold = st.slider("Limiar de risco para lista de clientes:", 0.0, 1.0, st.session_state.get('decision_threshold', 0.5), 0.01, key="export_threshold")
        high_risk_df = results_df[results_df['probabilidade_reclamacao'] >= export_threshold].sort_values('probabilidade_reclamacao', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(label=f"📥 Baixar Lista de {len(high_risk_df)} Clientes de Alto Risco (.csv)", data=high_risk_df.to_csv(index=False).encode('utf-8'), file_name=f"clientes_risco.csv", mime="text/csv", use_container_width=True)
        with col2:
            if 'feature_ranking_df' in selection_artifacts:
                feature_ranking_df = selection_artifacts['feature_ranking_df']
                st.download_button(label="📥 Baixar Ranking de Features (.csv)", data=feature_ranking_df.to_csv(index=False).encode('utf-8'), file_name="feature_ranking.csv", mime="text/csv", use_container_width=True)

def render_documentation_page():
    st.header("Documentação do Projeto e Metodologia Aplicada")
    st.markdown("Esta seção detalha o fluxo de trabalho completo, as ferramentas utilizadas e as justificativas para as decisões técnicas tomadas ao longo do projeto.")
    
    st.image("https://images.unsplash.com/photo-1542744173-8e7e53415bb0?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80", use_container_width=True)
    st.markdown("---")

    with st.expander("1. Definição do Problema e Objetivo Estratégico", expanded=True):
        st.markdown("""
        O objetivo central deste projeto é desenvolver uma solução de Machine Learning capaz de prever, com alta acurácia e recall, quais clientes de uma empresa de varejo estão mais propensos a registrar uma reclamação formal. A identificação proativa desses clientes permite a implementação de estratégias de retenção direcionadas, visando reduzir o churn (perda de clientes), otimizar os custos de suporte e aumentar a satisfação e lealdade do cliente. O problema é modelado como uma **classificação binária supervisionada** em um contexto de dados altamente desbalanceado.
        """)
    with st.expander("2. Pipeline de Dados: Da Ingestão ao Enriquecimento"):
        st.markdown("""
        Um pipeline robusto foi implementado para garantir a qualidade e a relevância dos dados utilizados na modelagem:
        - **Auditoria de Dados:** Uma análise de profiling inicial foi executada para identificar e quantificar problemas como valores ausentes, tipos de dados inconsistentes e a presença de outliers.
        - **Engenharia de Features:** Foram criadas mais de 10 novas variáveis para capturar nuances do comportamento do cliente que não estavam explícitas nos dados brutos. Exemplos incluem `Customer_Lifetime_Days`, `Total_Spent`, `Luxury_Purchase_Ratio` e `Marketing_Engagements`.
        - **Tratamento de Dados:** Variáveis numéricas foram padronizadas via `StandardScaler`. Variáveis categóricas foram convertidas em formato numérico usando `One-Hot Encoding` para evitar a criação de uma ordem artificial.
        - **Balanceamento de Classes (SMOTE):** Devido à raridade de reclamações (~1%), a técnica SMOTE foi aplicada **exclusivamente no conjunto de treino** para criar exemplos sintéticos da classe minoritária, permitindo que os modelos aprendessem seus padrões de forma mais eficaz.
        """)
    with st.expander("3. Estratégia de Modelagem e Avaliação"):
        st.markdown("""
        - **Seleção de Features (RFECV):** Para combater a "maldição da dimensionalidade" e reduzir o ruído, a técnica RFECV foi utilizada para selecionar automaticamente o subconjunto de features com maior poder preditivo, usando a performance em validação cruzada como critério.
        - **Modelagem de Baseline:** Um portfólio de 8 algoritmos de classificação foi treinado para estabelecer uma linha de base de performance.
        - **Otimização (Tuning):** Os melhores modelos da fase de baseline passaram por um processo de otimização de hiperparâmetros com `GridSearchCV`.
        - **Métricas Chave:** A **AUC** foi a principal métrica para otimização, e o **Recall** e a **Curva de Precisão-Recall** foram utilizados para a análise de negócio.
        """)

def display_modeling_page(df):
    st.header("Pipeline de Modelagem Preditiva", divider='rainbow')
    st.markdown("""
    Bem-vindo à central de Machine Learning. Nesta página, executaremos o pipeline completo, desde a preparação dos dados até o treinamento e avaliação de múltiplos modelos de classificação.
    Cada etapa foi projetada para ser executada sequencialmente, com explicações detalhadas para que você entenda não apenas **o que** está sendo feito, mas **por que** cada decisão é crucial para o sucesso do projeto.
    """)

    if df is None or df.empty:
        st.error("⚠️ Os dados precisam ser processados na página 'Análise do Dataset' antes de iniciar a modelagem.")
        return

    render_data_preparation_module(df)
    
    if 'modeling_data' in st.session_state.get('artifacts', {}):
        render_feature_selection_module(st.session_state.artifacts['modeling_data'])
    
    if 'selection_artifacts' in st.session_state.get('artifacts', {}):
        render_baseline_modeling_module(st.session_state.artifacts['modeling_data'], st.session_state.artifacts['selection_artifacts'])
    
    if 'baseline_artifacts' in st.session_state.get('artifacts', {}):
        render_model_deep_dive_module(st.session_state.artifacts['baseline_artifacts'])
    
    if 'baseline_artifacts' in st.session_state.get('artifacts', {}):
        render_hyperparameter_tuning_module(st.session_state.artifacts['baseline_artifacts'])
        
    if 'tuning_artifacts' in st.session_state.get('artifacts', {}):
        render_final_model_analysis_module(st.session_state.artifacts['tuning_artifacts'], st.session_state.artifacts['modeling_data'], st.session_state.artifacts['selection_artifacts'])

def main():
    st.sidebar.title("Navegação Principal 🚀")
    st.sidebar.markdown("Selecione a página que deseja visualizar:")
    
    page_options = [
        "Página Inicial", 
        "Análise do Dataset", 
        "Análise Exploratória (EDA)",
        "Modelagem e Avaliação",
        "Análise Avançada e de Negócio",
        "Documentação do Projeto"
    ]
    
    page_selection = st.sidebar.radio("Menu:", page_options)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: left;">
            Desenvolvido por:
            <h5 style="margin-top: 5px; margin-bottom: 5px;">Pedro Russo</h5>
            <a href="https://www.linkedin.com/in/pedro-richetti-russo-774189297" target="_blank">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    if page_selection == "Página Inicial":
        display_home_page()

    elif page_selection == "Análise do Dataset":
        display_dataset_page()
        
    elif page_selection == "Análise Exploratória (EDA)":
        display_eda_page()
    
    elif page_selection == "Modelagem e Avaliação":
        display_modeling_page(st.session_state.get('processed_df'))

    elif page_selection == "Análise Avançada e de Negócio":
        st.header("Análise Avançada e de Negócio", divider='rainbow')

        if 'final_artifacts' not in st.session_state.get('artifacts', {}):
            st.error("⚠️ Por favor, complete a Etapa 6 na aba 'Modelagem e Avaliação' primeiro para gerar os artefatos do modelo final.", icon="🚨")
            st.warning("É necessário clicar no botão 'Analisar Modelo Campeão e Gerar Explicações' para prosseguir.")
        else:
            final_artifacts = st.session_state.artifacts['final_artifacts']
            
            tab_xai, tab_roi, tab_export = st.tabs(["🤖 Interpretabilidade (XAI)", "📈 Simulação de ROI", "📤 Exportar Resultados"])
            
            with tab_xai:
                render_global_xai_module(final_artifacts)
                render_local_xai_module(final_artifacts, st.session_state.artifacts['modeling_data'], st.session_state.artifacts['selection_artifacts'])
            
            with tab_roi:
                render_business_impact_module(final_artifacts)
                
            with tab_export:
                render_export_module(final_artifacts, st.session_state.artifacts['selection_artifacts'], st.session_state.get('processed_df'))
    
    elif page_selection == "Documentação do Projeto":
        render_documentation_page()
        
if __name__ == "__main__":
    main()