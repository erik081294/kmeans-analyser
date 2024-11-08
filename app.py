import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple

def load_data() -> pd.DataFrame:
    """Laad de CSV data en return een pandas DataFrame."""
    with st.sidebar:
        st.header("📁 Data Import")
        uploaded_file = st.file_uploader("Upload je CSV bestand", type=['csv'])
        
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error bij het laden van het bestand: {e}")
    return None

def get_column_settings(df: pd.DataFrame) -> Tuple[List[str], dict, List[int], List[str]]:
    """Verbeterde kolom configuratie interface met view-only variabelen."""
    column_settings = {}
    
    # Maak containers voor verschillende kolomtypes
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = []
    
    # Probeer datum kolommen te identificeren
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
            categorical_cols.remove(col)
        except:
            continue
    
    with st.sidebar:
        st.header("🎯 Clustering Configuratie")
        
        # Numerieke kolommen
        st.subheader("Numerieke Kolommen")
        selected_numeric = st.multiselect(
            "Selecteer numerieke kolommen voor clustering:",
            options=numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols,
            help="Kies de numerieke kolommen die je wilt gebruiken voor clustering"
        )
        
        # Categorische kolommen
        if categorical_cols:
            st.subheader("Categorische Kolommen")
            selected_categorical = st.multiselect(
                "Selecteer categorische kolommen:",
                options=categorical_cols,
                help="Categorische kolommen worden automatisch gecodeerd"
            )
        else:
            selected_categorical = []
            
        # Datum kolommen
        if date_cols:
            st.subheader("Datum Kolommen")
            selected_dates = st.multiselect(
                "Selecteer datum kolommen:",
                options=date_cols,
                help="Datum kolommen worden omgezet naar jaar en maand features"
            )
        else:
            selected_dates = []
            
        # View-only kolommen
        st.header("👁️ View-Only Variabelen")
        all_columns = df.columns.tolist()
        clustering_columns = selected_numeric + selected_categorical + selected_dates
        available_view_columns = [col for col in all_columns if col not in clustering_columns]
        
        view_only_columns = st.multiselect(
            "Selecteer kolommen voor alleen visualisatie:",
            options=available_view_columns,
            help="Deze kolommen worden niet gebruikt voor clustering, maar wel meegenomen in de visualisaties"
        )
        
        # K-waarden configuratie
        st.header("🔢 K-waarden")
        k_min = st.number_input("Minimum aantal clusters", min_value=2, value=2)
        k_max = st.number_input("Maximum aantal clusters", min_value=k_min, value=5)
        k_values = list(range(k_min, k_max + 1))
    
    # Verzamel alle geselecteerde kolommen voor clustering
    selected_columns = selected_numeric + selected_categorical + selected_dates
    
    # Maak column settings dictionary
    for col in selected_numeric:
        column_settings[col] = {'type': 'numeriek', 'use_in_clustering': True}
    for col in selected_categorical:
        column_settings[col] = {'type': 'categorisch', 'use_in_clustering': True}
    for col in selected_dates:
        column_settings[col] = {'type': 'datum', 'use_in_clustering': True}
    for col in view_only_columns:
        column_settings[col] = {'type': 'view_only', 'use_in_clustering': False}
    
    return selected_columns, column_settings, k_values, view_only_columns

def prepare_data(df: pd.DataFrame, selected_columns: List[str], column_settings: dict, view_only_columns: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Verbeterde data voorbereiding met dummy variabelen voor categorische kolommen."""
    # Maak kopie van data met zowel clustering als view-only kolommen
    all_columns = selected_columns + view_only_columns
    X = df[all_columns].copy()
    
    # Debug informatie
    missing_info = X.isnull().sum()
    has_missing = missing_info[missing_info > 0]
    
    # Houdt verwijderde rijen bij
    removed_info = []
    
    if len(has_missing) > 0:
        with st.expander("🔧 Missing Values Behandeling", expanded=True):
            st.write(f"Er zijn {len(has_missing)} kolommen met missing values gevonden.")
            
            # Bulk actie voor alle kolommen
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("Bulk actie voor alle kolommen:")
            with col2:
                bulk_strategy = st.selectbox(
                    "Strategie",
                    options=["Individueel", "Verwijderen", "Mediaan", "Gemiddelde"],
                    key="bulk_strategy",
                    help="Pas dezelfde strategie toe op alle kolommen met missing values"
                )
            
            # Container voor missing value keuzes
            missing_choices = {}
            
            # Compacte tabel voor alle kolommen met missing values
            for col in has_missing.index:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{col}** ({missing_info[col]} missing, {(missing_info[col]/len(X)*100):.1f}%)")
                
                with col2:
                    is_numeric = X[col].dtype in ['int64', 'float64']
                    if not is_numeric:
                        st.write("*Niet-numeriek*")
                
                with col3:
                    if bulk_strategy != "Individueel":
                        missing_choices[col] = bulk_strategy
                        st.write(f"*{bulk_strategy}*")
                    else:
                        strategy_options = ["Verwijderen"]
                        if is_numeric:
                            strategy_options.extend(["Mediaan", "Gemiddelde"])
                        
                        missing_choices[col] = st.selectbox(
                            "Strategie",
                            options=strategy_options,
                            key=f"missing_{col}",
                            label_visibility="collapsed"
                        )
        
        # Pas de gekozen strategieën toe
        for col, strategy in missing_choices.items():
            if strategy == "Verwijderen":
                original_len = len(X)
                X = X.dropna(subset=[col])
                removed = original_len - len(X)
                if removed > 0:
                    removed_info.append(f"{removed} rijen verwijderd voor kolom '{col}'")
            
            elif strategy == "Mediaan" and X[col].dtype in ['int64', 'float64']:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                removed_info.append(f"Missing values in '{col}' vervangen met mediaan: {median_val:.2f}")
                
            elif strategy == "Gemiddelde" and X[col].dtype in ['int64', 'float64']:
                mean_val = X[col].mean()
                X[col] = X[col].fillna(mean_val)
                removed_info.append(f"Missing values in '{col}' vervangen met gemiddelde: {mean_val:.2f}")
        
        # Toon samenvatting van acties in een expander
        if removed_info:
            with st.expander("ℹ️ Uitgevoerde acties voor missing values", expanded=False):
                for info in removed_info:
                    st.write(f"- {info}")
    
    # Verwerk de kolommen die voor clustering gebruikt worden
    processed_columns = []
    dummy_columns = []  # Houdt dummy kolommen bij
    
    for col in selected_columns:
        if column_settings[col]['type'] == 'numeriek':
            processed_columns.append(col)
            
        elif column_settings[col]['type'] == 'categorisch':
            # Maak dummy variabelen
            dummies = pd.get_dummies(X[col], prefix=col)
            
            # Bereken proporties voor elke categorie
            for dummy_col in dummies.columns:
                X[dummy_col] = dummies[dummy_col].groupby(level=0).mean()
                dummy_columns.append(dummy_col)
            
            processed_columns.extend(dummy_columns)
            
        elif column_settings[col]['type'] == 'datum':
            try:
                X[col] = pd.to_datetime(X[col])
                X[f"{col}_year"] = X[col].dt.year
                X[f"{col}_month"] = X[col].dt.month
                processed_columns.extend([f"{col}_year", f"{col}_month"])
            except Exception as e:
                st.warning(f"Kon kolom {col} niet als datum verwerken: {str(e)}")
                continue
    
    # Maak een kopie van de volledige dataset voor visualisatie
    full_df = X.copy()
    
    # Selecteer alleen de verwerkte kolommen voor clustering
    X_cluster = X[processed_columns]
    
    # Toon informatie over de dummy variabelen
    if dummy_columns:
        with st.expander("ℹ️ Gecreëerde dummy variabelen", expanded=False):
            st.write("De volgende proporties zijn berekend voor categorische variabelen:")
            for col in dummy_columns:
                st.write(f"- {col}: {X[col].mean():.2%} van de observaties")
    
    # Laatste verificatie voor NaN waarden
    if X_cluster.isnull().any().any():
        st.error("Er zijn nog steeds NaN waarden aanwezig na preprocessing!")
        st.write("Kolommen met NaN waarden:", X_cluster.columns[X_cluster.isnull().any()].tolist())
        return None, None
    
    # Schaal alleen de data voor clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    return X_scaled, full_df

def perform_clustering(X: np.ndarray, k_values: List[int]) -> dict:
    """Voer K-means clustering uit voor verschillende k-waarden."""
    results = {}
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        results[k] = {
            'model': kmeans,
            'clusters': clusters,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X, clusters)
        }
    
    return results

def plot_metrics(results: dict):
    """Plot de metrics (silhouette score en inertia) in één gecombineerde grafiek."""
    st.header("Clustering Evaluatie")
    
    # Maak een overzichtelijke tabel met resultaten
    results_df = pd.DataFrame({
        'Aantal Clusters (K)': list(results.keys()),
        'Silhouette Score': [results[k]['silhouette'] for k in results.keys()],
        'Inertia': [results[k]['inertia'] for k in results.keys()]
    })
    
    # Voeg interpretatie toe aan silhouette scores
    def interpret_silhouette(score):
        if score < 0.2: return "Zwak"
        elif score < 0.5: return "Redelijk"
        elif score < 0.7: return "Goed"
        else: return "Uitstekend"
    
    results_df['Kwaliteit Clusters'] = results_df['Silhouette Score'].apply(interpret_silhouette)
    
    # Toon de tabel
    st.write("### Overzicht Clustering Resultaten")
    st.dataframe(
        results_df.style.format({
            'Silhouette Score': '{:.3f}',
            'Inertia': '{:.0f}'
        }).background_gradient(subset=['Silhouette Score'], cmap='RdYlGn'),
        hide_index=True
    )
    
    # Maak een gecombineerde plot met twee y-assen
    fig = go.Figure()

    # Voeg Silhouette Score toe (linker y-as)
    fig.add_trace(
        go.Scatter(
            x=list(results.keys()),
            y=[results[k]['silhouette'] for k in results.keys()],
            name='Silhouette Score',
            line=dict(color='#2ecc71', width=3),
            mode='lines+markers'
        )
    )

    # Voeg Inertia toe (rechter y-as)
    fig.add_trace(
        go.Scatter(
            x=list(results.keys()),
            y=[results[k]['inertia'] for k in results.keys()],
            name='Inertia',
            line=dict(color='#e74c3c', width=3),
            mode='lines+markers',
            yaxis='y2'
        )
    )

    # Update layout voor twee y-assen
    fig.update_layout(
        title='Evaluatie Metrics per Aantal Clusters',
        xaxis=dict(
            title='Aantal Clusters (K)',
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(
            title='Silhouette Score',
            titlefont=dict(color='#2ecc71'),
            tickfont=dict(color='#2ecc71'),
            range=[0, 1]  # Silhouette score is altijd tussen -1 en 1
        ),
        yaxis2=dict(
            title='Inertia',
            titlefont=dict(color='#e74c3c'),
            tickfont=dict(color='#e74c3c'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Update hover template
    fig.update_traces(
        hovertemplate="<b>K=%{x}</b><br>" +
                     "%{y:.3f}<br>" +
                     "<extra>%{fullData.name}</extra>"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Voeg uitleg toe
    with st.expander("ℹ️ Uitleg Evaluatie Metrics", expanded=False):
        st.write("""
        ### Silhouette Score
        - Bereik: -1 tot 1
        - Hogere waarde is beter
        - Meet hoe goed samples in hun eigen cluster passen vergeleken met andere clusters
        - Score > 0.5 wijst op een goede clustering
        
        ### Inertia (Within-cluster Sum of Squares)
        - Lagere waarde is beter
        - Meet hoe dicht samples bij hun cluster centroid liggen
        - Het 'elbow point' (waar de curve afbuigt) suggereert vaak het optimale aantal clusters
        """)

def export_cluster_profiles(cluster_profiles: dict, k: int, df_with_clusters: pd.DataFrame, key: str = "default"):
    """Exporteer cluster kenmerken naar een CSV bestand."""
    # Maak een lijst van alle profielen
    profile_data = []
    
    for cluster in range(k):
        profile = cluster_profiles[cluster]
        cluster_size = len(df_with_clusters[df_with_clusters['Cluster'] == cluster])
        cluster_percentage = (cluster_size / len(df_with_clusters) * 100)
        
        for feature, stats in profile.items():
            if stats['type'] == 'numeriek':
                profile_data.append({
                    'Cluster': cluster,
                    'Cluster Grootte': cluster_size,
                    'Cluster Percentage': f"{cluster_percentage:.1f}%",
                    'Kenmerk': feature,
                    'Type': 'Numeriek',
                    'Z-score': stats['z_score'],
                    'Waarde in Cluster': stats['cluster_mean'],
                    'Waarde Andere Clusters': stats['other_mean'],
                    'Absoluut Verschil': abs(stats['cluster_mean'] - stats['other_mean'])
                })
            else:  # categorisch
                profile_data.append({
                    'Cluster': cluster,
                    'Cluster Grootte': cluster_size,
                    'Cluster Percentage': f"{cluster_percentage:.1f}%",
                    'Kenmerk': feature,
                    'Type': 'Categorisch',
                    'Z-score': stats['z_score'],
                    'Percentage in Cluster': f"{stats['cluster_mean']:.1f}%",
                    'Percentage Andere Clusters': f"{stats['other_mean']:.1f}%",
                    'Absoluut Verschil (%)': abs(stats['cluster_mean'] - stats['other_mean'])
                })
    
    # Maak DataFrame van de profielen
    profile_df = pd.DataFrame(profile_data)
    
    # Sorteer op cluster en absolute Z-score
    profile_df['Abs_Z_score'] = profile_df['Z-score'].abs()
    profile_df = profile_df.sort_values(['Cluster', 'Abs_Z_score'], ascending=[True, False])
    profile_df = profile_df.drop('Abs_Z_score', axis=1)
    
    # Converteer naar CSV
    csv = profile_df.to_csv(index=False)
    
    # Download knop met unieke key
    st.download_button(
        label="📥 Download Cluster Kenmerken",
        data=csv,
        file_name=f"cluster_profiles_k{k}.csv",  # Voeg K toe aan bestandsnaam
        mime="text/csv",
        help="Download een gedetailleerd overzicht van alle cluster kenmerken",
        key=f"download_profiles_{key}"  # Unieke key
    )

def visualize_clusters(df: pd.DataFrame, results: dict, selected_columns: List[str]):
    """Visualiseer de clusters met verbeterde interactieve analyse."""
    st.header("Cluster Analyse")
    
    # Initialiseer sessie state voor persistente resultaten
    if 'current_k' not in st.session_state:
        st.session_state.current_k = list(results.keys())[0]
    
    # Laat gebruiker K-waarde kiezen voor visualisatie
    k_to_visualize = st.selectbox(
        "Selecteer aantal clusters voor analyse",
        options=list(results.keys()),
        key='k_selector',
        help="Kies het aantal clusters dat je in detail wilt analyseren"
    )
    
    clusters = results[k_to_visualize]['clusters']
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    
    # Maak tabs voor verschillende analyses
    tab1, tab2 = st.tabs(["Cluster Kenmerken", "Gedetailleerde Statistieken"])
    
    with tab1:
        st.write("### Wat kenmerkt elk cluster?")
        
        # Bereken cluster profielen
        cluster_profiles = {}
        for cluster in range(k_to_visualize):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            other_data = df_with_clusters[df_with_clusters['Cluster'] != cluster]
            
            # Bereken z-scores voor numerieke en dummy kolommen
            profile = {}
            
            # Voor numerieke kolommen
            for col in selected_columns:
                if df[col].dtype in ['int64', 'float64']:
                    cluster_mean = cluster_data[col].mean()
                    other_mean = other_data[col].mean()
                    cluster_std = df_with_clusters[col].std()
                    if cluster_std != 0:
                        z_score = (cluster_mean - other_mean) / cluster_std
                        profile[col] = {
                            'z_score': z_score,
                            'cluster_mean': cluster_mean,
                            'other_mean': other_mean,
                            'type': 'numeriek'
                        }
            
            # Voor dummy kolommen
            dummy_cols = [col for col in df_with_clusters.columns if '_' in col 
                        and col != 'Cluster' 
                        and col not in selected_columns]
            for col in dummy_cols:
                cluster_mean = cluster_data[col].mean()
                other_mean = other_data[col].mean()
                # Gebruik standaarddeviatie van de proportie
                n = len(df_with_clusters)
                p = df_with_clusters[col].mean()
                cluster_std = np.sqrt(p * (1-p))  # Standaard deviatie voor proporties
                
                if cluster_std != 0:
                    z_score = (cluster_mean - other_mean) / cluster_std
                    profile[col] = {
                        'z_score': z_score,
                        'cluster_mean': cluster_mean * 100,  # Convert to percentage
                        'other_mean': other_mean * 100,      # Convert to percentage
                        'type': 'categorisch'
                    }
            
            cluster_profiles[cluster] = profile
        
        # Toon cluster definities in een nettere layout
        for cluster in range(k_to_visualize):
            with st.expander(
                f"📊 Cluster {cluster} - {len(df_with_clusters[df_with_clusters['Cluster'] == cluster])} samples "
                f"({(len(df_with_clusters[df_with_clusters['Cluster'] == cluster])/len(df_with_clusters)*100):.1f}%)",
                expanded=True
            ):
                profile = cluster_profiles[cluster]
                sorted_features = sorted(
                    profile.items(),
                    key=lambda x: abs(x[1]['z_score']),
                    reverse=True
                )
                
                # Maak een DataFrame voor de kenmerken
                feature_data = []
                for feature, stats in sorted_features[:5]:
                    if stats['type'] == 'numeriek':
                        feature_data.append({
                            'Kenmerk': feature,
                            'Type': 'Numeriek',
                            'Verschil (σ)': f"{stats['z_score']:.2f}",
                            'Gemiddelde in cluster': f"{stats['cluster_mean']:.2f}",
                            'Gemiddelde andere clusters': f"{stats['other_mean']:.2f}"
                        })
                    else:  # categorisch
                        feature_data.append({
                            'Kenmerk': feature,
                            'Type': 'Categorisch',
                            'Verschil (σ)': f"{stats['z_score']:.2f}",
                            'Percentage in cluster': f"{stats['cluster_mean']:.1f}%",
                            'Percentage andere clusters': f"{stats['other_mean']:.1f}%"
                        })
                
                if feature_data:
                    st.dataframe(
                        pd.DataFrame(feature_data),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.write("Geen significante onderscheidende kenmerken gevonden.")
        
        # Voeg export functionaliteit toe met beide types kenmerken
        st.write("### Export Cluster Kenmerken")
        st.write("Download een gedetailleerd overzicht van alle cluster kenmerken:")
        export_cluster_profiles(cluster_profiles, k_to_visualize, df_with_clusters)
    
    with tab2:
        # Feature importance heatmap met omgedraaide assen en verbeterde tooltip
        st.write("### Feature Importance per Cluster")
        
        # Filter kolommen voor de heatmap (numeriek + dummy variabelen)
        numeric_columns = df_with_clusters.select_dtypes(include=['int64', 'float64']).columns
        heatmap_columns = [col for col in numeric_columns if col in df_with_clusters.columns]  # Inclusief dummy vars
        
        if len(heatmap_columns) == 0:
            st.warning("Geen numerieke of categorische kolommen beschikbaar voor feature importance visualisatie.")
            return df_with_clusters
        
        cluster_means = df_with_clusters.groupby('Cluster')[heatmap_columns].mean()
        cluster_means_normalized = (cluster_means - cluster_means.mean()) / cluster_means.std()
        
        # Maak een custom hover text matrix met dezelfde vorm als de data
        hover_matrix = []
        for col in cluster_means_normalized.columns:  # Voor elke feature
            hover_row = []
            for cluster in cluster_means_normalized.index:  # Voor elke cluster
                z_score = cluster_means_normalized.loc[cluster, col]
                mean_val = cluster_means.loc[cluster, col]
                hover_row.append(f"Z-score: {z_score:.2f}<br>Gemiddelde: {mean_val:.2f}")
            hover_matrix.append(hover_row)
        
        fig_features = px.imshow(
            cluster_means_normalized.T,
            title='Kenmerken per Cluster (Genormaliseerd)',
            labels=dict(x="Cluster", y="Kenmerken", color="Z-score"),
            color_continuous_scale="RdBu_r",
            height=max(400, len(numeric_columns) * 30)
        )
        
        # Update hover template met alle informatie
        fig_features.update_traces(
            customdata=np.array(hover_matrix),
            hovertemplate="<b>Cluster: %{x}</b><br>" +
                         "<b>Kenmerk: %{y}</b><br>" +
                         "%{customdata}<extra></extra>"
        )
        
        fig_features.update_layout(
            xaxis_title="Cluster",
            yaxis_title="Kenmerken",
        )
        st.plotly_chart(fig_features, use_container_width=True)
        
        # Distributie analyse
        st.write("### Distributie Analyse")
        
        # Voeg een filter toe voor kolomtype
        col_type = st.radio(
            "Selecteer type variabele:",
            options=["Numeriek", "Categorisch (Proporties)"],
            horizontal=True
        )
        
        if col_type == "Numeriek":
            available_columns = df_with_clusters.select_dtypes(include=['int64', 'float64']).columns
            available_columns = [col for col in available_columns if not col.startswith('dummy_')]
        else:
            available_columns = [col for col in df_with_clusters.columns if '_' in col and col != 'Cluster']
        
        selected_feature = st.selectbox(
            "Selecteer kenmerk voor distributie analyse",
            options=available_columns
        )
        
        fig_dist = px.histogram(
            df_with_clusters,
            x=selected_feature,
            color='Cluster',
            marginal="box",
            title=f"Distributie van {selected_feature} per Cluster",
            barmode='group',  # Bars naast elkaar
            hover_data={
                selected_feature: ':.2f',
                'Cluster': True
            }
        )
        
        # Update hover template voor histogram
        fig_dist.update_traces(
            hovertemplate="<b>%{x:.2f}</b><br>" +
                         "Aantal: %{y}<br>" +
                         "Cluster: %{customdata[0]}<extra></extra>"
        )
        
        fig_dist.update_layout(
            height=400,
            bargap=0.1,  # Ruimte tussen groepen bars
            bargroupgap=0.05  # Ruimte tussen bars binnen een groep
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    return df_with_clusters

def add_cluster_insights(df_with_clusters: pd.DataFrame, selected_columns: List[str], k: int):
    """Voeg diepere cluster inzichten toe."""
    st.subheader("Gedetailleerde Cluster Analyse")
    
    # Voeg filters toe voor specifieke clusters
    selected_cluster = st.selectbox(
        "Selecteer een cluster voor gedetailleerde analyse",
        options=range(k)
    )
    
    # Toon belangrijkste kenmerken van het cluster
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == selected_cluster]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Cluster Statistieken")
        st.write(f"Aantal samples in cluster: {len(cluster_data)}")
        st.write(f"Percentage van totaal: {(len(cluster_data)/len(df_with_clusters)*100):.2f}%")
        
    with col2:
        st.write("### Unieke Kenmerken")
        # Bereken z-scores voor dit cluster vs andere clusters
        cluster_means = cluster_data[selected_columns].mean()
        other_means = df_with_clusters[df_with_clusters['Cluster'] != selected_cluster][selected_columns].mean()
        differences = (cluster_means - other_means).abs().sort_values(ascending=False)
        
        st.write("Meest onderscheidende features:")
        st.write(differences.head())

def export_data(original_df: pd.DataFrame, processed_df: pd.DataFrame, results: dict, selected_k: int):
    """Functie voor het exporteren van de data met cluster labels voor de geselecteerde K."""
    # Gebruik de verwerkte dataset als basis voor export
    export_df = processed_df.copy()
    
    # Voeg cluster kolom toe voor geselecteerde K
    export_df[f'Cluster (K={selected_k})'] = results[selected_k]['clusters']
    
    # Toon preview van de export data
    st.write("### Preview van export data")
    st.dataframe(export_df.head(), use_container_width=True)
    
    # Maak CSV download knop
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Ruwe Data met Clusters",
        data=csv,
        file_name=f"clustering_results_k{selected_k}.csv",
        mime="text/csv",
        help="Download de volledige dataset inclusief cluster labels"
    )
    
    # Toon wat statistieken
    st.write("### Export Statistieken")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Aantal rijen: {len(export_df)}")
        st.write(f"Aantal kolommen: {len(export_df.columns)}")
        if len(export_df) != len(original_df):
            st.warning(f"⚠️ {len(original_df) - len(export_df)} rijen zijn verwijderd tijdens de voorbewerking")
    
    with col2:
        st.write("") # Lege kolom waar voorheen de cluster verdeling stond

def main():
    st.set_page_config(page_title="K-means Clustering Tool", layout="wide")
    
    st.title("🎯 K-means Clustering Analyse Tool")
    
    # Introductie tekst in een expander
    with st.expander("ℹ️ Over deze tool", expanded=False):
        st.write("""
        Deze tool helpt je bij het uitvoeren van K-means clustering analyse op je dataset.
        
        **Hoe te gebruiken:**
        1. Zorg dat je dataset is opgeschoond en klaar is voor analyse
        2. Upload je CSV bestand in de sidebar
        3. Selecteer de relevante kolommen voor clustering
        4. Configureer het aantal clusters
        5. Bekijk de resultaten en analyses
        6. Exporteer de data met cluster labels
        """)
    
    # Laad data
    df = load_data()
    
    if df is not None:
        # Maak tabs voor de verschillende stappen
        tab_data, tab_cluster, tab_eval, tab_results, tab_export = st.tabs([
            "📊 Data Verkenning",
            "🔍 Clustering Analyse",
            "📏 Evaluatie",
            "📈 Cluster Analyse",
            "💾 Export"
        ])
        
        with tab_data:
            st.header("Data Verkenning")
            
            # Metrics at the top
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Aantal Rijen", len(df))
            with col2:
                st.metric("Aantal Kolommen", len(df.columns))
            with col3:
                numeric_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
                st.metric("Numerieke Kolommen", numeric_cols)
            
            # Dataset preview
            st.write("### Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Datatypes
            st.write("### Datatypes")
            st.dataframe(
                pd.DataFrame({
                    'Kolom': df.dtypes.index,
                    'Type': df.dtypes.values
                }),
                hide_index=True,
                use_container_width=True,
                height=min(35 * len(df.columns) + 38, 1000)  # 35px per row + 38px for header, max 1000px
            )
        
        # Configureer kolommen
        selected_columns, column_settings, k_values, view_only_columns = get_column_settings(df)
        
        if len(selected_columns) > 0:
            with tab_cluster:
                # Bereid data voor
                result = prepare_data(df, selected_columns, column_settings, view_only_columns)
                if result is not None:
                    X, processed_df = result
                    if st.button("Start Clustering Analyse", type="primary"):
                        with st.spinner("Bezig met clustering..."):
                            # Voer clustering uit met de k-waarden uit de sidebar
                            results = perform_clustering(X, k_values)
                            st.session_state.results = results
                            st.session_state.X = X
                            st.session_state.processed_df = processed_df  # Store processed df
                            st.success("Clustering analyse voltooid!")
                else:
                    st.error("Los eerst de missing values op voordat je verder gaat met de clustering analyse.")
            
            with tab_eval:
                if 'results' in st.session_state:
                    # Plot metrics
                    plot_metrics(st.session_state.results)
                else:
                    st.info("Voer eerst de clustering analyse uit in de 'Clustering Analyse' tab.")
            
            with tab_results:
                if 'results' in st.session_state:
                    # Visualiseer clusters met de verwerkte dataset
                    df_with_clusters = visualize_clusters(
                        st.session_state.processed_df,
                        st.session_state.results,
                        selected_columns
                    )
                else:
                    st.info("Voer eerst de clustering analyse uit in de 'Clustering Analyse' tab.")

            with tab_export:
                if 'results' in st.session_state and 'processed_df' in st.session_state:
                    st.header("💾 Data Export")
                    
                    # K-selector voor beide exports
                    selected_k = st.selectbox(
                        "Selecteer aantal clusters (K) voor export:",
                        options=list(st.session_state.results.keys()),
                        help="De geselecteerde K wordt gebruikt voor beide exports"
                    )
                    
                    # Maak twee kolommen voor de verschillende exports
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Export Ruwe Data")
                        export_data(
                            df, 
                            st.session_state.processed_df, 
                            st.session_state.results,
                            selected_k
                        )
                    
                    with col2:
                        st.subheader("Export Cluster Profielen")
                        # Bereken cluster profielen voor geselecteerde K
                        clusters = st.session_state.results[selected_k]['clusters']
                        df_with_clusters = st.session_state.processed_df.copy()
                        df_with_clusters['Cluster'] = clusters
                        
                        # Toon cluster verdeling voor geselecteerde K
                        cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
                        st.write(f"Cluster verdeling voor K={selected_k}:")
                        st.dataframe(
                            pd.DataFrame({
                                'Cluster': cluster_counts.index,
                                'Aantal': cluster_counts.values,
                                'Percentage': (cluster_counts.values / len(df_with_clusters) * 100).round(1)
                            }),
                            hide_index=True
                        )
                        
                        # Bereken cluster profielen voor export
                        cluster_profiles = {}
                        for cluster in range(selected_k):
                            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
                            other_data = df_with_clusters[df_with_clusters['Cluster'] != cluster]
                            
                            # Bereken z-scores voor numerieke en dummy kolommen
                            profile = {}
                            
                            # Voor numerieke kolommen
                            for col in selected_columns:
                                if df[col].dtype in ['int64', 'float64']:
                                    cluster_mean = cluster_data[col].mean()
                                    other_mean = other_data[col].mean()
                                    cluster_std = df_with_clusters[col].std()
                                    if cluster_std != 0:
                                        z_score = (cluster_mean - other_mean) / cluster_std
                                        profile[col] = {
                                            'z_score': z_score,
                                            'cluster_mean': cluster_mean,
                                            'other_mean': other_mean,
                                            'type': 'numeriek'
                                        }
                            
                            # Voor dummy kolommen
                            dummy_cols = [col for col in df_with_clusters.columns if '_' in col 
                                        and col != 'Cluster' 
                                        and col not in selected_columns]
                            for col in dummy_cols:
                                cluster_mean = cluster_data[col].mean()
                                other_mean = other_data[col].mean()
                                # Gebruik standaarddeviatie van de proportie
                                n = len(df_with_clusters)
                                p = df_with_clusters[col].mean()
                                cluster_std = np.sqrt(p * (1-p))  # Standaard deviatie voor proporties
                                
                                if cluster_std != 0:
                                    z_score = (cluster_mean - other_mean) / cluster_std
                                    profile[col] = {
                                        'z_score': z_score,
                                        'cluster_mean': cluster_mean * 100,  # Convert to percentage
                                        'other_mean': other_mean * 100,      # Convert to percentage
                                        'type': 'categorisch'
                                    }
                            
                            cluster_profiles[cluster] = profile
                        
                        # Export de profielen met unieke key
                        export_cluster_profiles(
                            cluster_profiles, 
                            selected_k, 
                            df_with_clusters,
                            key=f"export_tab_{selected_k}"  # Unieke key voor deze tab
                        )
                else:
                    st.info("Voer eerst de clustering analyse uit voordat je de data kunt exporteren.")

if __name__ == "__main__":
    main()
