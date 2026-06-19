import json
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# parametres du modele exportes depuis le notebook (scaler + centres des clusters)
# on predit a la main avec numpy pour ne dependre d'aucune version de scikit-learn (portable)
params = json.load(open('model_params.json'))
mean    = np.array(params['mean'])
scale   = np.array(params['scale'])
centers = np.array(params['centers'])

# on recharge les clients deja segmentes
clients = pd.read_csv('rfm_segments.csv')

noms_segments = {
    0: 'Clients fideles (Champions)',
    1: 'Clients perdus / a risque',
    2: 'Nouveaux clients',
    3: 'Clients reguliers'
}

profils = {
    0: "Achete souvent, recemment et depense beaucoup. A choyer et recompenser.",
    1: "N'a pas achete depuis longtemps. A relancer avec une promotion.",
    2: "Vient d'arriver, peu d'historique. A accompagner pour le fideliser.",
    3: "Client regulier avec du potentiel. A pousser vers le haut de gamme."
}

# pour retrouver le numero de cluster a partir du nom du segment
nom_vers_cluster = {nom: c for c, nom in noms_segments.items()}


def predire_cluster(recency, frequency, monetary):
    # meme traitement que dans le notebook : log puis normalisation
    x = np.log1p(np.array([recency, frequency, monetary], dtype=float))
    x = (x - mean) / scale
    # cluster = centre le plus proche (comme K-Means)
    distances = np.linalg.norm(centers - x, axis=1)
    return int(distances.argmin())


# ============================ MISE EN PAGE ============================
st.set_page_config(page_title="Segmentation RFM", layout="wide")

st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Aller a :",
    ["Vue d'ensemble", "Prediction", "Segments", "Details"]
)


# ============================ PAGE 1 : VUE D'ENSEMBLE ============================
if page == "Vue d'ensemble":
    st.title("Vue d'ensemble")
    st.write("Resume de la segmentation des clients par la methode RFM.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de clients", len(clients))
    col2.metric("Nombre de segments", clients['Cluster'].nunique())
    col3.metric("Chiffre d'affaires total", f"{clients['Monetary'].sum():,.0f}")

    st.subheader("Repartition des clients par segment")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.countplot(y='Segment', data=clients, palette='bright',
                  order=clients['Segment'].value_counts().index, ax=ax1)
    ax1.set_xlabel("Nombre de clients")
    st.pyplot(fig1)

    st.subheader("Profil moyen RFM par segment")
    profil = clients.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
    st.dataframe(profil)


# ============================ PAGE 2 : PREDICTION ============================
elif page == "Prediction":
    st.title("Prediction du segment d'un client")
    st.write("Saisir les valeurs RFM d'un client pour predire son segment et le situer parmi les autres.")

    recency   = st.number_input("Recency (jours depuis le dernier achat)", min_value=0,   value=30)
    frequency = st.number_input("Frequency (nombre de commandes)",          min_value=1,   value=3)
    monetary  = st.number_input("Monetary (montant total depense)",         min_value=0.0, value=500.0)

    if st.button("Predire le segment"):
        cluster = predire_cluster(recency, frequency, monetary)

        st.success("Segment : " + noms_segments[cluster])
        st.info(profils[cluster])

        st.subheader("Position du client parmi les autres")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x='Frequency', y='Monetary', hue='Segment', data=clients,
                        palette='bright', alpha=0.5, ax=ax)
        ax.scatter(frequency, monetary, color='black', s=350, marker='*',
                   edgecolor='white', label='Ce client', zorder=5)
        ax.set_title("Frequency vs Monetary")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        st.pyplot(fig)


# ============================ PAGE 3 : SEGMENTS ============================
elif page == "Segments":
    st.title("Exploration d'un segment")

    segment = st.selectbox("Choisir un segment", list(noms_segments.values()))
    cluster = nom_vers_cluster[segment]
    sous = clients[clients['Cluster'] == cluster]

    st.info(profils[cluster])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre de clients", len(sous))
    col2.metric("Recency moy.",   f"{sous['Recency'].mean():.0f} j")
    col3.metric("Frequency moy.", f"{sous['Frequency'].mean():.1f}")
    col4.metric("Monetary moy.",  f"{sous['Monetary'].mean():.0f}")

    st.subheader("Ce segment compare aux autres")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x='Frequency', y='Monetary', data=clients, color='lightgray',
                    alpha=0.4, ax=ax, label='Autres clients')
    sns.scatterplot(x='Frequency', y='Monetary', data=sous, color='red',
                    alpha=0.7, ax=ax, label=segment)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Liste des clients de ce segment")
    st.dataframe(sous[['Recency', 'Frequency', 'Monetary']].reset_index(drop=True))


# ============================ PAGE 4 : DETAILS ============================
elif page == "Details":
    st.title("Details des donnees")

    choix = st.multiselect("Filtrer par segment", list(noms_segments.values()),
                           default=list(noms_segments.values()))
    donnees = clients[clients['Segment'].isin(choix)]

    st.write(f"{len(donnees)} clients affiches")
    st.dataframe(donnees.reset_index(drop=True))

    st.subheader("Distribution des variables RFM")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(donnees['Recency'],   ax=axes[0], color='steelblue')
    axes[0].set_title('Recency')
    sns.histplot(donnees['Frequency'], ax=axes[1], color='orange')
    axes[1].set_title('Frequency')
    sns.histplot(donnees['Monetary'],  ax=axes[2], color='green')
    axes[2].set_title('Monetary')
    plt.tight_layout()
    st.pyplot(fig)
