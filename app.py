import json
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# parametres du modele exportes depuis le notebook (scaler + centres des clusters)
# on predit a la main pour ne dependre d'aucune version de scikit-learn (portable)
params = json.load(open('model_params.json'))
mean    = np.array(params['mean'])
scale   = np.array(params['scale'])
centers = np.array(params['centers'])

# on recharge les clients deja segmentes pour les graphiques
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


def predire_cluster(recency, frequency, monetary):
    # meme traitement que dans le notebook : log puis normalisation
    x = np.log1p(np.array([recency, frequency, monetary], dtype=float))
    x = (x - mean) / scale
    # cluster = centre le plus proche (comme K-Means)
    distances = np.linalg.norm(centers - x, axis=1)
    return int(distances.argmin())


st.title("Segmentation des clients (RFM)")
st.write("Saisir les valeurs RFM d'un client pour predire son segment et le situer parmi les autres clients.")

# ----- saisie des valeurs RFM -----
recency   = st.number_input("Recency (jours depuis le dernier achat)", min_value=0,   value=30)
frequency = st.number_input("Frequency (nombre de commandes)",          min_value=1,   value=3)
monetary  = st.number_input("Monetary (montant total depense)",         min_value=0.0, value=500.0)

if st.button("Predire le segment"):
    cluster = predire_cluster(recency, frequency, monetary)

    # ----- affichage du profil client -----
    st.success("Segment : " + noms_segments[cluster])
    st.info(profils[cluster])

    # ----- visualisation graphique -----
    st.subheader("Position du client parmi les autres")

    # graphique 1 : Frequency vs Monetary, le client saisi est l'etoile noire
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x='Frequency', y='Monetary', hue='Segment', data=clients,
                    palette='bright', alpha=0.5, ax=ax1)
    ax1.scatter(frequency, monetary, color='black', s=350, marker='*',
                edgecolor='white', label='Ce client', zorder=5)
    ax1.set_title("Frequency vs Monetary")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    st.pyplot(fig1)

    # graphique 2 : nombre de clients par segment
    st.subheader("Repartition des clients par segment")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.countplot(y='Segment', data=clients, palette='bright',
                  order=clients['Segment'].value_counts().index, ax=ax2)
    ax2.set_title("Nombre de clients par segment")
    st.pyplot(fig2)

    # graphique 3 : profil moyen RFM du segment du client
    st.subheader("Profil moyen du segment")
    moyennes = clients[clients['Cluster'] == cluster][['Recency', 'Frequency', 'Monetary']].mean()
    st.write("Valeurs moyennes des clients de ce segment :")
    st.dataframe(moyennes.round(1).to_frame(name='Moyenne du segment'))
