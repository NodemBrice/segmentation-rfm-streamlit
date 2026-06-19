# Segmentation des clients (RFM) — App Streamlit

Application de segmentation client basee sur la methode **RFM** (Recency, Frequency,
Monetary) et un modele de clustering **K-Means** entraine sur les donnees de ventes.

L'utilisateur saisit les valeurs RFM d'un client, l'app predit son **segment** et le
**situe graphiquement** parmi les autres clients.

## Contenu du dossier
- `app.py` — l'application Streamlit
- `scaler_rfm.pkl` — le StandardScaler entraine
- `kmeans_rfm.pkl` — le modele K-Means entraine (4 segments)
- `rfm_segments.csv` — les clients deja segmentes (pour les graphiques)
- `requirements.txt` — les dependances

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Puis ouvrir http://localhost:8501

## Deploiement sur Streamlit Community Cloud
1. Pousser ce dossier sur un depot **GitHub** (public).
2. Aller sur https://share.streamlit.io et se connecter avec GitHub.
3. Cliquer **"Create app"** > **"Deploy a public app from GitHub"**.
4. Choisir le repo, la branche `main` et le fichier principal `app.py`.
5. Cliquer **Deploy**. L'app est en ligne en quelques minutes avec une URL publique.
