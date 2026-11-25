import numpy as np
import msicpe.san as san
from plotly import express as px
import plotly.graph_objects as go

# Vérifie une fois les paramètres exacts de tes fonctions dans la console :
# help(san.detect_fondamentale)
# help(san.detect_pics)


def extraire_vecteur_harmoniques(chemin_fichier, H):
    """
    Charge un fichier .npz, calcule son spectre normalisé en log10
    et retourne le vecteur de dimension (H + 1) des amplitudes à f0, 2*f0, ..., (H+1)*f0.
    """
    # 1. Chargement du signal
    dat = np.load(chemin_fichier)
    s = dat['s']
    t = dat['t']

    # 2. Transformée de Fourier et sélection des fréquences positives
    f, tf = san.trans_fourier(s, t)
    masque_pos = f > 0
    f_pos = f[masque_pos]
    spec = np.abs(tf[masque_pos])

    # 3. Normalisation par le maximum (indispensable pour detect_fondamentale et comparer les signaux)
    spec_norm = spec / np.max(spec)

    # 4. Spectre en logarithme base 10 (+1e-12 pour éviter log10(0))
    spec_log = np.log10(spec_norm + 1e-12)

    # 5. Détection de la fondamentale f0 et des pics
    # NOTE : adapte le déballage selon help(san.detect_fondamentale) et help(san.detect_pics)
    # Exemple standard : f0 = san.detect_fondamentale(f_pos, spec_norm)
    res_f0 = san.detect_fondamentale(f_pos, spec_norm)
    f0 = res_f0[0] if isinstance(res_f0, (tuple, list, np.ndarray)) else res_f0

    res_pics = san.detect_pics(f_pos, spec_norm)
    f_pics = res_pics[0]  # fréquences des pics détectés

    # 6. Extraction des H+1 amplitudes aux fréquences k * f0
    vecteur_H = np.zeros(H + 1)
    tolerance = 0.25 * f0

    for k in range(H + 1):
        f_cible = (k + 1) * f0
        ecarts = np.abs(f_pics - f_cible)
        idx_min = np.argmin(ecarts)

        if ecarts[idx_min] <= tolerance:
            # On retrouve l'indice dans f_pos le plus proche du pic détecté
            idx_freq = np.argmin(np.abs(f_pos - f_pics[idx_min]))
        else:
            # Repli : maximum local autour de f_cible si le pic était trop faible pour detect_pics
            zone = np.where((f_pos >= f_cible - tolerance) & (f_pos <= f_cible + tolerance))[0]
            idx_freq = zone[np.argmax(spec_log[zone])] if len(zone) > 0 else np.argmin(np.abs(f_pos - f_cible))

        vecteur_H[k] = spec_log[idx_freq]

    return vecteur_H


def apprendre_voyelle(voyelle, H, nb_sons=5, dossier="data/apprentissage"):
    """
    Calcule les 5 vecteurs individuels et le vecteur moyen pour une voyelle donnée.
    """
    vecteurs = []
    for i in range(1, nb_sons + 1):
        chemin = f"{dossier}/{voyelle}_{i}.npz"
        v = extraire_vecteur_harmoniques(chemin, H)
        vecteurs.append(v)

    vecteurs = np.array(vecteurs)          # Matrice de taille (5, H+1)
    vecteur_moyen = np.mean(vecteurs, axis=0)  # Vecteur de taille (H+1)
    return vecteurs, vecteur_moyen


# --- QUESTION 1 & 2 : Test sur une voyelle (ex: 'a') avec H = 12 ---
H = 12
vecteurs_a, moyen_a = apprendre_voyelle('a', H)

# Affichage des 5 vecteurs + du vecteur moyen en fonction des indices d'harmoniques
indices_harm = np.arange(1, H + 2)  # 1 = fondamentale (f0), 2 = 2*f0, ...
fig = go.Figure()

for i in range(5):
    fig.add_trace(go.Scatter(
        x=indices_harm, y=vecteurs_a[i],
        mode='lines+markers', name=f"a_{i+1}", opacity=0.45
    ))

fig.add_trace(go.Scatter(
    x=indices_harm, y=moyen_a,
    mode='lines+markers', name="Moyenne /a/",
    line=dict(width=4, color='black')
))

fig.update_layout(
    title=f"Vecteurs caractéristiques et moyenne pour /a/ (H={H})",
    xaxis_title="Rang de l'harmonique (1 = fondamentale f0)",
    yaxis_title="Log10(Amplitude normalisée)"
)
fig.show()


# --- QUESTION 3 : Répétition sur toutes les voyelles ---
voyelles = ['a', 'e', 'i', 'o', 'u']
modeles_moyens = {}

fig_all = go.Figure()
for v in voyelles:
    _, vec_moy = apprendre_voyelle(v, H)
    modeles_moyens[v] = vec_moy
    fig_all.add_trace(go.Scatter(
        x=indices_harm, y=vec_moy,
        mode='lines+markers', name=f"Voyelle /{v}/"
    ))

fig_all.update_layout(
    title=f"Signatures spectrales moyennes des 5 voyelles (H={H})",
    xaxis_title="Rang de l'harmonique (1 = fondamentale f0)",
    yaxis_title="Log10(Amplitude normalisée)"
)
fig_all.show()