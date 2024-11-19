from sklearn.metrics import root_mean_squared_error # 1.5.1
import pandas as pd # 1.3.5
import matplotlib.pyplot as plt # 3.8.1
import argparse, os, random
import numpy as np # 1.22.4
import json

######################################################################
### PARAMETRES A MODIFIER POUR UTILISER LES FONCTIONS CI-DESSOUS : ###
######################################################################
pred_csv = "y_test.csv"
y_csv = "pred_test.csv"
x_csv = "x_test.csv"
out_png_path = "dossier_image/"
min_max_path = "description_data.json"

#################
### FONCTIONS ###
#################
def get_json_stat(min_max_path):
    """
    Cette fonction permet de lire le fichier json défini dans min_max_path et retourne les 4 valeurs qui y sont écrites
    :param min_max_path: Chemin vers le fichier json de description des données
    :type min_max_path: string
    :return: 4 valeurs (min, max, moyenne et std)
    """
    with open(min_max_path) as json_file:
        data = json.load(json_file)
    return data['min'], data['max'], data['moyenne'], data['std']

def eval_model(pred_csv, y_csv):
    pred_test = pd.read_csv(pred_csv, sep=',', header=None, index_col=False).to_numpy()
    y_test = pd.read_csv(y_csv, sep=',', header=None, index_col=False).to_numpy() 
    min_d, max_d = 176, 381085
    min_val, max_val = int(min_d), int(max_d)
    global_nrmse = 0.0
    i = 0
    for pred, gt in zip(pred_test, y_test):
        if i == 0:
            i +=1
            continue
        global_nrmse += root_mean_squared_error(gt, pred) / (max_val - min_val)
    global_nrmse /= len(pred_test)
    print(f"global test nRMSE: {global_nrmse:.8f}")

def plot_results(input, prediction, gt, filename):
    """
    Cette fonction génère un graphique contenant deux sous-graphes :
    - Le premier sous-graphe affiche les données réelles (combinaison des entrées et des valeurs réelles).
    - Le second sous-graphe affiche les prédictions du modèle (combinaison des entrées et des prédictions).
    Une ligne verte en pointillés est ajoutée pour marquer la séparation entre les données d'entrée et les valeurs de sortie (réelles ou prédites).
    param input (array-like): Les données d'entrée utilisées pour la prédiction.
    param prediction (array-like): Les valeurs prédites par le modèle.
    param gt (array-like): Les valeurs réelles (ground truth) correspondant aux données d'entrée.
    param filename (str): Le nom du fichier où l'image du graphique sera sauvegardée.
    returns: None: La fonction génère un graphique et l'enregistre sous le nom spécifié sans retourner de valeur.
    """
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(np.concatenate([input, gt]))
    ax1.axvline(x=len(input), ymin=0.05, ymax=0.95, color='green', ls='--')
    ax1.set_title('real data')
    ax2.plot(np.concatenate([input, prediction]), 'tab:orange')
    ax2.axvline(x=len(input), ymin=0.05, ymax=0.95, color='green', ls='--')
    ax2.set_title('prediction')
    fig.savefig(filename)
    plt.close()

def generer_plot(pred_csv, x_csv, y_csv, nb_plots = 50):
    """
    Cette fonction sélectionne aléatoirement un nombre défini de cas dans les ensembles de données de test, génère un graphique pour chaque cas, 
    et enregistre ces graphiques dans un répertoire spécifié. Chaque graphique montre les données d'entrée, les prédictions du modèle, 
    et les valeurs réelles (ground truth).
    Les fichiers sont sauvegardés sous forme d'images PNG, avec un nom de fichier formaté comme `test_{n}.png`, où `{n}` est l'indice du graphique.
    param pred_csv (str): Chemin vers le fichier CSV contenant les prédictions du modèle (une seule colonne, sans en-tête).
    param x_csv (str): Chemin vers le fichier CSV contenant les données d'entrée.
    param y_csv (str): Chemin vers le fichier CSV contenant les valeurs réelles (ground truth).
    param nb_plots (int, optionnel): Le nombre de graphiques à générer. Par défaut, 50 graphiques seront générés.
    returns: None: La fonction génère et sauvegarde des graphiques sans retourner de valeur.
    """
    pred_test = pd.read_csv(pred_csv, sep=',', header=None, index_col=False).to_numpy()
    x_test = pd.read_csv(x_csv, sep=',', header=None, index_col=False).to_numpy()       
    y_test = pd.read_csv(y_csv, sep=',', header=None, index_col=False).to_numpy()     
    if not os.path.exists(out_png_path):
            os.makedirs(out_png_path)
    n =0
    while n < nb_plots:
        rnd_idx = random.randint(0, len(x_test) - 1)
        pred = pred_test[rnd_idx]
        input = x_test[rnd_idx]
        gt = y_test[rnd_idx] 
        plot_results(input, pred, gt, os.path.join(out_png_path, f'test_{n}.png'))
        n += 1

#############################
### EXEMPLE D'UTILISATION ###
#############################

# generer_plot(pred_csv, x_csv, y_csv)
# eval_model(pred_csv, y_csv, min_max_path)