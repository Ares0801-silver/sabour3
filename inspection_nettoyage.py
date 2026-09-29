"""
Pipeline complet du projet : inspection, nettoyage, echantillonnage,
cinq piliers, tableau de bord et comparaison statistique echantillon/population.
A adapter une fois le dataset Kaggle confirme : changer DATASET_PATH.

Ce fichier est aussi importe comme module par le dashboard Streamlit (app.py,
pages/) et par le notebook, pour ne pas dupliquer le code entre les trois.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif necessaire, juste enregistrer les figures
import matplotlib.pyplot as plt
from scipy import stats
import os

from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent / "ecommerce_dataset_+1m.csv"
# le chemin est calcule par rapport a l'emplacement de ce fichier, pas par
# rapport au dossier depuis lequel le script est lance, pour eviter les
# erreurs FileNotFoundError selon l'outil utilise pour executer le code

# Colonnes ou un vide est attendu et normal, pas une erreur a corriger
# return_reason : vide si le produit n'a pas ete retourne
# customer_feedback : vide si le client n'a pas laisse d'avis
# coupon_code : vide si aucun coupon n'a ete utilise
COLONNES_VIDE_ATTENDU = ["return_reason", "customer_feedback", "coupon_code"]

journal = []  # garde une trace de chaque decision de nettoyage, avec sa raison


def log(action, raison):
    journal.append({"action": action, "raison": raison})
    print(f"- {action} : {raison}")


# ---------- ETAPE 1 : INSPECTION ----------

def inspecter(df):
    print("\n=== Forme du dataset ===")
    print(f"Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")

    print("\n=== Types de colonnes ===")
    print(df.dtypes.value_counts())

    print("\n=== Valeurs manquantes (proportion) ===")
    manquants = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    print(manquants[manquants > 0].round(2))

    print("\n=== Vides attendus, a ne pas traiter comme une erreur ===")
    for col in COLONNES_VIDE_ATTENDU:
        if col in df.columns:
            taux = df[col].isna().mean() * 100
            print(f"{col} : {taux:.1f}% de vide, logique si la description Kaggle dit vrai, a verifier quand meme")

    print("\n=== Doublons ===")
    print(f"Lignes dupliquees : {df.duplicated().sum()}")
    if "id" in df.columns:
        print(f"Identifiants dupliques : {df['id'].duplicated().sum()}")

    print("\n=== Colonnes categorielles : valeurs uniques ===")
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique < 30:
            print(f"{col} ({n_unique} valeurs) : {df[col].unique()}")
        else:
            print(f"{col} : {n_unique} valeurs uniques, trop nombreuses pour affichage")

    print("\n=== Resume statistique des colonnes numeriques ===")
    print(df.select_dtypes(include=np.number).describe().T)

    return manquants, cat_cols


# ---------- ETAPE 2 : NETTOYAGE ----------

def nettoyer(df, manquants, seuil_suppression_colonne=60.0):
    df = df.copy()

    # 1. colonnes trop vides : on les retire, inutile de les imputer
    # sauf celles ou le vide est attendu et porte une information en soi
    colonnes_a_retirer = [
        col for col in manquants[manquants > seuil_suppression_colonne].index.tolist()
        if col not in COLONNES_VIDE_ATTENDU
    ]
    if colonnes_a_retirer:
        df = df.drop(columns=colonnes_a_retirer)
        log(
            f"Suppression de {len(colonnes_a_retirer)} colonnes : {colonnes_a_retirer}",
            f"plus de {seuil_suppression_colonne}% de valeurs manquantes",
        )

    # 2. doublons stricts
    n_avant = len(df)
    df = df.drop_duplicates()
    if len(df) < n_avant:
        log(f"Suppression de {n_avant - len(df)} lignes dupliquees", "doublons stricts sur toutes les colonnes")

    # 3. colonnes de date : uniformiser le format
    # order_date et account_creation_date sont les deux colonnes de date connues sur ce dataset
    for col in df.columns:
        if "date" in col.lower() or col.lower() in ("order_date", "account_creation_date"):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                log(f"Conversion de {col} en date", "uniformiser le format pour l'analyse temporelle")
            except Exception:
                pass

    # 4. valeurs manquantes restantes sur colonnes numeriques : imputation par mediane
    # les colonnes de COLONNES_VIDE_ATTENDU sont exclues, leur vide est logique et pas une erreur
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if col in COLONNES_VIDE_ATTENDU:
            continue
        n_na = df[col].isna().sum()
        if n_na > 0:
            mediane = df[col].median()
            df[col] = df[col].fillna(mediane)
            log(f"Imputation de {n_na} valeurs manquantes dans {col} par la mediane ({mediane})",
                "colonne numerique importante, taux de vide raisonnable")

    # 5. valeurs aberrantes evidentes a signaler, pas a supprimer automatiquement
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bas, haut = q1 - 3 * iqr, q3 + 3 * iqr
        n_extremes = ((df[col] < bas) | (df[col] > haut)).sum()
        if n_extremes > 0:
            print(f"A verifier manuellement : {col} a {n_extremes} valeurs extremes (hors [{bas:.1f}, {haut:.1f}])")

    return df


# ---------- ETAPE 3 : ECHANTILLONNAGE ----------

TAILLE_ECHANTILLON = 30000  # taille cible de chaque echantillon
COLONNE_STRATE = "category"           # variable pour l'echantillonnage stratifie
COLONNE_GRAPPE = "shipping_country"   # variable pour l'echantillonnage par grappes


def echantillon_aleatoire_simple(df, taille=TAILLE_ECHANTILLON, seed=42):
    """Tirage uniforme, chaque ligne a la meme chance d'etre choisie."""
    return df.sample(n=taille, random_state=seed)


def echantillon_stratifie(df, colonne_strate=COLONNE_STRATE, taille=TAILLE_ECHANTILLON, seed=42):
    """Tirage qui respecte les proportions de chaque categorie de colonne_strate."""
    if colonne_strate not in df.columns:
        raise ValueError(f"Colonne {colonne_strate} absente du dataset")
    fraction = taille / len(df)
    morceaux = [
        groupe.sample(frac=fraction, random_state=seed)
        for _, groupe in df.groupby(colonne_strate)
    ]
    return pd.concat(morceaux)


def echantillon_systematique(df, taille=TAILLE_ECHANTILLON):
    """Tirage d'une ligne toutes les k lignes, sur le dataset trie par index."""
    pas = max(len(df) // taille, 1)
    return df.iloc[::pas].head(taille)


def echantillon_grappes(df, colonne_grappe=COLONNE_GRAPPE, taille=TAILLE_ECHANTILLON, seed=42):
    """Tirage de grappes entieres (ex : tous les pays choisis), pas d'individus isoles."""
    if colonne_grappe not in df.columns:
        raise ValueError(f"Colonne {colonne_grappe} absente du dataset")
    rng = np.random.default_rng(seed)
    grappes = df[colonne_grappe].unique()
    rng.shuffle(grappes)

    lignes_choisies = pd.DataFrame()
    for grappe in grappes:
        lignes_choisies = pd.concat([lignes_choisies, df[df[colonne_grappe] == grappe]])
        if len(lignes_choisies) >= taille:
            break
    return lignes_choisies.head(taille)


def construire_echantillons(df):
    """Construit les quatre echantillons et les garde distincts pour comparaison."""
    echantillons = {
        "aleatoire_simple": echantillon_aleatoire_simple(df),
        "stratifie": echantillon_stratifie(df),
        "systematique": echantillon_systematique(df),
        "grappes": echantillon_grappes(df),
    }
    for nom, ech in echantillons.items():
        log(f"Echantillon {nom} construit, {len(ech)} lignes", "methode demandee dans le brief")
    return echantillons


# ---------- ETAPE 4 : LES CINQ PILIERS ----------
# Colonnes utilisees ici, a adapter si les noms different une fois le fichier ouvert
COL_PRIX = "unit_price_usd"
COL_PROFIT = "profit_usd"
COL_QUANTITE = "quantity"
COL_REMISE = "discount_percent"
COL_CATEGORIE = "category"
COL_SEGMENT = "customer_segment"
COL_DATE = "order_date"

DOSSIER_FIGURES = "dashboard"
os.makedirs(DOSSIER_FIGURES, exist_ok=True)


def pilier_distribution(df, nom_echantillon):
    """Univarie : comment se repartit chaque variable, seule."""
    resume = df[[COL_PRIX, COL_PROFIT, COL_QUANTITE]].describe().T

    fig, ax = plt.subplots()
    df[COL_PRIX].hist(bins=40, ax=ax)
    ax.set_title(f"Distribution du prix unitaire — {nom_echantillon}")
    ax.set_xlabel(COL_PRIX)
    ax.set_ylabel("nombre de commandes")
    fig.savefig(f"{DOSSIER_FIGURES}/distribution_{nom_echantillon}.png", bbox_inches="tight")
    plt.close(fig)

    return resume


def pilier_relation(df, nom_echantillon):
    """Bivarie : comment deux variables evoluent l'une par rapport a l'autre."""
    colonnes = [COL_PRIX, COL_QUANTITE, COL_REMISE, COL_PROFIT]
    correlations = df[colonnes].corr()

    fig, ax = plt.subplots()
    im = ax.imshow(correlations, vmin=-1, vmax=1)
    ax.set_xticks(range(len(colonnes)))
    ax.set_yticks(range(len(colonnes)))
    ax.set_xticklabels(colonnes, rotation=45, ha="right")
    ax.set_yticklabels(colonnes)
    ax.set_title(f"Correlations entre variables — {nom_echantillon}")
    fig.colorbar(im, ax=ax)
    fig.savefig(f"{DOSSIER_FIGURES}/relation_{nom_echantillon}.png", bbox_inches="tight")
    plt.close(fig)

    return correlations


def pilier_groupes(df, nom_echantillon):
    """Multivarie : profit moyen croise categorie de produit et segment client."""
    tableau = df.groupby([COL_CATEGORIE, COL_SEGMENT])[COL_PROFIT].mean().unstack()

    fig, ax = plt.subplots()
    im = ax.imshow(tableau.values, aspect="auto")
    ax.set_xticks(range(len(tableau.columns)))
    ax.set_yticks(range(len(tableau.index)))
    ax.set_xticklabels(tableau.columns, rotation=45, ha="right")
    ax.set_yticklabels(tableau.index)
    ax.set_title(f"Profit moyen par categorie et segment — {nom_echantillon}")
    fig.colorbar(im, ax=ax)
    fig.savefig(f"{DOSSIER_FIGURES}/groupes_{nom_echantillon}.png", bbox_inches="tight")
    plt.close(fig)

    return tableau


def pilier_synthese(df, nom_echantillon):
    """Indicateurs globaux qui resument l'echantillon en quelques chiffres."""
    indicateurs = {
        "nombre_commandes": len(df),
        "revenu_total_usd": df[COL_PRIX].mul(df[COL_QUANTITE]).sum(),
        "profit_total_usd": df[COL_PROFIT].sum(),
        "panier_moyen_usd": df[COL_PRIX].mul(df[COL_QUANTITE]).mean(),
        "taux_remise_moyen": df[COL_REMISE].mean(),
    }

    fig, ax = plt.subplots()
    ax.axis("off")
    texte = "\n".join(f"{cle} : {valeur:,.2f}" for cle, valeur in indicateurs.items())
    ax.text(0.05, 0.5, texte, fontsize=11, va="center")
    ax.set_title(f"Synthese — {nom_echantillon}")
    fig.savefig(f"{DOSSIER_FIGURES}/synthese_{nom_echantillon}.png", bbox_inches="tight")
    plt.close(fig)

    return indicateurs


def pilier_temps(df, nom_echantillon):
    """Evolution mensuelle du volume de commandes et du profit."""
    if COL_DATE not in df.columns:
        log(f"Pilier temps ignore pour {nom_echantillon}", f"colonne {COL_DATE} absente de l'echantillon")
        return None

    serie = df.set_index(COL_DATE).resample("ME")[COL_PROFIT].sum()

    fig, ax = plt.subplots()
    serie.plot(ax=ax)
    ax.set_title(f"Profit mensuel — {nom_echantillon}")
    ax.set_xlabel("mois")
    ax.set_ylabel("profit total (usd)")
    fig.savefig(f"{DOSSIER_FIGURES}/temps_{nom_echantillon}.png", bbox_inches="tight")
    plt.close(fig)

    return serie


def analyser_cinq_piliers(echantillons):
    """Applique les cinq piliers sur chaque echantillon, garde les resultats a part."""
    resultats = {}
    for nom, df in echantillons.items():
        resultats[nom] = {
            "distribution": pilier_distribution(df, nom),
            "relation": pilier_relation(df, nom),
            "groupes": pilier_groupes(df, nom),
            "synthese": pilier_synthese(df, nom),
            "temps": pilier_temps(df, nom),
        }
        log(f"Cinq piliers calcules pour {nom}", "analyse univariee, bivariee et multivariee completee")
    return resultats


# ---------- ETAPE 6 : COMPARAISON ECHANTILLON / POPULATION ----------

def comparer_variable_numerique(df_pop, df_ech, colonne):
    """Compare une variable numerique entre la population et un echantillon.
    Le test de Kolmogorov-Smirnov verifie si les deux distributions se ressemblent,
    ses hypotheses (variable continue, echantillons independants) sont respectees ici."""
    pop_vals = df_pop[colonne].dropna()
    ech_vals = df_ech[colonne].dropna()

    moyenne_pop, moyenne_ech = pop_vals.mean(), ech_vals.mean()
    erreur_absolue = abs(moyenne_ech - moyenne_pop)
    erreur_relative = erreur_absolue / moyenne_pop * 100 if moyenne_pop != 0 else np.nan

    ks_stat, ks_pvalue = stats.ks_2samp(pop_vals, ech_vals)

    return {
        "moyenne_population": moyenne_pop,
        "moyenne_echantillon": moyenne_ech,
        "erreur_absolue": erreur_absolue,
        "erreur_relative_pct": erreur_relative,
        "diff_mediane": abs(ech_vals.median() - pop_vals.median()),
        "diff_ecart_type": abs(ech_vals.std() - pop_vals.std()),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "distributions_proches": ks_pvalue > 0.05,
    }


def comparer_variable_categorielle(df_pop, df_ech, colonne):
    """Compare la repartition des categories entre population et echantillon.
    Le test du khi-deux suppose des effectifs attendus d'au moins 5 par case,
    sinon son resultat n'est pas fiable et un avertissement est renvoye a la place."""
    prop_pop = df_pop[colonne].value_counts(normalize=True) * 100
    prop_ech = df_ech[colonne].value_counts(normalize=True) * 100
    diff_proportions = (prop_ech - prop_pop).abs().sort_values(ascending=False)

    table_contingence = pd.DataFrame({
        "population": df_pop[colonne].value_counts(),
        "echantillon": df_ech[colonne].value_counts(),
    }).fillna(0)

    chi2_stat, chi2_pvalue, _, effectifs_attendus = stats.chi2_contingency(table_contingence)
    hypotheses_respectees = (effectifs_attendus >= 5).all()

    resultat = {
        "ecart_max_proportions_pct": diff_proportions.max(),
        "categorie_plus_ecartee": diff_proportions.index[0],
        "chi2_stat": chi2_stat,
        "chi2_pvalue": chi2_pvalue if hypotheses_respectees else None,
        "hypotheses_khi_deux_respectees": hypotheses_respectees,
    }
    if not hypotheses_respectees:
        log(
            f"Test du khi-deux non fiable sur {colonne}",
            "au moins une case a un effectif attendu inferieur a 5, p-value ignoree",
        )
    return resultat


def comparer_correlations(df_pop, df_ech, colonnes):
    """Mesure si les correlations entre variables cles sont conservees dans l'echantillon."""
    corr_pop = df_pop[colonnes].corr()
    corr_ech = df_ech[colonnes].corr()
    ecart_moyen = (corr_pop - corr_ech).abs().values[np.triu_indices(len(colonnes), k=1)].mean()
    return {"ecart_moyen_correlations": ecart_moyen}


def comparer_echantillon_population(df_pop, df_ech, nom_echantillon):
    """Assemble toutes les comparaisons pour un echantillon et calcule
    un score global de representativite, de 0 (mauvais) a 100 (excellent)."""
    num_prix = comparer_variable_numerique(df_pop, df_ech, COL_PRIX)
    num_profit = comparer_variable_numerique(df_pop, df_ech, COL_PROFIT)
    cat = comparer_variable_categorielle(df_pop, df_ech, COL_CATEGORIE)
    corr = comparer_correlations(df_pop, df_ech, [COL_PRIX, COL_QUANTITE, COL_REMISE, COL_PROFIT])

    # chaque composante est ramenee entre 0 et 100, puis moyennee
    score_prix = max(0, 100 - num_prix["erreur_relative_pct"])
    score_profit = max(0, 100 - num_profit["erreur_relative_pct"]) if not np.isnan(num_profit["erreur_relative_pct"]) else 100
    score_categorie = max(0, 100 - cat["ecart_max_proportions_pct"])
    score_correlation = max(0, 100 - corr["ecart_moyen_correlations"] * 100)
    score_global = np.mean([score_prix, score_profit, score_categorie, score_correlation])

    return {
        "echantillon": nom_echantillon,
        "erreur_relative_prix_pct": num_prix["erreur_relative_pct"],
        "erreur_relative_profit_pct": num_profit["erreur_relative_pct"],
        "diff_mediane_prix": num_prix["diff_mediane"],
        "diff_ecart_type_prix": num_prix["diff_ecart_type"],
        "ks_pvalue_prix": num_prix["ks_pvalue"],
        "distributions_prix_proches": num_prix["distributions_proches"],
        "ecart_max_proportions_categorie_pct": cat["ecart_max_proportions_pct"],
        "categorie_plus_ecartee": cat["categorie_plus_ecartee"],
        "chi2_pvalue_categorie": cat["chi2_pvalue"],
        "ecart_moyen_correlations": corr["ecart_moyen_correlations"],
        "score_representativite": round(score_global, 1),
    }


def comparer_tous_echantillons(df_pop, echantillons):
    """Construit le tableau comparatif complet, une ligne par echantillon,
    trie du plus representatif au moins representatif."""
    lignes = [comparer_echantillon_population(df_pop, ech, nom) for nom, ech in echantillons.items()]
    tableau = pd.DataFrame(lignes).sort_values("score_representativite", ascending=False)
    for _, ligne in tableau.iterrows():
        log(
            f"Comparaison {ligne['echantillon']} vs population",
            f"score de representativite {ligne['score_representativite']}/100",
        )
    return tableau


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    manquants, cat_cols = inspecter(df)
    df_propre = nettoyer(df, manquants)

    echantillons = construire_echantillons(df_propre)
    resultats = analyser_cinq_piliers(echantillons)
    tableau_comparatif = comparer_tous_echantillons(df_propre, echantillons)

    print("\n=== Journal complet ===")
    for entree in journal:
        print(f"{entree['action']} -> {entree['raison']}")
    pd.DataFrame(journal).to_csv("journal.csv", index=False)

    df_propre.to_csv("dataset_nettoye.csv", index=False)
    print(f"\nDataset nettoye enregistre : {df_propre.shape[0]} lignes, {df_propre.shape[1]} colonnes")

    for nom, ech in echantillons.items():
        ech.to_csv(f"echantillon_{nom}.csv", index=False)
        print(f"Echantillon {nom} enregistre : {ech.shape[0]} lignes")

    tableau_comparatif.to_csv("comparaison_echantillons.csv", index=False)
    print("\n=== Tableau comparatif ===")
    print(tableau_comparatif.to_string(index=False))

    # indicateurs de synthese, population et chaque echantillon, pour le rapport et la presentation
    synthese_population = pilier_synthese(df_propre, "population")
    lignes_synthese = [{"echantillon": "population", **synthese_population}]
    for nom, res in resultats.items():
        lignes_synthese.append({"echantillon": nom, **res["synthese"]})
    pd.DataFrame(lignes_synthese).to_csv("synthese_echantillons.csv", index=False)
    print("\nFichiers prets pour le rapport et la presentation : journal.csv, "
          "synthese_echantillons.csv, comparaison_echantillons.csv, dossier dashboard/")

