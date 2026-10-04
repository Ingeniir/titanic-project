import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}


MODEL_FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "FamilyGroup",
    "Title",
    "CabinKnown",
    "Deck",
    "TicketPrefix",
    "NameLength",
]


NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "NameLength"]


CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Embarked",
    "IsAlone",
    "FamilyGroup",
    "Title",
    "CabinKnown",
    "Deck",
    "TicketPrefix",
]


def build_features(dataframe):
    """
    Crée des variables supplémentaires à partir des données brutes
    du Titanic.

    La fonction n'utilise pas la cible et n'apprend aucun paramètre
    à partir des données.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Données brutes sans la cible Survived.

    Returns
    -------
    pandas.DataFrame
        Données enrichies avec les nouvelles variables.
    """
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Colonnes nécessaires absentes : {sorted(missing_columns)}")

    features = dataframe.copy()

    features["FamilySize"] = features["SibSp"] + features["Parch"] + 1

    features["IsAlone"] = (features["FamilySize"] == 1).astype(int)

    features["FamilyGroup"] = np.select(
        condlist=[
            features["FamilySize"].eq(1),
            features["FamilySize"].between(2, 4),
            features["FamilySize"].between(5, 7),
            features["FamilySize"].ge(8),
        ],
        choicelist=["Alone", "Small", "Medium", "Large"],
        default="Unknown",
    )

    features["Title"] = (
        features["Name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
        .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    )

    common_titles = {"Mr", "Miss", "Mrs", "Master"}

    features["Title"] = features["Title"].where(
        features["Title"].isin(common_titles), "Rare"
    )

    features["CabinKnown"] = (features["Cabin"].notna()).astype(int)

    features["Deck"] = features["Cabin"].str.strip().str[0].str.upper().fillna("U")

    features["TicketPrefix"] = (
        features["Ticket"]
        .astype(str)
        .str.upper()
        .str.replace(r"\d+", "", regex=True)
        .str.replace(r"[\s./]+", "", regex=True)
        .replace("", "NONE")
    )

    features["NameLength"] = features["Name"].str.len()

    return features


def select_model_features(dataframe):
    """
    Sélectionne les variables destinées aux modèles.
    """
    missing_columns = set(MODEL_FEATURES).difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Variables de modélisation absentes : {sorted(missing_columns)}"
        )

    return dataframe[MODEL_FEATURES].copy()
