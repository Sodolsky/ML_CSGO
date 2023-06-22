# Importowanie niezbędnych bibliotek
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Pobranie zbioru danych
url = "https://www.openml.org/data/download/22102255/dataset"
r = requests.get(url, allow_redirects=True)
with open("dataset.txt", "wb") as f:
    f.write(r.content)

# Odczytanie danych z pliku zbioru
data = []
with open("dataset.txt", "r") as f:
    for line in f.read().split("\n"):
        if line.startswith("@") or line.startswith("%") or line == "":
            continue
        data.append(line)

# Wyodrębnienie nazw kolumn
columns = []
with open("dataset.txt", "r") as f:
    for line in f.read().split("\n"):
        if line.startswith("@ATTRIBUTE"):
            columns.append(line.split(" ")[1])

# Zapisanie danych do pliku CSV
with open("df.csv", "w") as f:
    f.write(",".join(columns))
    f.write("\n")
    f.write("\n".join(data))

# Odczytanie danych z pliku CSV do DataFrame
df = pd.read_csv("df.csv")

# Zakodowanie zmiennej docelowej
df['t_win'] = df.round_winner.astype("category").cat.codes

# Obliczenie korelacji i wybór kolumn o najwyższej korelacji
correlations = df[columns+["t_win"]].corr(numeric_only=True)
selected_columns = []
for col in columns+["t_win"]:
    try:
        if abs(correlations[col]['t_win']) > 0.15:
            selected_columns.append(col)
    except KeyError:
        pass
df_selected = df[selected_columns]

# Wygenerowanie wykresu korelacji
plt.figure(figsize=(18, 12))
sns.heatmap(df_selected.corr().sort_values(by="t_win"), annot=True, cmap="YlGnBu")
plt.savefig("Mapa_Korelacji.png")

# Podział danych na zbiór treningowy i testowy
X, y = df_selected.drop(["t_win"], axis=1), df_selected["t_win"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standaryzacja danych cech
scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(X_train)
XTestScaled = scaler.transform(X_test)

# Utworzenie modelu sieci neuronowej
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(20,)))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# Kompilacja modelu
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Trening modelu z wcześniejszym zatrzymaniem
early_stopping_callback = keras.callbacks.EarlyStopping(patience=5)
XTrainScaledTrain, XValidation, y_train_train, y_valid = train_test_split(XTrainScaled, y_train, test_size=0.15)
model.fit(XTrainScaledTrain, y_train_train, epochs=30, callbacks=[early_stopping_callback], validation_data=(XValidation, y_valid))

# Ocena modelu na zbiorze testowym
model.evaluate(XTestScaled, y_test)

# Zapisanie modelu
model.save("Round_Predictions_CSGO.h5")
