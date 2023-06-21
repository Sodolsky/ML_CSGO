import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Download dataset
url = "https://www.openml.org/data/download/22102255/dataset"
r = requests.get(url, allow_redirects=True)
with open("dataset.txt", "wb") as f:
    f.write(r.content)

# Read data from dataset file
data = []
with open("dataset.txt", "r") as f:
    for line in f.read().split("\n"):
        if line.startswith("@") or line.startswith("%") or line == "":
            continue
        data.append(line)

# Extract column names
columns = []
with open("dataset.txt", "r") as f:
    for line in f.read().split("\n"):
        if line.startswith("@ATTRIBUTE"):
            columns.append(line.split(" ")[1])

# Save data to CSV file
with open("df.csv", "w") as f:
    f.write(",".join(columns))
    f.write("\n")
    f.write("\n".join(data))

# Read data from CSV into a DataFrame
df = pd.read_csv("df.csv")

# Encode the target variable
df['t_win'] = df.round_winner.astype("category").cat.codes

# Calculate correlations and select columns that have the highest correlation
correlations = df[columns+["t_win"]].corr(numeric_only=True)
selected_columns = []
for col in columns+["t_win"]:
    try:
        if abs(correlations[col]['t_win']) > 0.15:
            selected_columns.append(col)
    except KeyError:
        pass
df_selected = df[selected_columns]

# Plot correlation heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(df_selected.corr().sort_values(by="t_win"), annot=True, cmap="YlGnBu")
plt.savefig("Correlation_Map.png")
# Split data into training and testing sets
X, y = df_selected.drop(["t_win"], axis=1), df_selected["t_win"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the feature data
scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(X_train)
XTestScaled = scaler.transform(X_test)


# Create a neural network model
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(20,)))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model with early stopping
early_stopping_callback = keras.callbacks.EarlyStopping(patience=5)
XTrainScaledTrain, XValidation, y_train_train, y_valid = train_test_split(XTrainScaled, y_train, test_size=0.15)
model.fit(XTrainScaledTrain,y_train_train,epochs=30,callbacks=[early_stopping_callback],validation_data=(XValidation,y_valid))
model.evaluate(XTestScaled,y_test)
model.save("Round_Predictions_CSGO.h5")
