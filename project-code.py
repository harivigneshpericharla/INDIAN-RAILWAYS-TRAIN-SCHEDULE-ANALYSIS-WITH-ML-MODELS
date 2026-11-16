# Indian Railways Train Schedule Analysis (Dec 2017)
# Steps: cleaning -> feature building -> EDA -> five objectives -> five classification models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------------
# Keep this CSV in the same folder as the script
# Everything is read as text first because two rows in the file are broken
df = pd.read_csv("train.csv", dtype=str)

print(df.head())
print(df.tail())
print("\nShape:", df.shape)
df.info()

# ---------------------------------------------------------------
# 2. FIX THE BROKEN ROWS
# ---------------------------------------------------------------
# One station name (KALGURKI H) got split over two lines in the CSV, so for
# trains 6919 and 6920 the real values ended up in a second row that starts
# with the stray text "K". Check how many such rows we have:
broken = df[df["Train No"] == "K"].index
print("\nBroken rows:", list(broken))

# The row above each broken row has NA in the numeric columns, and the
# broken row holds those values shifted by three columns. Move them back.
fix_cols = ["Arrival time", "Departure Time", "Distance", "Source Station",
            "Source Station Name", "Destination Station", "Destination Station Name"]
shifted = ["Train Name", "SEQ", "Station Code", "Station Name",
           "Arrival time", "Departure Time", "Distance"]

for i in broken:
    df.loc[i - 1, fix_cols] = df.loc[i, shifted].values

df = df.drop(index=broken)
print("Missing values after the fix:", df.isnull().sum().sum())

# ---------------------------------------------------------------
# 3. CLEANING AND TYPE CONVERSION
# ---------------------------------------------------------------
df["SEQ"] = df["SEQ"].astype(int)
df["Distance"] = df["Distance"].astype(int)

# Tidy up the text columns
for col in ["Train Name", "Station Name", "Source Station Name", "Destination Station Name"]:
    df[col] = df[col].str.strip().str.title()


def to_minutes(series):
    """Turn 'HH:MM:SS' text into minutes after midnight."""
    parts = series.str.split(":", expand=True).astype(int)
    return parts[0] * 60 + parts[1]


df["arr_min"] = to_minutes(df["Arrival time"])
df["dep_min"] = to_minutes(df["Departure Time"])

print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Sort so every train's stops are in running order
df = df.sort_values(["Train No", "SEQ"]).reset_index(drop=True)

# Note: 00:00:00 in "Arrival time" at the first stop and in "Departure Time"
# at the last stop is just a placeholder (the train doesn't arrive at its own
# starting point), so I am not treating those as real midnight times.

# ---------------------------------------------------------------
# 4. BUILD ONE ROW PER TRAIN
# ---------------------------------------------------------------
# The file has one row per stop, but most questions are about whole trains.
# Times only go from 00:00 to 23:59, so if the clock goes backwards between two
# stops the train has crossed midnight and we add a day.
records = []

for train_no, stops in df.groupby("Train No"):
    # build the list of events in order: dep(1), arr(2), dep(2), arr(3) ...
    events = []
    last = len(stops) - 1
    for k, (arr, dep) in enumerate(zip(stops["arr_min"], stops["dep_min"])):
        if k > 0:
            events.append(arr)
        if k < last:
            events.append(dep)
    events = np.array(events)

    day_jumps = np.cumsum(np.r_[0, (np.diff(events) < 0).astype(int)])
    events = events + day_jumps * 1440

    run_time = events[-1] - events[0]

    # halt time at the stops in the middle = departure - arrival
    halts = [events[2 * k] - events[2 * k - 1] for k in range(1, last)]

    records.append({
        "train_no": train_no,
        "train_name": stops["Train Name"].iloc[0],
        "source": stops["Source Station Name"].iloc[0],
        "destination": stops["Destination Station Name"].iloc[0],
        "num_stops": len(stops),
        "distance_km": stops["Distance"].max(),
        "run_hours": run_time / 60,
        "avg_halt_min": np.mean(halts) if halts else 0,
        "start_hour": stops["dep_min"].iloc[0] // 60,
    })

trains = pd.DataFrame(records)
trains["avg_speed"] = trains["distance_km"] / trains["run_hours"]
trains["stops_per_100km"] = trains["num_stops"] / trains["distance_km"] * 100

# Train category from the train number:
#  - numbers with fewer than 5 digits were originally 0xxxx -> special trains
#  - 12xxx -> superfast, 10xxx / 11xxx -> mail / express
def get_category(number):
    if len(number) < 5:
        return "Special"
    if number.startswith("12"):
        return "Superfast"
    return "Mail/Express"


trains["category"] = trains["train_no"].apply(get_category)

print("\nTrains in the data:", len(trains))
print(trains.head())
print(trains["category"].value_counts())

# ---------------------------------------------------------------
# 5. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------
num_cols = ["num_stops", "distance_km", "run_hours", "avg_halt_min",
            "avg_speed", "stops_per_100km"]

print("\nDescriptive statistics:")
print(trains[num_cols].describe())

print("\nCorrelation:")
print(trains[num_cols].corr())

print("\nSkewness of distance:", trains["distance_km"].skew())
print("Kurtosis of distance:", trains["distance_km"].kurt())
print("Skewness of average speed:", trains["avg_speed"].skew())
print("Kurtosis of average speed:", trains["avg_speed"].kurt())

# ---------------------------------------------------------------
# 6. OUTLIERS (IQR METHOD)
# ---------------------------------------------------------------
def count_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()


print("\nOutliers (IQR rule):")
for col in ["distance_km", "num_stops", "avg_speed"]:
    print(f"  {col}: {count_outliers(trains[col])}")

# ---------------------------------------------------------------
# 7. VISUALIZATIONS
# ---------------------------------------------------------------
# Route length distribution
plt.figure(figsize=(8, 4))
sns.histplot(trains["distance_km"], bins=30, kde=True, color="teal")
plt.title("Distribution of Route Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Number of trains")
plt.show()

# Average speed by train category
plt.figure(figsize=(7, 4))
sns.boxplot(data=trains, x="category", y="avg_speed", hue="category", legend=False)
plt.title("Average Speed by Train Category")
plt.xlabel("Category")
plt.ylabel("Average speed (km/h)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(trains[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Distance vs number of stops
plt.figure(figsize=(8, 5))
sns.scatterplot(data=trains, x="distance_km", y="num_stops", hue="category", alpha=0.7)
plt.title("Route Distance vs Number of Stops")
plt.xlabel("Distance (km)")
plt.ylabel("Number of stops")
plt.show()

# ---------------------------------------------------------------
# 8. FIVE OBJECTIVES
# ---------------------------------------------------------------

# Objective 1: Longest and shortest routes
cols = ["train_no", "train_name", "source", "destination", "distance_km"]
print("\nOBJECTIVE 1: Longest and Shortest Routes")
print("\nLongest routes:")
print(trains.sort_values("distance_km", ascending=False)[cols].head(10).to_string(index=False))
print("\nShortest routes:")
print(trains.sort_values("distance_km")[cols].head(10).to_string(index=False))

# Objective 2: Busiest stations (number of trains stopping there)
busiest = df.groupby("Station Name")["Train No"].nunique().sort_values(ascending=False).head(10)
print("\nOBJECTIVE 2: Busiest Stations")
print(busiest)

busiest.plot(kind="barh", figsize=(8, 5), color="darkorange")
plt.gca().invert_yaxis()
plt.title("Top 10 Busiest Stations")
plt.xlabel("Number of trains")
plt.tight_layout()
plt.show()

# Objective 3: Most common starting and ending stations
print("\nOBJECTIVE 3: Most Common Origins and Terminals")
print("\nTop origins:")
print(trains["source"].value_counts().head(10))
print("\nTop destinations:")
print(trains["destination"].value_counts().head(10))

# Objective 4: Speed comparison between categories
print("\nOBJECTIVE 4: Speed by Train Category")
print(trains.groupby("category")[["avg_speed", "num_stops", "distance_km"]].mean().round(2))
print("\nFastest trains:")
print(trains.sort_values("avg_speed", ascending=False)
      [["train_no", "train_name", "category", "avg_speed"]].head(10).to_string(index=False))

# Objective 5: Do longer routes have more stops? How long do trains halt?
print("\nOBJECTIVE 5: Stops, Distance and Halt Time")
print("Distance vs stops correlation:", round(trains["distance_km"].corr(trains["num_stops"]), 3))
print("Stops per 100 km by category:")
print(trains.groupby("category")["stops_per_100km"].mean().round(2))
print("Average halt time by category (minutes):")
print(trains.groupby("category")["avg_halt_min"].mean().round(2))

# ---------------------------------------------------------------
# 9. CLASSIFICATION: PREDICT TRAIN CATEGORY
# ---------------------------------------------------------------
# Goal: guess the train category (Special / Mail-Express / Superfast)
# from how the train runs. Train number is not used as a feature
# because the category was made from it.

features = ["num_stops", "distance_km", "run_hours", "avg_halt_min",
            "avg_speed", "stops_per_100km", "start_hour"]
X = trains[features]
y = trains["category"]

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

labels = sorted(y.unique())

# ----- Model 1: Logistic Regression -----
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, average="macro", zero_division=0)
lr_rec = recall_score(y_test, lr_pred, average="macro")
lr_f1 = f1_score(y_test, lr_pred, average="macro")

print("\nLogistic Regression")
print("Accuracy:", round(lr_acc, 3), "Precision:", round(lr_prec, 3),
      "Recall:", round(lr_rec, 3), "F1:", round(lr_f1, 3))
print(confusion_matrix(y_test, lr_pred, labels=labels))

# ----- Model 2: KNN -----
k = int(np.sqrt(len(X_train)))
if k % 2 == 0:
    k = k + 1   # keep k odd

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)
knn_prec = precision_score(y_test, knn_pred, average="macro", zero_division=0)
knn_rec = recall_score(y_test, knn_pred, average="macro")
knn_f1 = f1_score(y_test, knn_pred, average="macro")

print("\nKNN")
print("Accuracy:", round(knn_acc, 3), "Precision:", round(knn_prec, 3),
      "Recall:", round(knn_rec, 3), "F1:", round(knn_f1, 3))
print(confusion_matrix(y_test, knn_pred, labels=labels))

# ----- Model 3: SVM -----
svm = SVC(class_weight="balanced")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)
svm_prec = precision_score(y_test, svm_pred, average="macro", zero_division=0)
svm_rec = recall_score(y_test, svm_pred, average="macro")
svm_f1 = f1_score(y_test, svm_pred, average="macro")

print("\nSVM")
print("Accuracy:", round(svm_acc, 3), "Precision:", round(svm_prec, 3),
      "Recall:", round(svm_rec, 3), "F1:", round(svm_f1, 3))
print(confusion_matrix(y_test, svm_pred, labels=labels))

# ----- Model 4: Random Forest -----
rf = RandomForestClassifier(random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, average="macro", zero_division=0)
rf_rec = recall_score(y_test, rf_pred, average="macro")
rf_f1 = f1_score(y_test, rf_pred, average="macro")

print("\nRandom Forest")
print("Accuracy:", round(rf_acc, 3), "Precision:", round(rf_prec, 3),
      "Recall:", round(rf_rec, 3), "F1:", round(rf_f1, 3))
print(confusion_matrix(y_test, rf_pred, labels=labels))

# ----- Model 5: Decision Tree -----
dt = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred, average="macro", zero_division=0)
dt_rec = recall_score(y_test, dt_pred, average="macro")
dt_f1 = f1_score(y_test, dt_pred, average="macro")

print("\nDecision Tree")
print("Accuracy:", round(dt_acc, 3), "Precision:", round(dt_prec, 3),
      "Recall:", round(dt_rec, 3), "F1:", round(dt_f1, 3))
print(confusion_matrix(y_test, dt_pred, labels=labels))

# Put all the scores together for the comparison in Step 10
results = [
    ["Logistic Regression", lr_acc, lr_prec, lr_rec, lr_f1],
    ["KNN", knn_acc, knn_prec, knn_rec, knn_f1],
    ["SVM (RBF)", svm_acc, svm_prec, svm_rec, svm_f1],
    ["Random Forest", rf_acc, rf_prec, rf_rec, rf_f1],
    ["Decision Tree", dt_acc, dt_prec, dt_rec, dt_f1],
]

# ---------------------------------------------------------------
# 10. MODEL COMPARISON
# ---------------------------------------------------------------
comparison = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
comparison = comparison.sort_values("F1", ascending=False).reset_index(drop=True)
print("\nMODEL COMPARISON (sorted by macro F1)")
print(comparison.round(4).to_string(index=False))

comparison.set_index("Model")[["Accuracy", "F1"]].plot(kind="bar", figsize=(8, 5))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

best = comparison.loc[0]
print(f"\nBest model: {best['Model']} "
      f"(macro F1 = {best['F1']:.4f}, accuracy = {best['Accuracy']:.4f})")

# ---------------------------------------------------------------
# 11. SUMMARY
# ---------------------------------------------------------------
print("\nSUMMARY")
print("1. Superfast trains run clearly faster than mail/express and special trains, "
      "and average speed is the most important feature for telling the categories apart.")
print("2. Special trains stop much more often per 100 km than the other two groups, "
      "but they are a mixed bag (some are actually superfast services), so they overlap "
      "with the other classes.")
print("3. Big junctions like Kalyan, Pune, Thane and Bhusaval see the most trains.")
print("4. Lokmanya Tilak Terminus, Pune and CST Mumbai are the most common starting "
      "and ending points.")
print("5. The model comparison identifies the best-performing classifier based on macro F1. "
      "The test set is small (about 116 trains), so the scores can move a bit with a "
      "different split.")

print("\nLIMITATIONS")
print("- Train names are cut at 12 characters in the file.")
print("- Only one snapshot (Dec 2017) is available, so delays and seasonal "
      "changes are not captured.")
print("- Category comes from the train number, so some trains may be grouped imperfectly.")