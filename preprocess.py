import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

M = 200

df = pd.read_csv("TrafficVolumeData.csv")

df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
df = df.dropna(subset=["date_time"]).copy()

df["hour"] = df["date_time"].dt.hour

weather_cols = [
    "air_pollution_index",
    "humidity",
    "wind_speed",
    "wind_direction",
    "visibility_in_miles",
    "dew_point",
    "temperature",
    "rain_p_h",
    "snow_p_h",
    "clouds_all"
]

df = df.dropna(subset=weather_cols + ["traffic_volume"]).copy()

X = df[weather_cols]
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=1)
df["PC1"] = pca.fit_transform(X_scaled)

df["weather_bin"] = pd.cut(df["PC1"], bins=M, labels=False, include_lowest=True)

grid_z = (
    df.groupby(["hour", "weather_bin"])["traffic_volume"]
      .mean()
      .unstack(fill_value=0)
)

grid_z = grid_z.reindex(index=range(24), columns=range(M), fill_value=0)

grid_z.to_csv("grid_z.csv", index=False, header=False)

print("Generated grid_z.csv successfully")
print(f"Grid shape: {grid_z.shape}")
print(f"Explained variance ratio of PC1: {pca.explained_variance_ratio_[0]:.6f}")
