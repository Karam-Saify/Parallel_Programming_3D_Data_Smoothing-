import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('TrafficVolumeData.csv')

# 1. Time Dimension
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour

# 2. PCA for Weather Index
weather_cols = ['air_pollution_index', 'humidity', 'wind_speed', 'wind_direction', 
                'visibility_in_miles', 'dew_point', 'temperature', 'rain_p_h', 
                'snow_p_h', 'clouds_all']
X = df[weather_cols]
X_scaled = StandardScaler().fit_transform(X)
df['PC1'] = PCA(n_components=1).fit_transform(X_scaled)

# 3. Discretization into M=200 bins
M = 200 
df['weather_bin'] = pd.cut(df['PC1'], bins=M, labels=False)

# 4. Grid Construction Z(x,y)
grid_z = df.groupby(['hour', 'weather_bin'])['traffic_volume'].mean().unstack(fill_value=0)
grid_z = grid_z.reindex(index=range(24), columns=range(M), fill_value=0)

grid_z.to_csv('grid_z.csv', index=False, header=False)
print("Input 'grid_z.csv' generated successfully.")
