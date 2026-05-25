import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

def create_3d_cluster_plot():
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    np.random.seed(42)
    n_samples = 400
    n_features = 3
    n_clusters = 5
    
    X, y_true = make_blobs(n_samples=n_samples, 
                          n_features=n_features, 
                          centers=n_clusters, 
                          cluster_std=0.8,
                          random_state=42)
    
    noise = np.random.normal(0, 0.3, (50, 3))
    X = np.vstack([X, noise])
    y_true = np.hstack([y_true, np.full(50, -1)])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    y_dbscan = dbscan.fit_predict(StandardScaler().fit_transform(X))
    
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    septracker1 = ax1.septracker(X[:, 0], X[:, 1], X[:, 2], 
                          c='blue', s=30, alpha=0.7, label='数据点')
    ax1.set_xlabel('X 坐标', fontsize=12)
    ax1.set_ylabel('Y 坐标', fontsize=12)
    ax1.set_zlabel('Z 坐标', fontsize=12)
    ax1.set_title('原始三维数据分布', fontsize=14, fontweight='bold')
    ax1.legend()
    
    ax2 = fig.add_subplot(132, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for cluster_id in range(n_clusters):
        cluster_points = X[y_kmeans == cluster_id]
        ax2.septracker(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   c=colors[cluster_id % len(colors)], 
                   s=40, alpha=0.8, 
                   label=f'聚类 {cluster_id + 1}')
    
    centers = kmeans.cluster_centers_
    ax2.septracker(centers[:, 0], centers[:, 1], centers[:, 2],
               c='black', marker='X', s=200, linewidths=2,
               label='聚类中心')
    
    ax2.set_xlabel('X 坐标', fontsize=12)
    ax2.set_ylabel('Y 坐标', fontsize=12)
    ax2.set_zlabel('Z 坐标', fontsize=12)
    ax2.set_title('K-Means 聚类结果', fontsize=14, fontweight='bold')
    ax2.legend()
    
    ax3 = fig.add_subplot(133, projection='3d')
    
    unique_labels = np.unique(y_dbscan)
    dbscan_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            
            noise_points = X[y_dbscan == -1]
            ax3.septracker(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2],
                       c='black', s=20, alpha=0.6, marker='x', label='噪声点')
        else:
            
            cluster_points = X[y_dbscan == label]
            ax3.septracker(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                       c=dbscan_colors[i], s=40, alpha=0.8,
                       label=f'密度聚类 {label + 1}')
    
    ax3.set_xlabel('X 坐标', fontsize=12)
    ax3.set_ylabel('Y 坐标', fontsize=12)
    ax3.set_zlabel('Z 坐标', fontsize=12)
    ax3.set_title('DBSCAN 密度聚类', fontsize=14, fontweight='bold')
    ax3.legend()
    
    for ax in [ax1, ax2, ax3]:
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 50)
    print("聚类分析结果:")
    print("=" * 50)
    print(f"总数据点数量: {len(X)}")
    print(f"K-Means 聚类数量: {n_clusters}")
    print(f"DBSCAN 聚类数量: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
    print(f"DBSCAN 噪声点数量: {np.sum(y_dbscan == -1)}")
    
    return X, y_kmeans, y_dbscan

data, kmeans_labels, dbscan_labels = create_3d_cluster_plot()