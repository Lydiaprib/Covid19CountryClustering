# Importacion de librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import plotly
import plotly.graph_objs as go

def traza(x, y, z, c, label, s=2):
    puntosTraza = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
    name=label
    )
    return puntosTraza;

def mostrarGrafica(title, x_colname, x_range, y_colname, y_range, z_colname, z_range, trazas):
    layout = go.Layout(
        title=title,
        scene = dict(
            xaxis=dict(title=x_colname, range = x_range),
            yaxis=dict(title=y_colname, range = y_range),
            zaxis=dict(title=z_colname, range = z_range)
        )
    )

    fig = go.Figure(data=trazas, layout=layout)
    plotly.offline.plot(fig)

# Permite que se imprima el dataframe por completo
pd.set_option("display.max_rows", None)

# Se carga el conjunto de datos que se van a usar para hacer clustering
df_original = pd.read_csv("datos/Datos_Covid19_paises.csv", sep = ";", encoding='latin-1')
# Se cambian los valores nulos por -1 para facilitar el tratamiento de los datos
df_original = df_original.replace(np.nan, -1)
# Se genera la matriz de las variables que interesan para el estudio
arrayVariables = df_original.iloc[:, [5, 8, 9]].values
# Se tratan los datos de la matriz para que estén en forma de integer
for i in range(len(arrayVariables)):
    population = [None] * len(arrayVariables)
    for j in range(len(arrayVariables[i])):
        if arrayVariables[i][j] != -1:
            if ',' in arrayVariables[i][j]:
                arrayVariables[i][j] = int(arrayVariables[i][j].replace(",", ""))
        numeroHabitantes = df_original.iloc[i, [12]].values
        population[i] = numeroHabitantes[0].replace(",", "")      
        # Cálculo del número total de recuperados por cada millón de habitantes
        if int(population[i]) != 0 and j == 0 and arrayVariables[i][j] != -1:
            arrayVariables[i][j] = int(arrayVariables[i][j]) / int(population[i]) * 1000000
    arrayVariables[i] = list(map(float, arrayVariables[i]))


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 100)
    kmeans.fit(arrayVariables)
    wcss.append(kmeans.inertia_)

# Grafica de la suma de las distancias
# Para encontrar el número óptimo de clusters, se usa el metodo del codo
plt.plot(range(1, 11), wcss)
plt.title('Método codo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show() # En esta gráfica se muestra que el número óptimo de clusters es 3


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 100)
kmeans.fit(arrayVariables)
centroides = kmeans.cluster_centers_
y_kmeans = kmeans.fit_predict(arrayVariables)

# Se pinta cada uno de los clusters obtenidos en una gráfica 3D junto con sus centroides
centroids = traza(centroides[:, 0], centroides[:, 1], centroides[:, 2], s= 8, c = 'yellow', label='Centroides')
cluster1 = traza(arrayVariables[y_kmeans == 0, 0], arrayVariables[y_kmeans == 0, 1], arrayVariables[y_kmeans == 0, 2], s= 4, c='red', label = 'Cluster 1') #match with red=1 initial class
cluster2 = traza(arrayVariables[y_kmeans == 1, 0], arrayVariables[y_kmeans == 1, 1], arrayVariables[y_kmeans == 1, 2], s= 4, c='black', label = 'Cluster 2') #match with black=3 initial class
cluster3 = traza(arrayVariables[y_kmeans == 2, 0], arrayVariables[y_kmeans == 2, 1], arrayVariables[y_kmeans == 2, 2], s= 4, c='blue', label = 'Cluster 3') #match with blue=2 initial class

# Se establecen los ejes acordes a las variables de la matriz
x=arrayVariables[:,0]
y=arrayVariables[:,1]
z=arrayVariables[:,2]

# Se muestra la gráfica
mostrarGrafica("K-Medias", "Número Recuperados", [min(x),max(x)], "Casos totales", [min(y),max(y)], "Muertes por millón de habitantes", [min(z)-1,max(z)], [t1,t2,t3,centroids])

# Visualizacion grafica de los clusters
plt.scatter(arrayVariables[:, 0], arrayVariables[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'pink', label = "Centroides")

plt.title('Clusters of customers')
plt.xlabel('Cases')
plt.ylabel('Deaths')
plt.legend()
plt.show()