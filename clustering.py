# Importacion de librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import plotly
import plotly.graph_objs as go
import math

# Permite que se imprima el dataframe por completo
pd.set_option("display.max_rows", None)

def traza(x, y, z, c, label, s=2):
    puntosTraza = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
    name=label
    )
    return puntosTraza

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

# Se tratan los datos de la matriz para que estén en el formato necesario
def formato_datos():
    population = [None] * len(X)
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] != -1:
                X[i][j] = float(str(X[i][j]).replace(",", ""))
            numeroHabitantes = df_original.iloc[i, [12]].values
            population[i] = numeroHabitantes[0].replace(",", "")      
            # Cálculo del número total de recuperados por cada millón de habitantes
            if int(population[i]) != 0 and j == 0 and X[i][j] != -1:
                X[i][j] = int(X[i][j]) / int(population[i]) * 1000000
            X[i][j] = round(X[i][j] + 0.49)
        X[i] = list(map(float, X[i]))
    #print(X)

# Seleccion de numero optimo de clusters siguiendo el método del codo
def num_optimo_clusters():
    inercia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 100)
        kmeans.fit(X)
        inercia.append(kmeans.inertia_)

    # Grafica de la suma de las distancias
    # Para encontrar el número óptimo de clusters, se usa el metodo del codo
    plt.plot(range(1, 11), inercia)
    plt.title('Método del codo')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inercia')
    plt.show()

# Aplicacion del algoritmo k-medias para un numero especifico de clusters
def K_medias(clusters):
    kmeans = KMeans(n_clusters = clusters, init = 'k-means++', random_state = 100)
    kmeans.fit(X)
    centroides = kmeans.cluster_centers_
    y_kmeans = kmeans.fit_predict(X)
    
    return kmeans, centroides, y_kmeans

# Se crea un dataframe que relaciona los paises con el cluster al que pertenecen
def relacion_paises_cluster(paises):
    columnas = ["Paises", "Nº Cluster", "Variables"]
    datos = []
    relacionPaisesCluster = pd.DataFrame(columns = columnas)

    for i in range(len(paises)):
        valores = [paises[i].tolist(), y_kmeans[i], X[i]]
        zipped = zip(columnas, valores)
        diccionario = dict(zipped)
        datos.append(diccionario)

    relacionPaisesCluster = relacionPaisesCluster.append(datos, True)
    return relacionPaisesCluster

# Se carga el conjunto de datos que se van a usar para hacer clustering
df_original = pd.read_csv("datos/Datos_Covid19_paises.csv", sep = ";", encoding='latin-1')
# Se cambian los valores nulos por -1 para facilitar el tratamiento de los datos
df_original = df_original.replace(np.nan, 0)



''' Inicio del analisis inicial '''

# Se genera la matriz de las variables que interesan para el estudio
X = df_original.iloc[:, [5, 8, 9]].values
# Se genera la matriz con únicamente los nombres de los paises
paises_original = df_original.iloc[:, [0]].values
formato_datos()
num_optimo_clusters()
kmeans, centroides, y_kmeans = K_medias(3)
relacionPaisesCluster = relacion_paises_cluster(paises_original)
print("Agrupación de paises en el análisis inicial:")
print(relacionPaisesCluster)
print("")

# Se pinta cada uno de los clusters obtenidos en una gráfica 3D junto con sus centroides
centroids = traza(centroides[:, 0], centroides[:, 1], centroides[:, 2], s= 8, c = "silver", label = "Centroides")
cluster1 = traza(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s= 4, c='red', label = 'Cluster 0')
cluster2 = traza(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s= 4, c='green', label = 'Cluster 1')
cluster3 = traza(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s= 4, c='blue', label = 'Cluster 2')

# Se establecen los ejes acordes a las variables de la matriz
x=X[:,0] # Casos totales
y=X[:,1] # Numero de recuperados
z=X[:,2] # Fallecimientos

# Se muestra la gráfica
mostrarGrafica("K-Medias", "Número Recuperados / M", [min(x),max(x)], "Casos totales / M", [min(y),max(y)], "Fallecimientos / M", [min(z)-1,max(z)], [cluster1, cluster2, cluster3, centroids])

''' Fin del analisis inicial '''



''' Inicio del analisis sin outliers '''

# Extraccion de los outliers detectados
elementosCluster = relacionPaisesCluster["Nº Cluster"].value_counts()
print("Número de elementos que presenta cada cluster (3 clusters):")
print(elementosCluster) # Así se ve mas facilmente que el cluster 2 esta formado por 4 outliers que hay que eleminar
print("")
filasCluster = relacionPaisesCluster[relacionPaisesCluster["Nº Cluster"] == 1]
print("Países y variables que son outliers:")
print(filasCluster[["Paises", "Variables"]]) # Asi se detectan las filas en las que se encuentran los outliers
print("")

# Se eliminan las filas de los outliers
df_no_outliers = df_original.drop([18, 118, 125, 199])
df_no_outliers.reset_index(drop=True, inplace=True) # Se regeneran los indices para mayor comodidad
# Se vuelve a generar la matriz de las variables que interesan para el estudio
X = df_no_outliers.iloc[:, [5, 8, 9]].values
formato_datos()
num_optimo_clusters()
kmeans, centroides, y_kmeans = K_medias(2)
paises_outliers = df_no_outliers.iloc[:, [0]].values
relacionPaisesCluster = relacion_paises_cluster(paises_outliers)
print("Agrupación de paises en el análisis sin outliers:")
print(relacionPaisesCluster)
print("")
elementosCluster = relacionPaisesCluster["Nº Cluster"].value_counts()
print("Número de elementos que presenta cada cluster (2 clusters):")
print(elementosCluster)
print("")

# Se pinta cada uno de los clusters obtenidos en una gráfica 3D junto con sus centroides
centroids = traza(centroides[:, 0], centroides[:, 1], centroides[:, 2], s= 8, c = "silver", label = "Centroides")
cluster1 = traza(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s= 4, c='red', label = 'Cluster 1')
cluster2 = traza(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s= 4, c='green', label = 'Cluster 2')

# Se establecen los ejes acordes a las variables de la matriz
x=X[:,0] # Casos totales
y=X[:,1] # Numero de recuperados
z=X[:,2] # Fallecimientos

# Se muestra la gráfica
mostrarGrafica("K-Medias sin outliers", "Número Recuperados / M", [min(x),max(x)], "Casos totales / M", [min(y),max(y)], "Fallecimientos / M", [min(z)-1,max(z)], [cluster1, cluster2, centroids])

''' Fin del analisis sin outliers '''