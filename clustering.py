# Importacion de librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Permite que se imprima el dataframe por completo
pd.set_option("display.max_rows", None)

# Se carga el conjunto de datos que se van a usar para hacer clustering
df_original = pd.read_csv("datos/Datos_Covid19_paises.csv", sep = ";", encoding='latin-1')
# Se cambian los valores nulos por -1 para facilitar el tratamiento de los datos
df_original = df_original.replace(np.nan, -1)
# Se genera la matriz de las variables que interesan para el estudio
X = df_original.iloc[:, [5, 8, 9]].values
# Se tratan los datos de la matriz para que estén en forma de integer
for i in range(len(X)):
    population = [None] * len(X)
    for j in range(len(X[i])):
        if X[i][j] != -1:
            if ',' in X[i][j]:
                X[i][j] = int(X[i][j].replace(",", ""))
        numeroHabitantes = df_original.iloc[i, [12]].values
        population[i] = numeroHabitantes[0].replace(",", "")      
        # Cálculo del número total de recuperados por cada millón de habitantes
        if int(population[i]) != 0 and j == 0 and X[i][j] != -1:
            X[i][j] = int(X[i][j]) / int(population[i]) * 1000000
    X[i] = list(map(float, X[i]))

# Para encontrar el número óptimo de clusters, se usa el metodo del codo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 100)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Grafica de la suma de las distancias
'''plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show() # En esta gráfica se muestra que el número óptimo de clusters es 3'''

# Creando el k-Means para los 3 grupos encontrados
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 100)
y_kmeans = kmeans.fit_predict(X) # y_kmeans guarda el grupo al que corresponde cada fila de los datos
print(y_kmeans)


# Visualizacion grafica de los clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'pink')

plt.title('Clusters of customers')
plt.xlabel('Cases')
plt.ylabel('Deaths')
plt.legend()
plt.show()