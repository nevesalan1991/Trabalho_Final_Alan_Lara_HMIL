#!/usr/bin/env python
# coding: utf-8

# In[75]:

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

########################################################################################################################
#######################   1) Pré-processamento   #######################################################################
########################################################################################################################
# Function to plot signals in the time domain
def plot_time_domain_signal(file_name, time, signal_data, class_name, save_folder):
    plt.figure()
    plt.plot(time, signal_data)
    plt.title(f'Time Domain Signal - Class {class_name}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(save_folder, file_name))
    plt.close()


# Crie listas para armazenar os vetores de espectrograma e as classes
vetores_espectrograma = []
classes = []

def padronizar_espectrograma(Sxx):
    novo_shape = (343, 129)  # Defina o novo shape desejado

    # Crie uma grade regular para as coordenadas x e y
    x_vals = np.linspace(0, Sxx.shape[1] - 1, Sxx.shape[1])
    y_vals = np.linspace(0, Sxx.shape[0] - 1, Sxx.shape[0])

    interp_func = RegularGridInterpolator((y_vals, x_vals), Sxx, method='linear', bounds_error=False, fill_value=None)

    # Crie uma nova grade de coordenadas
    new_y_vals = np.linspace(0, Sxx.shape[0] - 1, novo_shape[0])
    new_x_vals = np.linspace(0, Sxx.shape[1] - 1, novo_shape[1])

    # Crie a grade completa de coordenadas para avaliar a função interpoladora
    coords = np.array(np.meshgrid(new_y_vals, new_x_vals, indexing='ij')).T.reshape(-1, 2)

    # Avalie a função interpoladora na nova grade de coordenadas
    Sxx_padronizado = interp_func(coords).reshape(novo_shape)

    return Sxx_padronizado


def read_mat_files_and_save(folder_path, save_folder):
    global vetores_espectrograma, classes

    os.makedirs(save_folder, exist_ok=True)
    #folder_class = len(set(classes)) + 1
    folder_class = len(set(classes)) + 1 if 'folder_class' not in locals() else folder_class
    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    for mat_file in mat_files:
        file_path = os.path.join(folder_path, mat_file)
        mat_data = loadmat(file_path)
        terceira_coluna = mat_data['dados_xlsx'][:, 2]  # Assuming the sound signature column is the third column
        time_column = mat_data['dados_xlsx'][:, 0]      # Assuming the time column is the first column

        # Plot and save the time domain signal
        time_domain_save_folder = os.path.join('C:\\Users\\larac\\OneDrive - puc-rio.br\\DOUTORADO PUC-Rio\\2_Periodo\\Hands-on Machine Learning\\Projeto Final\\Flow Pattern Classification\\dados_mat\\Time_Domain_Plots', folder_name)
        os.makedirs(time_domain_save_folder, exist_ok=True)
        # Extract the file name from the mat file without extension
        base_file_name = os.path.splitext(os.path.basename(mat_file))[0]
        file_name = f"time_domain_signal_{base_file_name}.png"
        plot_time_domain_signal(file_name, time_column, terceira_coluna, folder_name, time_domain_save_folder)


        fs = 1000               #sampling frequency (fs)
        nperseg = 256           #512
        noverlap = nperseg // 2
        window_type = 'hann'    # Specify the window type (e.g., 'hann', 'hamming', 'triang', etc.)

        f, _, Sxx = signal.spectrogram(terceira_coluna, fs, nperseg=nperseg, noverlap=noverlap,window=window_type)
        t = time_column

        # Padronizar o espectrograma
        Sxx_padronizado = padronizar_espectrograma(Sxx)

        #print("Dimensões de t:", t.shape)
        # print("Dimensões de f:", f.shape)
        # print("Dimensões de Sxx_padronizado:", Sxx_padronizado.shape)

        # Antes de chamar plt.pcolormesh, ajuste as dimensões de t e f
        t = np.linspace(0, Sxx.shape[1] - 1, Sxx.shape[1])  # Certifique-se de ajustar o intervalo conforme necessário
        f = np.linspace(0, Sxx.shape[0] - 1, Sxx.shape[0])  # Certifique-se de ajustar o intervalo conforme necessário

        # Certifique-se de que t e f tenham uma dimensão a menos do que Sxx_padronizado
        t, f = np.meshgrid(t, f, indexing='ij')

        # Plotagem do espectrograma
        plt.figure()
        plt.pcolormesh(t, f,10 * np.log10(Sxx_padronizado), shading='auto')
        plt.colorbar(label='Potência (dB)')
        plt.ylabel('Frequência (Hz)')
        plt.xlabel('Tempo (s)')
        plt.title(f'Espectrograma - Classe {folder_class}')

        # Salva o espectrograma plotado como imagem
        save_path = os.path.join(save_folder, f"espectrograma_{mat_file.replace('.mat', '.png')}")
        plt.savefig(save_path)
        plt.close()

        # Adiciona às listas globais
        #vetores_espectrograma.append(Sxx_padronizado.flatten())
        vetores_espectrograma.append(Sxx.ravel())
        #classes.append(folder_class)
        
        # Calcula estatísticas descritivas no Sxx_padronizado
        mean_value = np.mean(Sxx_padronizado)
        std_dev = np.std(Sxx_padronizado)
        max_value = np.max(Sxx_padronizado)
        min_value = np.min(Sxx_padronizado)


        # Adiciona as estatísticas descritivas às listas globais
        estatisticas_descritivas = [mean_value, std_dev, max_value, min_value]
        #vetores_espectrograma.append(estatisticas_descritivas)
        classes.append(folder_class)

    print(f"Processed folder: {folder_path}, Class: {folder_class}")
    # print(f"Class: {folder_class}")

# In[]:

# Dicionário de pastas com seus respectivos caminhos
folder_paths = {
    'Estrat_Liso_7nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Estrat_Liso_7nm3h',
    'Estrat_ond_16nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Estrat_ond_16nm3h',
    'Estrat_ond_24nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Estrat_ond_24nm3h',
    'Plug_5nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Plug_5nm3h',
    'Plug_10nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Plug_10nm3h',
    'Slug_15nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Slug_15nm3h',
    'Slug_45nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/Slug_45nm3h',
    'wavy--slug_55nm3h': 'C:/Users/larac/OneDrive - puc-rio.br/DOUTORADO PUC-Rio/2_Periodo/Hands-on Machine Learning/Projeto Final/Flow Pattern Classification/dados_mat/wavy--slug_55nm3h',
}

# Iterar sobre o dicionário e chamar a função para cada pasta
for folder_name, folder_path in folder_paths.items():
    print(f"Processing folder: {folder_name}")
    save_folder = os.path.join('C:\\Users\\larac\\OneDrive - puc-rio.br\\DOUTORADO PUC-Rio\\2_Periodo\\Hands-on Machine Learning\\Projeto Final\\Flow Pattern Classification\\dados_mat\\Espectograma', folder_name)
    read_mat_files_and_save(folder_path, save_folder)

########################################################################################################################
#######################  2) Extração dos dados #########################################################################
########################################################################################################################
vetores_espectrograma = [arr.flatten() if isinstance(arr, np.ndarray) and len(arr.shape) > 1 else arr for arr in vetores_espectrograma if not isinstance(arr, list)]

# Removendo elementos não-array da lista
vetores_espectrograma = [arr for arr in vetores_espectrograma if isinstance(arr, np.ndarray)]

# Converter a lista de vetores e classes em matrizes
matriz_espectrograma = np.array(vetores_espectrograma)
vetor_classes = np.array(classes)

# Imprima a forma das matrizes
print('Matriz de Espectrograma:', matriz_espectrograma.shape)
print('Vetor de Classes:', vetor_classes.shape)

########################################################################################################################
####################### 3) Redução de Dimensionalidade #################################################################
########################################################################################################################

# Aplicar PCA na Matriz de Espectrograma
num_componentes = 0.95  # Especifica diretamente o número de componentes
pca = PCA(n_components=num_componentes)
matriz_reduzida = pca.fit_transform(matriz_espectrograma)

# Imprimir a nova forma da matriz após a redução
print(f'Matriz Reduzida após PCA ({num_componentes} componentes):', matriz_reduzida.shape)

# Criar um DataFrame para plotar o gráfico dispersão
pca_df = pd.DataFrame(matriz_reduzida, columns=[f'Componente Principal {i + 1}' for i in range(matriz_reduzida.shape[1])])
pca_df['Classe'] = np.repeat([folder_name for folder_name in folder_paths.keys()], [len(os.listdir(folder_paths[folder_name])) for folder_name in folder_paths.keys()])

# Plotar o gráfico de dispersão
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='Componente Principal 1', y='Componente Principal 2', hue='Classe', palette='colorful', s=80)
plt.title('PCA - Gráfico de Dispersão com Legendas')

# Salvar o gráfico
save_pca_folder = 'C:\\Users\\larac\\OneDrive - puc-rio.br\\DOUTORADO PUC-Rio\\2_Periodo\\Hands-on Machine Learning\\Projeto Final\\Flow Pattern Classification\\dados_mat\\PCA_Graficos'
os.makedirs(save_pca_folder, exist_ok=True)
plt.savefig(os.path.join(save_pca_folder, 'pca_plot.png'))

''''''''''''
'PCA 3D'
''''''''''''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Suponha que você já tenha a matriz reduzida após o PCA (matriz_reduzida)
# Substitua isso pelos seus dados reais.

# Criar dados de exemplo
np.random.seed(42)
data = np.random.rand(100, 10)  # 100 amostras, 10 recursos

# Aplicar PCA
num_componentes = 3
pca = PCA(n_components=num_componentes)
matriz_reduzida = pca.fit_transform(data)

# Criar DataFrame para plotar o gráfico
pca_df = pd.DataFrame(matriz_reduzida, columns=[f'Componente Principal {i + 1}' for i in range(num_componentes)])

# Plotar o gráfico de dispersão 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_df['Componente Principal 1'], pca_df['Componente Principal 2'], pca_df['Componente Principal 3'], c='blue', marker='o')

# Configurações do gráfico
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('PCA - Gráfico de Dispersão 3D')

# Exibir o gráfico
plt.show()

########################################################################################################################
########################## 4) Divisão dos dados ########################################################################
########################################################################################################################

X_train, X_test, y_train, y_test = train_test_split(matriz_reduzida, vetor_classes, test_size=0.4, random_state=42)

########################################################################################################################
########################## 5) Treinamento de Classificação  ############################################################
########################################################################################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Definir modelos e parâmetros
modelos_parametros = [
    (KNeighborsClassifier(n_neighbors=5), 'KNN'),
    (SVC(kernel='linear'), 'SVM'),
    (RandomForestClassifier(n_estimators=100, random_state=42), 'Random Forest')
]

# Listas para armazenar os resultados
acuracias = []
matrizes_confusao = []

# Loop sobre modelos
for modelo, nome_modelo in modelos_parametros:
    # Treinar modelo
    modelo.fit(X_train, y_train)

    # Prever com o conjunto de teste
    predictions = modelo.predict(X_test)

    # Avaliar o desempenho
    accuracy = accuracy_score(y_test, predictions)
    acuracias.append((nome_modelo, accuracy))

    # Classification Report
    print(f"{nome_modelo} Accuracy: {accuracy}")
    print(f"Classification Report ({nome_modelo}):\n", classification_report(y_test, predictions,zero_division=1))

    # Confusion Matrix
    matriz_confusao = confusion_matrix(y_test, predictions)
    matrizes_confusao.append((nome_modelo, matriz_confusao))

    # Plotar Confusion Matrix
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{nome_modelo} Confusion Matrix')
    plt.show()

# Exibir tabela de acurácias
df_acuracia = pd.DataFrame(acuracias, columns=['Modelo', 'Acurácia'])
print(df_acuracia)

# # Exibir tabela com as matrizes de confusão
# for nome_modelo, matriz_confusao in matrizes_confusao:
#     print(f"\nMatriz de Confusão ({nome_modelo}):\n", matriz_confusao)

########################################################################################################################
#################### 6) Avaliação do Modelo ############################################################################
########################################################################################################################
'''
# Testes dos Hiperparâmetros
'''
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Lista de modelos e seus respectivos parâmetros
modelos_parametros = [
    (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}),
    (SVC(), {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto']}),
    (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]})
]

# Lista para armazenar os resultados
resultados = []

# Loop sobre modelos e parâmetros
for modelo, parametros in modelos_parametros:
    grid_search = GridSearchCV(modelo, parametros, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

     # Avaliar desempenho no conjunto de teste
    predictions = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Matriz de Confusão
    matriz_confusao = confusion_matrix(y_test, predictions)

    # Plotar Matriz de Confusão
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{type(modelo).__name__} Confusion Matrix')
    plt.show()

   # Armazenar resultados
    resultados.append({
        'Modelo': type(modelo).__name__,
        'Melhores Hiperparâmetros': grid_search.best_params_,
        'Melhor Precisão': grid_search.best_score_,
        'Acurácia no Teste': accuracy
    })

# Criar DataFrame com os resultados
df_resultados = pd.DataFrame(resultados)

# Exibir a tabela
print(df_resultados)
plt.show(df_resultados)
