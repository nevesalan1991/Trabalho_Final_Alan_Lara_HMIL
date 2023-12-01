# Trabalho_Final_Alan_Lara_HMIL

Instruções para uso do programa flow_test_final:

O programa em Python tem como objetivo realizar a classificação de oito diferentes padrões de escoamento. Ele é dividido em seis blocos distintos:

Pré-processamento dos Sinais
1) Extração de Recursos
2) Redução de Dimensionalidade
3) Divisão de Dados
4) Treinamento do Modelo de Classificação
5) Avaliação do Modelo
   
Para executar o programa, siga as etapas abaixo:

1) Download do programa:
    Baixe o arquivo flow_test_final.py.

2) Preparação dos Dados:
    Na pasta Dados.Mat, baixe os 26 arquivos .rar e extraia a pasta Dados_MAT. Esta pasta contém oito subpastas, cada uma representando um tipo específico de escoamento.

3) Configuração dos Diretórios:
   
    No programa, na seção de pré-processamento, forneça o caminho de cada uma das oito pastas, conforme o exemplo abaixo:

          # Dicionário de pastas com seus respectivos caminhos
          folder_paths = {
              'Estrat_Liso_7nm3h': '/caminho/para/Estrat_Liso_7nm3h',
              'Estrat_ond_16nm3h': '/caminho/para/Estrat_ond_16nm3h',
              'Estrat_ond_24nm3h': '/caminho/para/Estrat_ond_24nm3h',
              'Plug_5nm3h': '/caminho/para/Plug_5nm3h',
              'Plug_10nm3h': '/caminho/para/Plug_10nm3h',
              'Slug_15nm3h': '/caminho/para/Slug_15nm3h',
              'Slug_45nm3h': '/caminho/para/Slug_45nm3h',
              'wavy--slug_55nm3h': '/caminho/para/wavy--slug_55nm3h',
          }


4)  Salvando Espectrogramas:
     Ainda na seção de pré-processamento, forneça o caminho de uma pasta que você criará para salvar os espectrogramas dos diferentes escoamentos, conforme o exemplo abaixo:

            # Iterar sobre o dicionário e chamar a função para cada pasta
              for folder_name, folder_path in folder_paths.items():
                  print(f"Processing folder: {folder_name}")
                  save_folder = os.path.join('/caminho/para/resultado_espectrograma/', folder_name)
                  read_mat_files_and_save(folder_path, save_folder)
   
        

  5) Por fim,  salvando Gráfico PCA:
      Na seção de Redução de Dimensionalidade, forneça o caminho de uma pasta que você criará para salvar o gráfico PCA, conforme o exemplo abaixo:

          # Substituir o diretório por um exemplo hipotético
          save_pca_folder = '/home/usuario/projetos/hipotetico_pca_graficos/'
          
          # Criar o diretório se não existir
          os.makedirs(save_pca_folder, exist_ok=True)
          
          # Salvar o gráfico
          plt.savefig(os.path.join(save_pca_folder, 'pca_plot.png'))


     

