# Trabalho de Conclusão de Curso - REDES NEURAIS SIAMESAS LSTM PARA DETERMINAÇÃO DE SIMILARIDADES ENTRE PARES DE SENTENÇAS LITERÁRIAS

Trabalho de Conclusão de Curso (TCC) como requisito para obtenção do título de Bacharel em Análise e Desenvolvimento de Sistemas pelo IFES, Campus Serra.

### Informações gerais
- **Autor**: Harã Heique dos Santos
- **Orientador**: Fidelis Zanetti de Castro
- **Linguagem de programação**: Python (versão 3.6.8+)
- **Ambiente de desenvolvimento**: Visual Studio Code (versão 1.35.1+) e PyCharm (versão 2019.2.2+)

As **Redes Neurais** são representados como modelos computacionais contendo nós interconectados que funcionam 
semelhante aos neurônios de um cérebro humano, sendo estes capazes, com o auxílio do aprendizado de máquina, no 
reconhecimento de padrões, correlações, agrupamentos e classificação entre dados através de algoritmos.

Como é mencionado pela [*SAS*](https://www.sas.com/en_us/insights.html), as Redes Neurais são comumente 
utilizadas para resolver problemas complexos do dia a dia, consequentemente auxiliando em melhores tomadas de decisões, 
onde dentre as suas aplicações encontradas no mercado são:

* Otimização de Logística em sistemas de transporte;
* Previsão de demanda de energia e carga elétrica;
* Reconhecimento de voz;
* Detecção de fraudes na área financeira, literária e afins;
* Entre outras.

Baseado nisto este trabalho de conclusão de curso tem como intuito analisar a performance de uma rede neural siamesa 
para classificação de textos literários usando medidas de similaridade.

### Rede Neural Siamesa

As **Redes Neurais Siamesas** são classificadas como redes computacionais que possuem como característica duas ou mais sub-redes idênticas. São 
comumente utilizadas em tarefas de similaridades entre diferentes instâncias, e aplicadas em classificações e 
verificações de autenticidade, tal como é um dos pontos abordados na monografia do trabalho.

A figura abaixo mostra uma arquitetura padrão de uma Rede Siamesa, em que as duas subredes são ligadas por camadas
compartilhando pesos entre si. Tanto no treinamento quanto nos testes de predição, a rede é alimentada por pares de 
entrada no intuito de aproximar pares de uma mesma classe e distanciar pares de classes diferentes. Esta métrica de
distanciamento é definida por uma medida de similaridade, que comumente geram valores de saída, como no exemplo do
trabalho que são valores de 0 a 1, em que quanto mais próximo de 1 maior é a similaridade entre os pares. Caso contrário
mais distante são eles, implicando que tendem a serem assimilares.  

<p align="center">
    <img src="./docs/images/arquitetura_rede_padrao_siamesa.png" alt="arquitetura-RNS" title="Arquitetura Rede Neural Siamesa"/>
</p>

Já na figura abaixo demonstra o fluxo de funcionamento principal da Rede Neural Siamesa utilizada neste trabalho de conclusão, onde na primeira etapa é basicamente realizada a captura dos dados, que neste caso são os textos literários de autores renomados da literatura inglesa. Em sequência são selecionadas sentenças de textos das obras que passam por um pré-processamento, realizando assim filtragem e normalização dos dados. Cada uma das frases selecionadas são armazenadas e transformadas em linhas em arquivos do formato _comma-separated values (CSV)_ no processo de estruturação dos dados.

Na próxima etapa é realizada o processo de normalização e preparação dos dados de entrada lidas de cada frase dos arquivos CSV, com a criação dos vetores de índices, dos _embeddings_ das palavras, definição de hiperparâmetros da rede e separação dos dados de treinamento e validação das sub-redes, que são alimentadas na próxima etapa por pares de entradas. Após isto a rede realiza o processamento dos dados nas camadas escondidas, aprendendo as _features_ dos estilos de escritas dos autores selecionados, sendo considerada uma etapa fundamental tanto no processo de treinamento quanto no de predição. Ambos os modelos utilizados das subredes são idênticos, podendo ser ambas CNNs ou LSTMs, as quais são previamente configuradas e modeladas.

Por fim, as saídas de ambas subredes, que são representações vetoriais das entradas, são submetidas por uma medida de similaridade, que é uma função matemática responsável por realizar o processo de _merge_ entre as duas subredes utilizadas. O resultado de saída da função determina o quão próximo são as saídas por meio de um valor numérico entre 0 e 1 (intervalo fechado) chamado índice de similaridade, onde quanto mais próximo de 1 maior a semelhança entre os diferentes pares de entrada, caso contrário maior a tendência de serem assimilares. As medidas de similaridade aplicadas para análise da rede siamesa no trabalho em questão são: distância de Manhattan, distância Euclidiana e similaridade por Cosseno.

<p align="center">
    <img src="./docs/images/etapas-processo-RNS.png" alt="processo-RNS" title="Fluxo de funcionamento principal da Rede Neural Siamesa"/>
</p>

### Descrição geral
A estrutura da aplicação está definida da seguinte maneira:

```
TCC-rede-neural-siamesa
    |_ docs
       |_ *arquivos de documentação*
    |_ src
       |_ core
          |_ data_structuring.py
          |_ experiments.py
          |_ helper.py
          |_ prediction.py
          |_ similarity_measure.py
          |_ training.py
       |_ data
          |_ prediction
             |_ *arquivos CSV de predição*
          |_ processed
             |_ *arquivos pré-processados e modelos pré-treinados*
          |_ training
             |_ *arquivos CSV de treinamento*
          |_ word_embeddings
             |_ *modelos de word embeddings pré-treinados*
          |_ works
             *obras literárias em texto bruto para treinamento e predição*
          |_ training_variables.txt
       |_ enums
          |_ DatasetType.py
          |_ NeuralNetworkType.py
          |_ SimilarityMeasureType.py
          |_ Stage.py
          |_ WordEmbeddingType.py
       |_ hyperparameters_optimization
          |_ hyperas_optimization.ipynb
          |_ hyperas_optimization_with_weights.ipynb
       |_ models
          |_ ManhattanDistance.py
       |_ results
          |_ *arquivos de resultados e análises*
       |_ user_interface
          |_ cli_input.py
          |_ cli_output.py
    |_ main.py
    |_ README.md
    |_ dependencies.sh
    |_ requirements.txt
```

#### Descrição geral dos arquivos
Descrição geral dos principais arquivos contidos nesta aplicação:

Arquivo|Path|Descrição
---|---|---
**ManhattanDistance.py**|src/models/ManhattanDistance.py|Classe que representa a medida/função de similaridade da distância de Manhattan. São necessárias pois servem para realizar o processo de merge da saída das subredes siamesas.
**Stage.py**|src/enums/Stage.py|É a classe enumerada em que seus valores determinam qual dos estágios o usuário deseja executar da rede no processo de interação com a interface. Os valores são: NONE(0), TRAINING(1) e PREDICTION(2), os quais suas nomenclaturas são auto-explicativas.
**NeuralNetworkType.py**|src/enums/NeuralNetworkType.py|É a classe enumerada em que seus valores determinam quais serão os tipos das subredes siamesas ao criar os modelos artificiais internos, o qual no trabalho pode ser: CNN ou LSTM.
**DatasetType.py**|src/enums/DatasetType.py|É uma classe enumerada que determina qual é o tipo do dataset em relação ao seu pré-processamento realizado, podendo ser cru, sem stopwords ou sem stopwords e lematizado.
**SimilarityMeasureType.py**|src/enums/SimilarityMeasureType.py|É a classe enumerada em que seus valores determinam qual será a medida de similaridade utilizada saída da rede neural.
**WordEmbeddingType.py**|src/enums/WordEmbeddingType.py|É a classe enumerada em que seus valores determinam qual é o word embedding utilizado para a criação da incorporação que representará as palavras.
**helper.py**|src/core/helper.py|É o módulo responsável por conter variáveis e funções auxiliares para os módulos principais da aplicação: *data_structuring.py*, *prediction.py* e *training.py*.
**data_structuring.py**|src/core/data_structuring.py|É o módulo que contém um conjunto de funções responsáveis por ler os dados brutos das obras literárias, aplicar o pré-processamento de filtragem e normalização e preparar os datasets de treinamento e predição da rede.
**experiments.py**|src/core/experiments.py|É o módulo que contém um conjunto de funções responsáveis por realizar o conjunto de experimentos para as N execuções determinadas no escopo do TCC. Os resultados desses experimentos podem ser encontrados no diretório /results.
**training.py**|src/core/prediction.py|É o módulo que contém funções para realização de todo o processo de treinamento da rede neural, ou seja, pré-processamento dos dados, criação da matrix incorporada, normalização/preparação dos dados, criação do modelo com suas camadas (criação da rede neural siamesa com uma arquitetura e medida de similaridade previamente escolhidas), execução do treinamento e seus resultados.
**prediction.py**|src/core/training.py|Este módulo contém funções para realização do processo de predição dado um conjunto de dados de entrada na rede previamente treinada, determinando assim o indíce de similiridade existente entre pares de entredas distintos.
**similarity_measure.py**|src/core/similarity_measure.py|Este módulo contém funções responsáveis por efetuar o cálculo de índice de similaridade na camada de merge da rede neural. Ela pode ser baseada em: distância de Manhattan, distância Euclidiana e similaridade de Cosseno.
**cli_input.py**|src/user_interface/cli_input.py|É um módulo que interage com o usuário fazendo o papel de receber, tratar e validar as entradas de informações requeridas pelo usuário.
**cli_output.py**|src/user_interface/cli_output.py|É um módulo que também interage com o usuário, mas com o papel de mostrar os dados e informações de saída, tais como mensagens, limpeza do prompt, quebras de linhas para mCampuselhor formatação e afins.
**main.py**|src/main.py|É o módulo principal (bootstrap) da aplicação, ou seja, contém a execução princpal e coordena as chamadas de todos os módulos e classes pertencentes.
**training_variables.txt**|src/data/training_variables.txt|Este arquivo guarda os principais paramêtros e hiperparâmetros definido no modelo da rede neural.
**arquivos CSV de predição**|src/data/prediction|São os arquivos, em geral no formato *.csv*, que contêm frases compostas de obras literárias dos autores diferentes das frases extraídas nos datasets de treinamento.
**arquivos CSV de treinamento**|src/data/training|São os arquivos, em geral no formato *.csv*, que contêm frases compostas de obras utilizadas na fase de treinamento da rede.
**arquivos de pré-processados**|src/data/processed|São arquivos já pré-processados com intuito de melhorar a performance nas etapas de treinamento e testes da rede siamesa implementada.
**arquivos de word embeddings**|src/data/word_embeddings|Sâo os arquivos de modelos de word embeddings pré-treinados.
**arquivos de obras literárias**|src/data/works|Sâo arquivos com as obras dos autores no formato de texto (txt).


### Como executar?
Para executar a aplicação no ambiente Linux, o qual é o principal utilizado, basta seguir os seguintes passos:

1. Primeiramente é necessário conter o interpretador do Python 3. Para baixar e instalar siga este 
[link](https://www.python.org/downloads/), onde pode ser encontrado a última versão estável;

2. O ideal é que antes de executar a **etapa 3** é recomendável realizar a instalação das dependências em um ambiente 
virtual do Python, pois ele empacota todas as dependências que um projeto precisa e armazena em um 
diretório, isolando-o do SO base. A seguir um [link](https://pythonacademy.com.br/blog/python-e-virtualenv-como-programar-em-ambientes-virtuais) 
explicando em detalhes o que são e o passo a passo de como instalar e utilizar;

3. Abra o terminal bash dentro do ambiente virtual e execute o seguinte comando para instalar as dependências da 
aplicação:

        $ sh dependencies.sh

4. Após instalar as dependências ao projeto é necessário baixar o word2vec no [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) 
e colocá-lo no diretório **/src/data/word_embeddings**;

5. Feito isto basta abrir o CLI (Command Line Interface) no diretório **/src** da aplicação e 
executar o seguinte comando para inicializar a aplicação:

        $ python3 main.py

6. Por fim basta interagir com a interface de linha de comando escolhendo as opções fornecidas pela aplicação, sendo as
principais de **treinamento** e **predição**.


### Informações adicionais
Todo o código fonte está hospedado no [GitHub](https://github.com/HaraHeique/TCC-Rede-Neural-Siamesa).
