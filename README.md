# Trabalho de Conclusão de Curso - Similaridades de Estilos Literários Baseadas em Aprendizado Profundo

Trabalho Conclusão referente ao curso de graduação de Bacharelado de Sistema de Informação do IFES - Serra.

### Informações gerais
- **Autor**: Harã Heique
- **Orientador**: Fidelis Castro
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

### Descrição geral
A estrutura da aplicação está definida da seguinte maneira:

```
TCC-rede-neural-siamesa
    |_ docs
       |_ *arquivos de documentação*
    |_ src
       |_ core
          |_ helper.py
          |_ prediction.py 
          |_ training.py
       |_ data
          |_ prediction
             |_ *arquivos de entrada de predição*
          |_ training
             |_ *arquivos de entrada de treinamento*
          |_ GoogleNews-vectors-negative300.bin.gz
          |_ *arquivos de modelos da rede neural*.h5
       |_ enums
          |_ Stage.py
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
```

#### Descrição geral dos arquivos
Descrição geral dos principais arquivos contidos nesta aplicação:

Arquivo|Path|Descrição
---|---|---
**ManhattanDistance.py**|src/models/ExecutionType.py|
**Stage.py**|src/models/Person.py|
**helper.py**|src/build.py|
**prediction.py**|src/handler_person.py|
**training.py**|src/sort_collection.py|
**cli_input.py**|src/sort_collection.py|
**cli_output.py**|src/sort_collection.py|
**main.py**
**GoogleNews-vectors-negative300.bin.gz**|src/|
**arquivos de entrada de predição**|src/files/input/arquivos entrada.csv|
**arquivos de entrada de treinamento**|src/files/output/arquivos saida.csv|
**arquivos de modelos da rede neural.h5**.|src/| 


### Como executar?
Para executar a aplicação no ambiente Linux, o qual é o principal utilizado, basta seguir os seguintes passos:

1. Primeiramente é necessário conter o interpretador do Python 3. Para baixar e instalar siga este 
[link](https://www.python.org/downloads/), onde pode ser encontrado a última versão estável;

2. O ideal é que antes de executar a **etapa 3** é recomendável realizar a instalação das dependências em um ambiente 
virtual do Python, pois ele empacota todas as dependências que um projeto precisa e armazena em um 
diretório, isolando-o do SO base. A seguir um [link](https://pythonacademy.com.br/blog/python-e-virtualenv-como-programar-em-ambientes-virtuais) 
explicando em detalhes o que são e o passo a passo de como instalar e utilizar.

3. Abra o terminal bash dentro do ambiente virtual e execute o seguinte comando para instalar as dependências da 
aplicação:

        $ sh dependencies.sh

4. Após instalar as dependências ao projeto é necessário baixar o word2vec no [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) 
e colocá-lo no diretório **/src/data**xxxxxxxxx

5. Feito isto basta abrir o CLI (Command Line Interface) no diretório **/src** da aplicação e 
executar o seguinte comando para inicializar a aplicação:

        $ python3 main.py

6. Por fim basta interagir com a interface de linha de comando escolhendo as opções fornecidas pela aplicação, sendo as
principais de **treinamento** e **predição**.


### Informações adicionais
Todo o código fonte está hospedado no [GitHub](https://github.com/HaraHeique/TCC-Rede-Neural-Siamesa).