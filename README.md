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





### Como executar?
Para executar a aplicação o app no ambiente Linux basta seguir os seguintes passos:

1. Primeiramente é necessário conter o interpretador do Python 3. Para baixar e instalar siga este [link](https://www.python.org/downloads/), onde pode ser encontrado a última versão estável;

2. Abra o terminal bash e execute o seguinte comando para instalar as dependências da aplicação:

        $ sh dependencies.sh

3. Após instalar as dependências do projeto (continuar)...

### Informações adicionais
Todo o código fonte está hospedado no [GitHub](https://github.com/HaraHeique/TCC-Rede-Neural-Siamesa).