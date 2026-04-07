Resumo

Este notebook implementa o cálculo de uma Regressão Linear Simples do zero utilizando a Equação Normal (álgebra linear) em um dataset artificial. Além disso, o código inicia o carregamento e a exploração do famoso dataset habitacional da Califórnia.

Dependências e Requisitos

Linguagem: Python 3.x

Bibliotecas: numpy, pandas e matplotlib

Ferramentas: Jupyter Notebook ou Google Colab.

Instruções de Instalação e Execução

Faça o download do arquivo .ipynb e certifique-se de ter os arquivos de dados artificial1d.csv e california.csv.

Em seu terminal, instale as dependências via pip:

pip install numpy pandas matplotlib jupyter


Inicie o ambiente do Jupyter:

jupyter notebook


Abra o notebook, altere os caminhos de leitura dos arquivos CSV (caso não esteja usando o Google Colab) e execute as células sequencialmente.

Exemplo de Uso

O notebook calcula os pesos (w_chapeu) da regressão utilizando exclusivamente operações matriciais:

# Adicionando a coluna de 'uns' (bias/viés)
X = np.vstack([np.ones_like(x), x]).T

# Calculando os pesos usando a Equação Normal: w = (X^T * X)^-1 * X^T * y
w_chapeu = np.linalg.inv(X.T @ X) @ X.T @ y

# Predição e Erro Quadrático Médio
y_pred = X @ w_chapeu
mse = np.mean((y - y_pred) ** 2)
