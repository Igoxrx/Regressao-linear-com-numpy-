# Regressão Linear e Matemática com NumPy

## Resumo

Este projeto foca na implementação dos fundamentos matemáticos da Regressão Linear Simples sem o uso de bibliotecas de alto nível para o treinamento do modelo. O notebook demonstra o cálculo dos parâmetros ideais através da Equação Normal, utilizando álgebra linear e operações matriciais puras com NumPy em um conjunto de dados artificial. Adicionalmente, o projeto explora o carregamento e a análise inicial do dataset habitacional da Califórnia (*California Housing*), preparando o terreno para análises estatísticas mais complexas.

## Dependências e Requisitos

* **Linguagem:** Python 3.x
* **Bibliotecas:** * `numpy`: Para cálculos de álgebra linear e manipulação de matrizes.
    * `pandas`: Para manipulação e exploração de datasets.
    * `matplotlib`: Para visualização dos dados e da reta de regressão.
* **Ferramentas:** Jupyter Notebook ou Google Colab.

## Instruções de Instalação e Execução

Para reproduzir os cálculos e visualizações, siga as instruções abaixo:

1.  **Arquivos Necessários:**
    Certifique-se de possuir o notebook `.ipynb` e os arquivos de dados `artificial1d.csv` e `california.csv` no mesmo diretório.

2.  **Instalação de Dependências:**
    Instale as bibliotecas necessárias via terminal utilizando o gerenciador de pacotes pip:
    ```bash
    pip install numpy pandas matplotlib jupyter
    ```

3.  **Inicialização:**
    Inicie o ambiente do Jupyter:
    ```bash
    jupyter notebook
    ```

4.  **Execução:**
    Abra o notebook no navegador. Caso não esteja utilizando o Google Colab, verifique se os caminhos de leitura dos arquivos CSV estão apontando para o diretório correto em sua máquina e execute as células sequencialmente.

## Exemplo de Uso

O núcleo deste projeto é o cálculo dos pesos do modelo ($w$) utilizando exclusivamente operações matriciais, dispensando algoritmos de gradiente descendente para casos simples:

```python
# Adicionando a coluna de 'uns' para o termo de viés (bias)
X = np.vstack([np.ones_like(x), x]).T

# Aplicação da Equação Normal: w = (X^T * X)^-1 * X^T * y
w_chapeu = np.linalg.inv(X.T @ X) @ X.T @ y

# Realização da predição e cálculo do Erro Quadrático Médio (MSE)
y_pred = X @ w_chapeu
mse = np.mean((y - y_pred) ** 2)
```

---

## Licença

Este repositório possui caráter puramente acadêmico, focado no estudo da implementação de algoritmos de aprendizado de máquina a partir de seus fundamentos matemáticos.
