import numpy as np
import pandas as pd

def calcular_coeficientes_regressao(caminho_arquivo: str) -> list:
    df = pd.read_csv(caminho_arquivo)
    
    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()
    
    n_amostras = len(preco)
    coluna_intercepto = [1.0] * n_amostras
    X = np.column_stack((coluna_intercepto, area, quartos))
    
    Y = np.array(preco)
    
    XT = X.T
    XTX = np.matmul(XT, X)
    XTX_inv = np.linalg.inv(XTX)
    XTY = np.matmul(XT, Y)
    
    beta = np.matmul(XTX_inv, XTY)
    
    return beta.tolist()
if __name__ == "__main__":
    arquivo = "imoveis_lar_perfeito.csv"
    coeficientes = calcular_coeficientes_regressao(arquivo)
    
    print("--- RESULTADO DA REGRESSÃO LINEAR MÚLTIPLA ---")
    print(f"Lista de coeficientes [beta_0, beta_1, beta_2]:\n{coeficientes}")
    print("\nEquação de Precificação:")
    print(f"Preco = {coeficientes[0]:.2f} + {coeficientes[1]:.2f} * Area + {coeficientes[2]:.2f} * Quartos")
