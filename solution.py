import numpy as np
import pandas as pd

def calcular_coeficientes_google_sheets(url_planilha: str) -> list:
    if "/edit" in url_planilha:
        url_csv = url_planilha.split("/edit")[0] + "/export?format=csv"
    else:
        url_csv = url_planilha


    df = pd.read_csv(url_csv)
    
    
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
