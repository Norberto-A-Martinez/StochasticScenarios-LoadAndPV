import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# set the desired number of scenarios
scenarios = 3

# loading data and adjusting to hourly blocks
pv_data = pd.read_csv(glob.glob("input-data/ninja_pv*")[0])
pv_curves = pv_data['electricity']
pv_curves_reshaped = pv_curves.values.reshape(365, 24)

load_data = pd.read_csv("input-data/load_clean.csv", sep=";")
load_curves = load_data['E1A_AZI_A']

# (365 days, 24 hours, 4 measurements of 15 min per hour)
load_curves_blocos = load_curves.values.reshape(365, 24, 4)

# Sum the 4 blocks of 15 min to obtain the energy fraction per hour
load_curves_1h = load_curves_blocos.sum(axis=2)

# normalization by the annual peak ()
pico_anual = load_curves_1h.max()
load_curves_reshaped = load_curves_1h / pico_anual

# Monthly clustering
dias_por_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

centroides_load_por_mes = {}
centroides_pv_por_mes = {}
probabilidades_load_por_mes = {}
probabilidades_pv_por_mes = {}

dia_inicial = 0

for i, dias in enumerate(dias_por_mes):
    mes = i + 1 
    dia_final = dia_inicial + dias
    
    # get one month
    load_mes_atual = load_curves_reshaped[dia_inicial:dia_final, :]
    pv_mes_atual = pv_curves_reshaped[dia_inicial:dia_final, :]
    
    # KMeans
    kmeans_loads = KMeans(n_clusters=scenarios, random_state=0, n_init=10).fit(load_mes_atual)
    kmeans_pv = KMeans(n_clusters=scenarios, random_state=0, n_init=10).fit(pv_mes_atual)
    
    # save the clusters
    centroides_load_por_mes[mes] = kmeans_loads.cluster_centers_
    centroides_pv_por_mes[mes] = kmeans_pv.cluster_centers_
    
    # calc the probabilities
    contagem_load = np.bincount(kmeans_loads.labels_, minlength=scenarios)
    contagem_pv = np.bincount(kmeans_pv.labels_, minlength=scenarios)
    probabilidades_load_por_mes[mes] = contagem_load / dias
    probabilidades_pv_por_mes[mes] = contagem_pv / dias
    
    dia_inicial = dia_final

# export to csv
dados_load_curvas = []
dados_pv_curvas = []
dados_load_probs = []
dados_pv_probs = []

for m in range(1, 13):
    for s in range(scenarios):
        cenario_id = s + 1
        
        dados_load_probs.append([m, cenario_id, probabilidades_load_por_mes[m][s]])
        dados_pv_probs.append([m, cenario_id, probabilidades_pv_por_mes[m][s]])
        
        for t in range(24):
            hora_id = t + 1
            
            valor_load = centroides_load_por_mes[m][s][t]
            dados_load_curvas.append([m, cenario_id, hora_id, valor_load])
            
            valor_pv = centroides_pv_por_mes[m][s][t]
            dados_pv_curvas.append([m, cenario_id, hora_id, valor_pv])

pd.DataFrame(dados_load_curvas, columns=['M', 'D', 'T', 'curve_load']).to_csv('input-data/parametros_load_curvas.csv', index=False)
pd.DataFrame(dados_pv_curvas, columns=['M', 'S', 'T', 'curve_pv']).to_csv('input-data/parametros_pv_curvas.csv', index=False)
pd.DataFrame(dados_load_probs, columns=['M', 'D', 'prob_D']).to_csv('input-data/parametros_load_probs.csv', index=False)
pd.DataFrame(dados_pv_probs, columns=['M', 'S', 'prob_S']).to_csv('input-data/parametros_pv_probs.csv', index=False)
