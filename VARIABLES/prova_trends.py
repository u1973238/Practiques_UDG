# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:21:57 2024

@author: Mar
"""
import pandas as pd

# Llegir les dades diàries des dels fitxers CSV
trends_apple = pd.read_csv('trends_apple.csv', parse_dates=['Dia'], index_col='Dia')
trends_bitcoin = pd.read_csv('trends_bitcoin.csv', parse_dates=['Dia'], index_col='Dia')
trends_criptomoneda = pd.read_csv('trends_criptomoneda.csv', parse_dates=['Dia'], index_col='Dia')

# Funció per extreure els valors diaris de cada setmana i assegurar-se que totes les setmanes tenen 7 dies
def get_weekly_values(df):
    # Resamplejar per setmanes, obtenint una llista de sèries de cada setmana
    weekly = df.resample('W').apply(lambda x: list(x.values.flatten()))
    # Omplir les setmanes incompletes amb NaN fins a tenir 7 dies
    weekly = weekly.apply(lambda x: x + [None]*(7-len(x)) if len(x) < 7 else x)
    return weekly

# Aplicar la funció a cada dataset
weekly_apple_values = get_weekly_values(trends_apple)
weekly_bitcoin_values = get_weekly_values(trends_bitcoin)
weekly_criptomoneda_values = get_weekly_values(trends_criptomoneda)

# Crear un DataFrame per cada dia de la setmana
apple_days = pd.DataFrame(weekly_apple_values.tolist(), index=weekly_apple_values.index, columns=['Apple_Day_1', 'Apple_Day_2', 'Apple_Day_3', 'Apple_Day_4', 'Apple_Day_5', 'Apple_Day_6', 'Apple_Day_7'])
bitcoin_days = pd.DataFrame(weekly_bitcoin_values.tolist(), index=weekly_bitcoin_values.index, columns=['Bitcoin_Day_1', 'Bitcoin_Day_2', 'Bitcoin_Day_3', 'Bitcoin_Day_4', 'Bitcoin_Day_5', 'Bitcoin_Day_6', 'Bitcoin_Day_7'])
criptomoneda_days = pd.DataFrame(weekly_criptomoneda_values.tolist(), index=weekly_criptomoneda_values.index, columns=['Criptomoneda_Day_1', 'Criptomoneda_Day_2', 'Criptomoneda_Day_3', 'Criptomoneda_Day_4', 'Criptomoneda_Day_5', 'Criptomoneda_Day_6', 'Criptomoneda_Day_7'])

# Concatenar els DataFrames en un sol DataFrame
combined_weekly_values = pd.concat([apple_days, bitcoin_days, criptomoneda_days], axis=1)

# Comprovar alguns valors
print(combined_weekly_values.head())
