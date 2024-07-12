# -*- coding: utf-8 -*-
"""
pràctiques eXiT - Cryptocurrency and stock change - Josep Serra i Mar Bulló
FILTREM LES DADES DEL PREU DE MATÈRIES PRIMERES(dades diàries dels últims tres mesos)
"""

import pandas as pd

# Ruta als fitxers CSV
file_or = 'preu_or.csv'
file_gasNatural = 'preu_gasNatural.csv'
file_petroliBrent = 'preu_petroliBrent.csv'
file_petroliCru = 'preu_petroliCru.csv'
file_plata = 'preu_plata.csv'

# Llegeix els fitxers CSV directament en DataFrames
df_or = pd.read_csv(file_or)
df_gasNatural = pd.read_csv(file_gasNatural)
df_petroliBrent = pd.read_csv(file_petroliBrent)
df_petroliCru = pd.read_csv(file_petroliCru)
df_plata = pd.read_csv(file_plata)

# Convertir les columnes adequades a numèric, excloent 'Fecha' i 'Vol.'
columns_of_interest = ["Fecha", "Último", "Apertura", "Máximo", "Mínimo", "Vol.", "% var."]

# Processament per l'or
for col in columns_of_interest:
    if col != "Fecha" and col != "Vol.":
        mean_value_or = df_or[col].mean(skipna=True)
        df_or[col].fillna(mean_value_or, inplace=True)

# Guarda el resultat en un nou fitxer Excel per l'or
file_or_filtrat = 'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/or_filtrat.xlsx'
try:
    df_or.to_excel(file_or_filtrat, index=False, engine='openpyxl')
    print(f"Dades filtrades de l'or guardades a {file_or_filtrat}")
except Exception as e:
    print(f"Error guardant el fitxer: {e}")

# Processament pel gas natural
for col in columns_of_interest:
    if col != "Fecha" and col != "Vol.":
        mean_value_gasNatural = df_gasNatural[col].mean(skipna=True)
        df_gasNatural[col].fillna(mean_value_gasNatural, inplace=True)

# Guarda el resultat en un nou fitxer Excel pel gas natural
file_gasNatural_filtrat = 'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/gasNatural_filtrat.xlsx'
try:
    df_gasNatural.to_excel(file_gasNatural_filtrat, index=False, engine='openpyxl')
    print(f"Dades filtrades del gas natural guardades a {file_gasNatural_filtrat}")
except Exception as e:
    print(f"Error guardant el fitxer: {e}")

# Processament pel petroli Brent
for col in columns_of_interest:
    if col != "Fecha" and col != "Vol.":
        mean_value_petroliBrent = df_petroliBrent[col].mean(skipna=True)
        df_petroliBrent[col].fillna(mean_value_petroliBrent, inplace=True)

# Guarda el resultat en un nou fitxer Excel pel petroli Brent
file_petroliBrent_filtrat = 'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/petroliBrent_filtrat.xlsx'
try:
    df_petroliBrent.to_excel(file_petroliBrent_filtrat, index=False, engine='openpyxl')
    print(f"Dades filtrades del petroli Brent guardades a {file_petroliBrent_filtrat}")
except Exception as e:
    print(f"Error guardant el fitxer: {e}")

# Processament pel petroli cru
for col in columns_of_interest:
    if col != "Fecha" and col != "Vol.":
        mean_value_petroliCru = df_petroliCru[col].mean(skipna=True)
        df_petroliCru[col].fillna(mean_value_petroliCru, inplace=True)

# Guarda el resultat en un nou fitxer Excel pel petroli cru
file_petroliCru_filtrat = 'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/petroliCru_filtrat.xlsx'
try:
    df_petroliCru.to_excel(file_petroliCru_filtrat, index=False, engine='openpyxl')
    print(f"Dades filtrades del petroli cru guardades a {file_petroliCru_filtrat}")
except Exception as e:
    print(f"Error guardant el fitxer: {e}")

# Processament per la plata
for col in columns_of_interest:
    if col != "Fecha" and col != "Vol.":
        mean_value_plata = df_plata[col].mean(skipna=True)
        df_plata[col].fillna(mean_value_plata, inplace=True)

# Guarda el resultat en un nou fitxer Excel per la plata
file_plata_filtrat = 'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/plata_filtrat.xlsx'
try:
    df_plata.to_excel(file_plata_filtrat, index=False, engine='openpyxl')
    print(f"Dades filtrades de la plata guardades a {file_plata_filtrat}")
except Exception as e:
    print(f"Error guardant el fitxer: {e}")

