# -*- coding: utf-8 -*-
"""
pràctiques eXiT - Cryptocurrency and stock change - Josep Serra i Mar Bulló
FILTREM LES DADES DE L'ATUR MUNDIAL - DE 2020 A 2024
"""

import pandas as pd

# Ruta al fitxer CSV
file_path = 'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/atur.csv'

# Llegim el fitxer CSV
df = pd.read_csv(file_path)
print("Fitxer llegit correctament")
print(df.head())

# Selecciona les columnes d'interès (modifica segons el nom real de les columnes)
# Assegura't que els anys són cadenes de text
columns_of_interest = [
   'Country Name','2020','2021','2022','2023'
]

# Comprova si les columnes d'interès existeixen en el DataFrame
missing_columns = [col for col in columns_of_interest if col not in df.columns]
if missing_columns:
    raise ValueError(f"Les columnes següents no es troben en el fitxer: {', '.join(missing_columns)}")

# Filtra les columnes d'interès
df_filtered = df[columns_of_interest]

# Substitueix 'no data' per NaN
df_filtered.replace('no data', pd.NA, inplace=True)

# Omple els valors buits amb la mitjana de cada any
for year in columns_of_interest[4:]:  # Omiteix les primeres quatre columnes que no són anys
    # Assegura't que la columna és numèrica abans de calcular la mitjana
    df_filtered[year] = pd.to_numeric(df_filtered[year], errors='coerce')
    mean_value = df_filtered[year].mean(skipna=True)
    df_filtered[year].fillna(mean_value, inplace=True)

# Guarda el resultat en un nou fitxer Excel
filtered_file_path = 'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/atur_filtrat.xlsx'
try:
    df_filtered.to_excel(filtered_file_path, index=False, engine='openpyxl')
    print(f"Dades filtrades guardades a {filtered_file_path}")
except Exception as e:
    print(f"Error guardant el fitxer: {e}")


