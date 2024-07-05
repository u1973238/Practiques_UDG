# -*- coding: utf-8 -*-
"""
pràctiques eXiT - Cryptocurrency and stock change - Josep Serra i Mar Bulló
FILTREM LES DADES DEL PREU DE L'OR DE L'ULTIM ANY (dades setmanals)
"""

import pandas as pd

# Ruta al fitxer CSV
file_path = 'preu_setm_or.csv'

# Llegeix el fitxer CSV directament en un DataFrame
df = pd.read_csv(file_path)

# Opcional: transposa el DataFrame si és necessari
# df = df.T

# Selecciona les columnes d'interès (modifica segons el nom real de les columnes)
columns_of_interest = [ 
    "30.06.2024", "23.06.2024", "16.06.2024", 
    "09.06.2024", "02.06.2024", "26.05.2024", 
    "19.05.2024", "12.05.2024", "05.05.2024", 
    "28.04.2024", "21.04.2024", "14.04.2024", "07.04.2024"
]

# Comprova si les columnes d'interès existeixen en el DataFrame
missing_columns = [col for col in columns_of_interest if col not in df.columns]
if missing_columns:
    raise ValueError(f"Les columnes següents no es troben en el fitxer: {', '.join(missing_columns)}")

# Filtra les columnes d'interès
df_filtered = df[columns_of_interest]

# Substitueix 'no data' per NaN
df_filtered.replace('no data', pd.NA, inplace=True)

# Omple els valors buits amb la mitjana de cada columna
for col in df_filtered.columns:
    # Assegura't que la columna és numèrica abans de calcular la mitjana
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    mean_value = df_filtered[col].mean(skipna=True)
    df_filtered[col].fillna(mean_value, inplace=True)

# Guarda el resultat en un nou fitxer Excel
filtered_file_path = 'C:/Users/Mar/Documents/GitHub/Practiques_UDG/VARIABLES/or_filtrat.xlsx'
try:
    df_filtered.to_excel(filtered_file_path, index=False, engine='openpyxl')
    print(f"Dades filtrades guardades a {filtered_file_path}")
except Exception as e:
    print(f"Error guardant el fitxer: {e}")
