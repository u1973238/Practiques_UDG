# -*- coding: utf-8 -*-
"""
pràctiques eXiT - Cryptocurrency and stock change - Josep Serra i Mar Bulló
FILTREM LES DADES DE LA INFLACIÓ MUNDIAL - DE 2020 A 2024

"""

import pandas as pd

# Ruta al fitxer Excel
file_path = 'inflacio.xls'

# Llegim el fitxer Excel
df = pd.read_excel(file_path)
print("Fitxer llegit correctament")
print(df.head())

# Selecciona les columnes d'interès (modifica segons el nom real de les columnes)
# Assegura't que els noms de les columnes coincidint amb els noms reals del fitxer
columns_of_interest = [
    'Inflation rate, end of period consumer prices (Annual percent change)', 
    2020, 
    2021, 
    2022, 
    2023, 
    2024
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
for year in columns_of_interest[1:]:  # Omiteix la columna d'inflació
    mean_value = df_filtered[year].mean(skipna=True)
    df_filtered[year].fillna(mean_value, inplace=True)

# Guardar les dades filtrades en un nou fitxer CSV
output_file_path = 'inflacio_filtrada.csv'
df_filtered.to_csv(output_file_path, index=False)

