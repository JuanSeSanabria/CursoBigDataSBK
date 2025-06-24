import pandas as pd
import numpy as np

df_INGREFAM_2022 = pd.read_csv("INGREFAM_EPHC_ANUAL_2022.csv", sep=";", decimal=",")
df_INGREFAM_2023 = pd.read_csv("INGREFAM_EPHC_ANUAL_2023.csv", sep=";", decimal=",")
df_INGREFAM_2024 = pd.read_csv("INGREFAM_EPHC_ANUAL_2024.csv", sep=";", decimal=",")

df_INGREFAM_completo = pd.concat([df_INGREFAM_2022, df_INGREFAM_2023, df_INGREFAM_2024], ignore_index=True)

df_INGREFAM_completo.columns = df_INGREFAM_completo.columns.str.strip()

df_REG01_2022 = pd.read_csv("REG01_EPHC_ANUAL_2022.csv", sep=";", decimal=",")
df_REG01_2023 = pd.read_csv("REG01_EPHC_ANUAL_2023.csv", sep=";", decimal=",")
df_REG01_2024 = pd.read_csv("REG01_EPHC_ANUAL_2024.csv", sep=";", decimal=",")

df_REG01_completo = pd.concat([df_REG01_2022, df_REG01_2023, df_REG01_2024], ignore_index=True)

df_REG01_completo.columns = df_REG01_completo.columns.str.strip()

df_REG01_completo = df_REG01_completo.rename(columns={
    "UPM": "upm",
    "NVIVI": "nvivi",
    "NHOGA": "nhoga",
    "DPTO": "dpto",
    "AREA": "area",
    "AÑO": "año"  
})

# Hacer el left join
df_INGREFAM_REG01 = pd.merge(
    df_INGREFAM_completo,
    df_REG01_completo,
    on=["upm", "nvivi", "nhoga", "dpto", "area", "año"],
    how="left"
)

columnas_deseadas = [
    "upm", "nvivi", "nhoga", "dpto", "area", "año", "ingrem", "ipcm", 
    "THOGAV", "POBREZAI", "POBNOPOI", "HOMBRES", "MUJERES", "TOTAL", "V01"
    ]

df_final = df_INGREFAM_REG01[columnas_deseadas]

print(df_final.head())

df_limpio = df_final.copy()
columnas_clave = ["ingrem", "ipcm", "THOGAV", "POBREZAI", "POBNOPOI"]
df_limpio = df_limpio.dropna(subset=columnas_clave)

df_limpio = df_limpio[
    (df_limpio["TOTAL"] >= 0) & (df_limpio["TOTAL"] <= 20) &
    (df_limpio["ingrem"] >= 0) & (df_limpio["ingrem"] < 200000000) &
    (df_limpio["ipcm"] >= 0) & (df_limpio["ipcm"] < 100000000)
    ]

categorias_validas = {
    "THOGAV": [1, 2, 3, 4, 5],
    "POBREZAI": [1, 2, 3],
    "POBNOPOI": [0, 1],
    }

for col, valores_validos in categorias_validas.items():
    df_limpio = df_limpio[df_limpio[col].isin(valores_validos)]

print(df_limpio.describe(include='all'))

#Grafico para analizar tendencias

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

mapa_dptos = {
    0: "Asunción",
    1: "Concepción",
    2: "San Pedro",
    3: "Cordillera",
    4: "Guairá",
    5: "Caaguazú",
    6: "Caazapá",
    7: "Itapúa",
    8: "Misiones",
    9: "Paraguarí",
    10: "Alto Paraná",
    11: "Central",
    12: "Ñeembucú",
    13: "Amambay",
    14: "Canindeyú",
    15: "Presidente Hayes"
}
df_limpio["dpto_nombre"] = df_limpio["dpto"].map(mapa_dptos)

mapa_area = {
    1: "Urbana",
    6: "Rural"
}
df_limpio["area_nombre"] = df_limpio["area"].map(mapa_area)

# Obtener ingreso promedio por departamento

df_barras_dpto = df_limpio.groupby("dpto_nombre")["ingrem"].mean().sort_values()

colores = sns.color_palette("Set3", n_colors=len(df_barras_dpto))  

plt.figure(figsize=(12, 6))
df_barras_dpto.plot(kind="bar", color=colores)

formatter = FuncFormatter(lambda x, _: f'{x / 1_000_000:.1f}M')
plt.gca().yaxis.set_major_formatter(formatter)

plt.title("Ingreso Familiar Promedio por Departamento")
plt.ylabel("Ingreso mensual promedio (millones de Gs.)")
plt.xlabel("Departamento")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Obtener ingreso promedio por área

df_barras = df_limpio.groupby(["dpto_nombre", "area_nombre"])["ingrem"].mean().reset_index()
plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_barras,
    x="dpto_nombre",
    y="ingrem",
    hue="area_nombre",
    palette={"Urbana": "hotpink", "Rural": "mediumseagreen"}
)

formatter = FuncFormatter(lambda x, _: f'{x / 1_000_000:.1f}M')
plt.gca().yaxis.set_major_formatter(formatter)

plt.title("Ingreso Familiar Promedio por Departamento y Área")
plt.xlabel("Departamento")
plt.ylabel("Ingreso mensual promedio (millones Gs.)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Área")
plt.tight_layout()
plt.show()

#Analisis Predictivo

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

variables_num = ["HOMBRES", "MUJERES", "TOTAL", "ingrem"]
variables_cat = ["THOGAV", "POBREZAI"]
df_modelo = df_limpio[variables_num + variables_cat + ["ipcm"]].copy()
df_modelo = pd.get_dummies(df_modelo, columns=variables_cat, drop_first=True)
X = df_modelo.drop(columns=["ipcm"])
y = df_modelo["ipcm"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Coeficientes con nombres
coeficientes = pd.Series(modelo.coef_, index=X.columns)
print("\nCoeficientes del modelo:")
print(coeficientes.sort_values(ascending=False))

# Gráfico 1: Real vs Predicho
formatter_millones = FuncFormatter(lambda x, _: f'{x / 1_000_000:.1f}M')
a = modelo.coef_[0]  # Asumiendo que es el coef de la primera variable, puede que necesites ajustar
b = modelo.intercept_
y_test_array = y_test.values.reshape(-1, 1)
y_pred_array = y_pred.reshape(-1, 1)
ajuste_lineal = LinearRegression().fit(y_test_array, y_pred_array)
pendiente = ajuste_lineal.coef_[0][0]
intercepto = ajuste_lineal.intercept_[0]

formula_texto = f"y = {pendiente:.2f}x + {intercepto:.2f}"

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='gray')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Ideal: y = x')
x_vals = np.linspace(y_test.min(), y_test.max(), 100)
y_vals = pendiente * x_vals + intercepto
plt.plot(x_vals, y_vals, 'b-', label=formula_texto)
plt.gca().xaxis.set_major_formatter(formatter_millones)
plt.gca().yaxis.set_major_formatter(formatter_millones)

plt.xlabel("Ingreso por persona real (ipcm)")
plt.ylabel("Ingreso por persona predicho")
plt.title("Predicción del ingreso por persona: real vs predicho")
plt.legend()
plt.tight_layout()
plt.savefig("prediccion_vs_real_formula.png", dpi=300)
plt.show()

# Gráfico 2: Importancia de coeficientes (valor absoluto)
# Traducción para nombres
traduccion_coef = {
    "POBREZAI_2": "Pobre no extremo",
    "POBREZAI_3": "No pobre",
    "THOGAV_2": "Familia Nuclear completo",
    "THOGAV_3": "Familia Nuclear incompleto",
    "THOGAV_4": "Familia Extendido",
    "THOGAV_5": "Familia Compuesto",
    "HOMBRES": "Cantidad de hombres",
    "MUJERES": "Cantidad de mujeres",
    "TOTAL": "Total personas hogar",
    "ingrem": "Ingreso familiar total"
}
coef_abs = coeficientes.abs()
coef_abs.index = [traduccion_coef.get(i, i) for i in coef_abs.index]

plt.figure(figsize=(10, 6))
coef_abs.sort_values().plot(kind='barh', color='mediumorchid')
plt.xlabel("Valor absoluto del coeficiente")
plt.title("Importancia relativa de variables (regresión lineal)")
plt.tight_layout()
plt.savefig("importancia_variables.png", dpi=300)
plt.show()

# Cantidad de hombres y mujeres por categoría de pobreza

df_pobreza_filtrado = df_limpio[df_limpio["POBREZAI"].isin([2, 3])]
df_pobreza_sum = df_pobreza_filtrado.groupby("POBREZAI")[["HOMBRES", "MUJERES"]].sum()

# Etiquetas
etiquetas = {
    2: "Pobre no extremo",
    3: "No pobre"
}
df_pobreza_sum.index = df_pobreza_sum.index.map(etiquetas)

ind = np.arange(len(df_pobreza_sum))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(ind - width/2, df_pobreza_sum["HOMBRES"], width, label='Hombres', color='steelblue')
ax.bar(ind + width/2, df_pobreza_sum["MUJERES"], width, label='Mujeres', color='hotpink')

ax.set_xlabel("Categoría Pobreza")
ax.set_ylabel("Cantidad total de personas")
ax.set_title("Cantidad de Hombres y Mujeres por Categoría de Pobreza")
ax.set_xticks(ind)
ax.set_xticklabels(df_pobreza_sum.index)
ax.legend()

plt.tight_layout()
plt.savefig("hombres_mujeres_por_pobreza.png", dpi=300)
plt.show()
