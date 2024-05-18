import streamlit as st
import pandas as pd
import prophet
import pickle
import datetime
import matplotlib.pyplot as plt

# raw_data = pd.read_excel("ipea_treated_1.xlsx")

# clean_data = raw_data.copy(deep=True)
# clean_data["date"] = pd.to_datetime(clean_data["date"])
# clean_data["price"] = clean_data["price"].interpolate().round(2)
# clean_data = clean_data.rename(columns={'date': 'ds', 'price': 'y'})

# ## Criando um modelo de exemplo abaixo
# modelo = pr.Prophet(seasonality_mode='additive', daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=5)

# modelo.add_country_holidays(country_name='GB')

# ## Treinando o modelo
# modelo.fit(clean_data)

# dataframefuturo = modelo.make_future_dataframe(freq='D', periods=365)

# previsao = modelo.predict(dataframefuturo)

# Load the model from the file
with open('prophet.pkl', 'rb') as f:
    modelo = pickle.load(f)

st.write("# Previsão do preço do petróleo Brent")

dashboard_url = "https://public.tableau.com/app/profile/beatriz.da.costa.inacio/viz/DashboarddoPreodoBarrildePetrleoBrent/Dashboard1"
st.image("dashboard.png", use_column_width="never")
st.write("Para ver o dashboard interativo, clique no [link](%s)" % dashboard_url)

st.write("# Escolha uma data para prever")
input_date = st.date_input("Escolha uma data de 04/05/2024 a 03/05/2025", value=datetime.date(2024, 5, 18), 
              min_value=datetime.date(2024, 5, 4), max_value=datetime.date(2025, 5, 3), format="DD/MM/YYYY")

if st.button("**Prever!**") == True:
    time_diff = input_date - datetime.date(2024, 5, 3)
    days = time_diff.days
    dataframefuturo = modelo.make_future_dataframe(freq='D', periods=days)
    previsao = modelo.predict(dataframefuturo)
    valor_previsto = round(previsao[(previsao['ds']==input_date.strftime("%Y-%m-%d"))]['yhat'].values[0], 2)
    st.write(f"## O preço previsto do petróleo para o dia {input_date.strftime('%d/%m/%Y')} é de {valor_previsto} dólares")

    fig = modelo.plot(previsao)

    # Criando um título e legenda
    plt.title("Previsão do Modelo Prophet para o preço do petróleo Brent")
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")

    # Adicionando uma legenda para as séries temporais observadas e previstas
    plt.legend(["Observado", "Previsão"])

    # Exibindo o gráfico
    st.pyplot(fig)