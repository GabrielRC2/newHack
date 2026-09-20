import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Painel de Frequência - MVP")

menu = st.sidebar.selectbox("Menu", ["Aulas Abertas / Presença"])

if menu == "Aulas Abertas / Presença":
    class_id = st.number_input("ID da Aula", min_value=1, step=1)

    if st.button("Ver Status dos Alunos"):
        res = requests.get(f"{API_URL}/aulas/{class_id}/presencas")
        if res.status_code == 200:
            data = res.json()
            st.write(f"Alunos na Aula {class_id}:")
            st.table(data)
        else:
            st.error("Erro ao buscar dados.")

    if st.button("Encerrar Aula"):
        res = requests.post(f"{API_URL}/aulas/{class_id}/fechar")
        if res.status_code == 200:
            st.success("Aula encerrada com sucesso! Faltas e presenças calculadas.")