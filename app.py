"""Interfaz multipágina para los motores de Investigación de Operaciones."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from modulos.markov import analizar_cadena
from modulos.metodo_grafico import ResultadoMetodoGrafico, resolver_programacion_lineal

st.set_page_config(page_title="Investigación de Operaciones", page_icon="📊", layout="wide")

PAGINAS = ("Programación Lineal (Método Gráfico)", "Cadenas de Markov")


def _cargar_ejemplo_lineal() -> None:
    st.session_state["lp_objetivo"] = "maximizar"
    st.session_state["lp_c1"] = 3.0
    st.session_state["lp_c2"] = 5.0
    st.session_state["lp_no_negatividad"] = True
    st.session_state["lp_num_restricciones"] = 3
    valores = ((2.0, 1.0, "<=", 18.0), (2.0, 3.0, "<=", 42.0), (3.0, 1.0, "<=", 24.0))
    for indice, (a1, a2, signo, b) in enumerate(valores):
        st.session_state[f"lp_a1_{indice}"] = a1
        st.session_state[f"lp_a2_{indice}"] = a2
        st.session_state[f"lp_signo_{indice}"] = signo
        st.session_state[f"lp_b_{indice}"] = b


def _dibujar_grafico_lineal(resultado: ResultadoMetodoGrafico) -> None:
    xmin, xmax, ymin, ymax = resultado.limites_grafico
    figura, eje = plt.subplots(figsize=(9, 6))
    for segmento in resultado.segmentos_rectas:
        eje.plot(segmento.x, segmento.y, linewidth=1.8, label=segmento.etiqueta)

    if len(resultado.region_ordenada) >= 3:
        poligono = np.asarray(resultado.region_ordenada)
        eje.fill(
            poligono[:, 0],
            poligono[:, 1],
            color="#4C9BE8",
            alpha=0.22,
            label=(
                "Región factible"
                if resultado.region_acotada
                else "Región factible visible (recortada)"
            ),
        )

    if resultado.vertices:
        eje.scatter(
            [punto.x1 for punto in resultado.vertices],
            [punto.x2 for punto in resultado.vertices],
            color="#167D5A",
            s=45,
            zorder=4,
            label="Puntos factibles",
        )
    if resultado.punto_optimo is not None:
        eje.scatter(
            [resultado.punto_optimo.x1],
            [resultado.punto_optimo.x2],
            color="#D62728",
            marker="*",
            s=210,
            zorder=5,
            label="Punto destacado",
        )
        eje.annotate(
            f"({resultado.punto_optimo.x1:.3g}, {resultado.punto_optimo.x2:.3g})",
            (resultado.punto_optimo.x1, resultado.punto_optimo.x2),
            xytext=(8, 8),
            textcoords="offset points",
        )

    eje.axhline(0, color="#555555", linewidth=0.8)
    eje.axvline(0, color="#555555", linewidth=0.8)
    eje.set_xlim(xmin, xmax)
    eje.set_ylim(ymin, ymax)
    eje.set_xlabel("x1")
    eje.set_ylabel("x2")
    eje.set_title("Restricciones y región factible")
    eje.grid(alpha=0.2)
    eje.legend(loc="best", fontsize="small")
    figura.tight_layout()
    st.pyplot(figura, width="stretch")
    plt.close(figura)


def _pagina_programacion_lineal() -> None:
    st.title("Programación Lineal · Método Gráfico")
    st.write("Optimiza una función lineal de dos variables sujeta a restricciones lineales.")
    st.button("Cargar ejemplo", on_click=_cargar_ejemplo_lineal)

    objetivo = st.selectbox("Tipo de objetivo", ("maximizar", "minimizar"), key="lp_objetivo")
    col1, col2, col3 = st.columns(3)
    c1 = col1.number_input("Coeficiente c1", key="lp_c1", value=3.0)
    c2 = col2.number_input("Coeficiente c2", key="lp_c2", value=5.0)
    no_negatividad = col3.checkbox(
        "Aplicar x1 ≥ 0 y x2 ≥ 0", key="lp_no_negatividad", value=True
    )
    st.latex(rf"Z = {c1:g}x_1 + {c2:g}x_2")

    cantidad = st.number_input(
        "Cantidad de restricciones",
        min_value=1,
        max_value=10,
        step=1,
        key="lp_num_restricciones",
        value=3,
    )
    restricciones = []
    st.markdown("#### Restricciones")
    for indice in range(int(cantidad)):
        a1_default, a2_default, signo_default, b_default = (1.0, 1.0, "<=", 10.0)
        ejemplo = ((2.0, 1.0, "<=", 18.0), (2.0, 3.0, "<=", 42.0), (3.0, 1.0, "<=", 24.0))
        if indice < len(ejemplo):
            a1_default, a2_default, signo_default, b_default = ejemplo[indice]
        columnas = st.columns((2, 1, 2, 1, 2))
        a1 = columnas[0].number_input(
            f"a1 · R{indice + 1}", key=f"lp_a1_{indice}", value=a1_default
        )
        columnas[1].markdown("x1 +")
        a2 = columnas[2].number_input(
            f"a2 · R{indice + 1}", key=f"lp_a2_{indice}", value=a2_default
        )
        signo = columnas[3].selectbox(
            f"Signo · R{indice + 1}",
            ("<=", ">=", "="),
            key=f"lp_signo_{indice}",
            index=("<=", ">=", "=").index(signo_default),
        )
        b = columnas[4].number_input(
            f"b · R{indice + 1}", key=f"lp_b_{indice}", value=b_default
        )
        restricciones.append((a1, a2, signo, b))
    st.caption("Cada fila representa a1·x1 + a2·x2 [signo] b.")

    if st.button("Resolver modelo", type="primary", key="lp_resolver"):
        resultado = resolver_programacion_lineal(
            objetivo, c1, c2, restricciones, no_negatividad
        )
        st.session_state["lp_resultado"] = resultado

    resultado = st.session_state.get("lp_resultado")
    if resultado is None:
        return
    if resultado.estado == "error":
        st.error(resultado.error or "Los datos del modelo no son válidos.")
        return

    if resultado.estado == "sin_solucion":
        st.error(resultado.mensaje)
    elif resultado.estado == "no_acotado":
        st.warning(resultado.mensaje)
        if resultado.punto_optimo is not None:
            st.metric("Punto factible de referencia", f"({resultado.punto_optimo.x1:.4g}, {resultado.punto_optimo.x2:.4g})")
    elif resultado.punto_optimo is not None:
        metrica1, metrica2, metrica3 = st.columns(3)
        metrica1.metric("Estado", "Óptimo")
        metrica2.metric("Punto destacado", f"({resultado.punto_optimo.x1:.4g}, {resultado.punto_optimo.x2:.4g})")
        metrica3.metric("Valor de Z", f"{resultado.punto_optimo.z:.6g}")

    if resultado.vertices:
        st.subheader("Puntos factibles evaluados")
        tabla = pd.DataFrame(
            [
                {"x1": punto.x1, "x2": punto.x2, "Z": punto.z}
                for punto in resultado.vertices
            ]
        )
        st.dataframe(tabla, hide_index=True, width="stretch")
    if not resultado.region_acotada:
        st.caption("El sombreado de la región factible se recorta a los límites visibles del gráfico.")
    _dibujar_grafico_lineal(resultado)


def _pagina_markov() -> None:
    st.title("Cadenas de Markov")
    st.write("Proyecta una distribución inicial y analiza regularidad y estado estable.")

    cantidad_estados = st.selectbox(
        "Número de estados",
        options=tuple(range(2, 7)),
        index=1,
        key="markov_num_estados",
    )
    etiquetas = tuple(f"Estado {indice}" for indice in range(1, cantidad_estados + 1))
    if cantidad_estados == 3:
        valores_matriz = [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ]
        valores_pi = [[1.0, 0.0, 0.0]]
    else:
        valores_matriz = np.full(
            (cantidad_estados, cantidad_estados), 1.0 / cantidad_estados
        )
        valores_pi = [[1.0] + [0.0] * (cantidad_estados - 1)]
    matriz_inicial = pd.DataFrame(
        valores_matriz, index=etiquetas, columns=etiquetas
    )
    pi_inicial = pd.DataFrame(valores_pi, columns=etiquetas)
    matriz = st.data_editor(
        matriz_inicial,
        key=f"markov_matriz_{cantidad_estados}",
        num_rows="fixed",
        width="stretch",
        column_config={
            columna: st.column_config.NumberColumn(columna, min_value=0.0, max_value=1.0, step=0.05)
            for columna in matriz_inicial.columns
        },
    )
    st.markdown("#### Distribución inicial π₀")
    distribucion = st.data_editor(
        pi_inicial,
        key=f"markov_pi_{cantidad_estados}",
        num_rows="fixed",
        width="stretch",
        column_config={
            columna: st.column_config.NumberColumn(columna, min_value=0.0, max_value=1.0, step=0.05)
            for columna in etiquetas
        },
    )
    pasos = st.number_input(
        "Número de pasos n", min_value=0, max_value=1000, step=1, key="markov_pasos", value=10
    )
    st.caption("Las filas de P describen el estado actual y las columnas el estado siguiente.")

    if st.button("Calcular cadena", type="primary", key="markov_calcular"):
        resultado = analizar_cadena(matriz.to_numpy(), distribucion.iloc[0].to_numpy(), int(pasos))
        st.session_state["markov_resultado"] = resultado

    resultado = st.session_state.get("markov_resultado")
    if resultado is None:
        return
    if not resultado.valido:
        st.error(resultado.error or "La cadena no es válida.")
        return

    if resultado.es_regular:
        st.success(resultado.mensaje)
    else:
        st.warning(resultado.mensaje)

    izquierda, derecha = st.columns(2)
    izquierda.metric(
        "Diferencia máxima |πn − π|",
        f"{resultado.diferencia_maxima:.6g}" if resultado.diferencia_maxima is not None else "—",
    )
    derecha.metric(
        "Potencia positiva mínima",
        f"P^{resultado.potencia_regular}" if resultado.potencia_regular is not None else "No encontrada",
    )

    estados = [f"Estado {indice + 1}" for indice in range(len(resultado.pi_n or ()))]
    filas = [
        {"Paso k": paso, **{estado: valor for estado, valor in zip(estados, probabilidades)}}
        for paso, probabilidades in resultado.historial
    ]
    st.subheader("Proyección paso a paso")
    tabla_historial = pd.DataFrame(filas).set_index("Paso k")
    st.dataframe(tabla_historial, width="stretch")

    comparacion = pd.DataFrame(
        {
            "Estado": estados,
            "πₙ": resultado.pi_n,
            "Estado estable π": resultado.estado_estable,
            "πₙ − π": resultado.diferencia_final,
        }
    )
    st.subheader("Comparación con el estado estable")
    st.dataframe(comparacion, hide_index=True, width="stretch")

    figura, eje = plt.subplots(figsize=(9, 4.5))
    eje.plot(tabla_historial.index, tabla_historial.to_numpy(), marker="o")
    eje.set_xlabel("Paso k")
    eje.set_ylabel("Probabilidad")
    eje.set_title("Evolución de la distribución por estado")
    eje.set_ylim(0, 1)
    eje.grid(alpha=0.25)
    eje.legend(estados, loc="best")
    figura.tight_layout()
    st.pyplot(figura, width="stretch")
    plt.close(figura)


st.sidebar.title("Investigación de Operaciones")
pagina = st.sidebar.radio("Selecciona un módulo", PAGINAS, key="pagina_activa")
if pagina == PAGINAS[0]:
    _pagina_programacion_lineal()
else:
    _pagina_markov()
