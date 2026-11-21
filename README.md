# Investigación de Operaciones con Streamlit

Aplicación educativa multipágina para explorar programación lineal de dos
variables con el método gráfico y cadenas de Markov. Los motores matemáticos
están separados de la interfaz y no dependen de Streamlit.

## Requisitos

- Python 3.10 o posterior
- `pip`

## Instalación y ejecución local

Desde la carpeta del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

En Linux o macOS, active el entorno con:

```bash
source .venv/bin/activate
```

La aplicación se abrirá normalmente en <http://localhost:8501>. En la barra
lateral puede cambiar entre **Programación Lineal (Método Gráfico)** y
**Cadenas de Markov**. Ambas páginas incluyen un ejemplo precargado y un botón
para restaurarlo.

## Despliegue en Streamlit Community Cloud

1. Publique esta carpeta en un repositorio de GitHub.
2. En [share.streamlit.io](https://share.streamlit.io/), seleccione **Create app**.
3. Elija el repositorio y la rama, e indique `app.py` como archivo principal
   (o `investigacion_operaciones_streamlit/app.py` si la carpeta está dentro de
   otro repositorio).
4. Configure Python 3.10 o posterior en la configuración avanzada si hace
   falta. Community Cloud instalará las dependencias desde `requirements.txt`.
5. Seleccione **Deploy**.

La aplicación no requiere secretos ni variables de entorno. Para un despliegue
reproducible, se recomienda fijar versiones de dependencias en
`requirements.txt` después de validar la combinación de Python elegida.

## Organización

```text
app.py                    Interfaz Streamlit y navegación
modulos/metodo_grafico.py Motor puro para programación lineal
modulos/markov.py         Motor puro para cadenas de Markov
requirements.txt          Dependencias de la aplicación
```
