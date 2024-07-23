#%%
# INITIALIZATION
import pandas as pd
import numpy as np
import pickle
# import copy
# import pdb
import streamlit as st
from streamlit_tree_select import tree_select
# from openai.embeddings_utils import get_embedding
# import tiktoken
from scipy.spatial import distance
from pathlib import Path
import openai
from openai import OpenAI
from PIL import Image
from io import BytesIO
from datetime import datetime
from ENAHOxplorer_small import *


key_ = ""#st.secrets["akey"]
#%%
# configurationpage
st.set_page_config(layout="wide")
#%%
st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 100px;
}
</style>
""",
    unsafe_allow_html=True,
)

path_str = Path(__file__).parent.parent
# key_ = st.secrets["akey"]

if 'xplorer' not in st.session_state:
    xplorer = ENAHOxplorer(key_=key_)
    st.session_state['xplorer'] = xplorer
else:
    xplorer = st.session_state['xplorer']
#%% Funciones
@st.cache_data(ttl=600)
def _find(str_buscado, xplorer = xplorer):
    emb_buscar = xplorer.get_embedding(str_buscado)
        # calcular distancias/similitud
    xplorer.dfvars['similitud'] = xplorer.dfvars['embedding'].apply(xplorer._get_similitud, args=(emb_buscar,))
    # index_bool = (xplorer.dfvars['similitud']>0.3)
    index_bool = xplorer.cluster_near_one_gmm(xplorer.dfvars['similitud']) # 2 cluster x defecto
    finded = xplorer.dfvars.loc[index_bool].sort_values(['similitud'], ascending=False)
    return finded

#%% 
#########################
image = Image.open(path_str / '01_Data/Logo_iaxta.jpeg')

col1, col2, col3, col4 = st.columns(4)

col1.image(image, width=200)
col1.title("Buscador semántico de variables en la Encuesta Nacional de Hogares (ENAHO 2023) con IA")
col1.text("Versión 1.0 (julio de 2024)")

# términos en total##################
terms_total = xplorer.dfvars.shape[0]
col2.metric(label="Preguntas totales", value=terms_total)

explorer, table = st.columns((1,3))

with explorer:
    st.text("Grupo de variables")
    return_select = tree_select(xplorer.tree)
    # st.write(return_select)

with table:
    selected_gr = return_select['checked'] #.values())
    selected_list = xplorer.dfgrvars.loc[xplorer.dfgrvars['@ID'].apply(lambda x: x in selected_gr),'@var'].to_list()
    selected = [y for x in selected_list for y in str(x).split(' ')]
    # st.write(selected_gr)
    # st.write("________")
    # st.write(selected)
    str_buscar = st.text_input("Introduzca el texto a buscar en el siguiente recuadro:")
    bool_buscar = st.button("Buscar")
    # st.text(bool_buscar)
    # st.text(str_buscar!="")
    findedterms_subtotal = terms_total
    if  bool_buscar | (str_buscar!=""):
        dffinded = _find(str_buscar)
        findedterms_subtotal = dffinded.shape[0]
        # ordenar la base
        # subtable = dffinded[['@ID','labl']]
        subtable = dffinded.loc[dffinded['@ID'].apply(lambda x: x in selected),['Nombre','Etiqueta','Definición','Pregunta']]
        st.dataframe(subtable, hide_index=True, use_container_width=True)#, width=2000)
    else:
        # # if buscar, get embedding, get normas, ordenar de mayor a menor similaridad
        # # quedarnos con los mayores a 4 # da igual ya que usamos in
        subtable = xplorer.dfvars.loc[xplorer.dfvars['@ID'].apply(lambda x: x in selected),['Nombre','Etiqueta','Definición','Pregunta']]
        if subtable.shape[0]==0:
            st.warning("Seleccione alguno de los Grupos en el panel de la izquierda para que se muestren los resultados.", icon="⚠️")
        st.dataframe(subtable, hide_index=True, use_container_width=True)#, width=2000)


col3.metric(label="Preguntas encontradas", value=findedterms_subtotal)

terms_subtotal = subtable.shape[0]
col4.metric(label="Preguntas seleccionadas", value=terms_subtotal)

@st.cache_data(ttl=600)
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

try:
    df_xlsx = to_excel(subtable)
    today = str(datetime.today())
    table.download_button(label='📥 Descargar las preguntas seleccionadas (.xlsx)',
                                data=df_xlsx, 
                                file_name= f'ENAHO_2023_descargado_el_{today}.xlsx')
except:
    # table.text("La tabla está vacía")
    table.warning("La tabla está vacía", icon="⚠️")