import streamlit as st
import pandas as pd
from anime_clusters import get_cluster

###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode

###################################

input_df = pd.read_csv('./pop_df.csv')
input_df = input_df.sort_values(by='popularity')
show_list = input_df['title_english'].to_list()
show_list.insert(0,'')

def _max_width_():
    max_width_str = f"max-width: 2400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="👺", page_title="Anime Match")

st.image(
    "https://emojis.wiki/thumbs/emojis/goblin.webp",
    width=100,
)

st.title("Anime Match")

c29, c30, c31 = st.columns([1, 6, 1])

with c30:

    name = st.selectbox("Pick your favorite anime:",show_list)
    radio = st.radio("Sort by:",('Closest Match','Score','Popularity'))
    if name != '':
        df, img_url = get_cluster(name,radio)
        with c31:
            st.image(img_url,width=200)
    else:
        st.info(
            f"""
                👆 Select your favorite anime!
                """
        )

        st.stop()

from st_aggrid import GridUpdateMode, DataReturnMode, JsCode


df['genres']=df['genres'].apply(lambda x: x.replace('[','').replace(']','').replace("'",''))
df['themes']=df['themes'].apply(lambda x: x.replace('[','').replace(']','').replace("'",''))
df['table_img']=df['images_url'].apply(lambda x: x.split('.jpg')[0]+'t.jpg')

image_nation = JsCode("""function (params) {
        var element = document.createElement("span");
        var imageElement = document.createElement("img");
    
        imageElement.src = params.value;
        imageElement.width="40";
        imageElement.height="57";

        element.appendChild(imageElement);;
        return element;
        }""")

df[' '] = df['table_img']
df['Title'] = df['grid_title']
gb = GridOptionsBuilder.from_dataframe(df[[' ','Title','popularity','score','genres','themes']].reset_index(drop=True))
gb.configure_column("Title",cellRenderer=JsCode('''function(params) {return params.value}'''))
gb.configure_column(" ",cellRenderer=image_nation)

# enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
gb.configure_default_column(enablePivot=False, enableValue=False, enableRowGroup=False)
gb.configure_selection(selection_mode="multiple", use_checkbox=False)
cellStyle= {'height': '100%','display': 'flex ','justify-content': 'center','align-items': 'center'}
gb.configure_grid_options(rowHeight=60,cellStyle=cellStyle)
gridOptions = gb.build()


st.success(
    f"""
        💡 Here is a list of similar anime!
        """
)

response = AgGrid(
    df,
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=False,
    allow_unsafe_jscode = True,
    theme = 'dark'
)
