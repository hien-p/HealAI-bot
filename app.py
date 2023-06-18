
import streamlit as st
from components.sidebar import show
from components.TextPage import renderPage
from components.imagePage import renderimagePage
from botchat import play_chat
# st.set_page_config(page_title="Heal.AI", page_icon=None, layout='centered', initial_sidebar_state='auto')



page = show()
if page ==  "Heal.AL":
    play_chat()
if page == 'Text':
    renderPage()
if page == 'Images':
    renderimagePage()
    
    
