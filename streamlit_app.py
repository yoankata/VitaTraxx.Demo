import numpy as np
import streamlit as st
from screens import home, generate_story, generate_noise, sleep_statistics, ai_recommendations
from screens import device_integration, today_statistics, rag_screen, deep_seek, image_typewriter
from screens import claude
import soundfile as sf
from gtts import gTTS
import streamlit_scrollable_textbox as stx
from difflib import get_close_matches

st.set_page_config(page_title="DreamWeaver AI", page_icon='💤', layout='wide')

PAGES = {
    "Home": home,
    "Today's Sleep Statistics": today_statistics,
    "Generate Story": generate_story,
    "Generate Noise": generate_noise,
    "Insights": sleep_statistics,
    "AI Recommendations": ai_recommendations,
    "Devices": device_integration,
    "Retrieval-Augmented Generation": rag_screen,
    "Image TypeWriter": image_typewriter,
    "DeepSeek": deep_seek,
    "Claude": claude,
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page.app()
