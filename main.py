import streamlit as st
import pyrebase
from streamlit_option_menu import option_menu
import pandas as pd
import geocoder
import folium

#Firebase config keys

firebaseConfig = {
  'apiKey': "AIzaSyDa8RyPDwMVj9Tw0XyPi7QSdcOOh2_wkHY",
  'authDomain': "stay-safe-fa2dd.firebaseapp.com",
  'databaseURL': "https://stay-safe-fa2dd-default-rtdb.firebaseio.com",
  'projectId': "stay-safe-fa2dd",
  'storageBucket': "stay-safe-fa2dd.appspot.com",
  'messagingSenderId': "765879193502",
  'appId': "1:765879193502:web:53588b7dd1cb2bcaf4108f",
  'measurementId': "G-Q5VGFLKBXH"
};


# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()

def maps():
    ip = geocoder.ip("202.51.247.22")
    ip.latlng

    data = {'Camera Name': ['PGPR', 'Com2'],
            'Latitude': ['1.291654', '1.294108'],
            'Longitude': ['103.780445', '103.773765']
            }

    df = pd.DataFrame(data)

    print(df)

    x = int(1)
    location = [df['Latitude'][x], df['Longitude'][x]]
    map = folium.Map(location=location, zoom_start=10)
    folium.Marker(location).add_to(map)
    map


firebaseConfig = {
  'apiKey': "AIzaSyDa8RyPDwMVj9Tw0XyPi7QSdcOOh2_wkHY",
  'authDomain': "stay-safe-fa2dd.firebaseapp.com",
  'databaseURL': "https://stay-safe-fa2dd-default-rtdb.firebaseio.com",
  'projectId': "stay-safe-fa2dd",
  'storageBucket': "stay-safe-fa2dd.appspot.com",
  'messagingSenderId': "765879193502",
  'appId': "1:765879193502:web:53588b7dd1cb2bcaf4108f",
  'measurementId': "G-Q5VGFLKBXH"
};


# model

#@st.cache(suppress_st_warning=True)


st.sidebar.title("Welcome to Carma")

choice = st.sidebar.selectbox("Login/Signup", ["Login","Signup"])

if choice == "Signup":
    email = st.sidebar.text_input("Enter your email address")
    password = st.sidebar.text_input("Enter your password", type="password")
    st.sidebar.text_input("Region/Department")
    Signup = st.sidebar.button("Signup")
    if Signup:
        user = auth.create_user_with_email_and_password(email,password)
        st.balloons()
        st.success("Account Created")

elif choice == "Login":
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password",type="password")
    Login = st.sidebar.button("Login")
    forgot = st.sidebar.button("Forgot Password")
    if Login:
        user = auth.sign_in_with_email_and_password(email, password)
        st.balloons()
        st.title("Carma")
        # Navigation Pane
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Home", "Maps", "Settings"],
            orientation="horizontal"
        )
    if  selected == "Home":
        pass
    if selected == "Maps":
        st.title("Maps")
        maps()
    if selected == "Settings":
        st.title("Settings")
