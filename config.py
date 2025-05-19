import streamlit as st

def init_config():
    if 'config' not in st.session_state:
        st.session_state.config = {
            'MAX_DAILY_MINUTES': 1440,
            'BREAK_TIME_PERCENTAGE': 0.03,
            'DOWN_TIME_PERCENTAGE': 0.05,
            'INITIAL_MAX_TERMINALS': 20,
            'MAX_SEALS_AC91': 3,
            'MAX_SEALS_AC96': 5,
            'MAX_SEALS_KOMAX': 4,
            'WIRE_CHANGE_TIME': 1,
            'AC_SEAL_CHANGE_TIME': 0.33,
            'AC_TERMINAL_CHANGE_TIME': 0.5,
            'KOMAX_SEAL_CHANGE_TIME': 6,
            'KOMAX_TERMINAL_CHANGE_TIME': 4,
            'K433_EFFICIENCY': 0.8,
            'K433H_EFFICIENCY': 0.8,
            'K355_EFFICIENCY': 0.7,
            'AC91_EFFICIENCY': 0.5,
            'AC96_EFFICIENCY': 1.0,  
            'K560_EFFICIENCY': 0.8
        }

def show_config_page():
    st.title("Configuration")
    
    st.header("General Parameters")
    st.session_state.config['MAX_DAILY_MINUTES'] = st.number_input("Max Daily Minutes", value=float(st.session_state.config['MAX_DAILY_MINUTES']))
    st.session_state.config['BREAK_TIME_PERCENTAGE'] = st.slider("Break Time Percentage", 0.0, 1.0, float(st.session_state.config['BREAK_TIME_PERCENTAGE']), 0.01)
    st.session_state.config['DOWN_TIME_PERCENTAGE'] = st.slider("Down Time Percentage", 0.0, 1.0, float(st.session_state.config['DOWN_TIME_PERCENTAGE']), 0.01)
    st.session_state.config['INITIAL_MAX_TERMINALS'] = st.number_input("Initial Max Terminals", value=int(st.session_state.config['INITIAL_MAX_TERMINALS']))

    st.header("Machine-Specific Parameters")
    st.session_state.config['MAX_SEALS_AC91'] = st.number_input("Max Seals AC91", value=int(st.session_state.config['MAX_SEALS_AC91']))
    st.session_state.config['MAX_SEALS_AC96'] = st.number_input("Max Seals AC96", value=int(st.session_state.config['MAX_SEALS_AC96']))
    st.session_state.config['MAX_SEALS_KOMAX'] = st.number_input("Max Seals KOMAX", value=int(st.session_state.config['MAX_SEALS_KOMAX']))

    st.header("Time Parameters")
    st.session_state.config['WIRE_CHANGE_TIME'] = st.number_input("Wire Change Time", value=float(st.session_state.config['WIRE_CHANGE_TIME']))
    st.session_state.config['AC_SEAL_CHANGE_TIME'] = st.number_input("AC Seal Change Time", value=float(st.session_state.config['AC_SEAL_CHANGE_TIME']))
    st.session_state.config['AC_TERMINAL_CHANGE_TIME'] = st.number_input("AC Terminal Change Time", value=float(st.session_state.config['AC_TERMINAL_CHANGE_TIME']))
    st.session_state.config['KOMAX_SEAL_CHANGE_TIME'] = st.number_input("KOMAX Seal Change Time", value=float(st.session_state.config['KOMAX_SEAL_CHANGE_TIME']))
    st.session_state.config['KOMAX_TERMINAL_CHANGE_TIME'] = st.number_input("KOMAX Terminal Change Time", value=float(st.session_state.config['KOMAX_TERMINAL_CHANGE_TIME']))

    st.header("Machine Efficiency Parameters")
    st.session_state.config['K433_EFFICIENCY'] = st.slider("K433 Efficiency", 0.0, 1.0, float(st.session_state.config['K433_EFFICIENCY']), 0.01)
    st.session_state.config['K433H_EFFICIENCY'] = st.slider("K433H Efficiency", 0.0, 1.0, float(st.session_state.config['K433H_EFFICIENCY']), 0.01)
    st.session_state.config['K355_EFFICIENCY'] = st.slider("K355 Efficiency", 0.0, 1.0, float(st.session_state.config['K355_EFFICIENCY']), 0.01)
    st.session_state.config['AC91_EFFICIENCY'] = st.slider("AC91 Efficiency", 0.0, 1.0, float(st.session_state.config['AC91_EFFICIENCY']), 0.01)
    st.session_state.config['AC96_EFFICIENCY'] = st.slider("AC96 Efficiency", 0.0, 1.0, float(st.session_state.config['AC96_EFFICIENCY']), 0.01)
    st.session_state.config['K560_EFFICIENCY'] = st.slider("K560 Efficiency", 0.0, 1.0, float(st.session_state.config['K560_EFFICIENCY']), 0.01)

def get_config():
    return st.session_state.config