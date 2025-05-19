import pandas as pd
import numpy as np
from config import get_config

def calculate_K433(row):
    config = get_config()
    X1 = row['WireLength']
    X2 = row['Variant_Numeric']
    CS = row['CrossSection']
    if pd.isna(X1) or pd.isna(X2) or pd.isna(CS):
        return np.nan
    base = 1.40372 + 0.00022376 * X1 + 0.19592 * X2
    if 0 <= X1 <= 6000 and 0 <= CS <= 4:
        return base / config['K433_EFFICIENCY']
    elif 6000 < X1 <= 20000 and 0 <= CS <= 4:
        return (base * 1.14) / config['K433_EFFICIENCY']
    elif 0 <= X1 <= 6000 and 4 < CS <= 6:
        return (base * 1.1) / config['K433_EFFICIENCY']
    elif 6000 < X1 <= 20000 and 4 < CS <= 6:
        return (base * 1.14 * 1.1) / config['K433_EFFICIENCY']
    return np.nan

def calculate_K433H(row):
    config = get_config()
    X1 = row['WireLength']
    X2 = row['CrossSection']
    variant = row['Variant_Numeric']
    if pd.isna(X1) or pd.isna(X2) or pd.isna(variant):
        return np.nan
    
    if 0 <= X1 <= 6000 and 0 <= X2 <= 4:
        return (1.40372 + 0.00022376 * X1 + 0.19592 * variant) / config['K433H_EFFICIENCY']
    elif 6000 < X1 <= 20000 and 0 <= X2 <= 4:
        return ((1.40372 + 0.00022376 * X1 + 0.19592 * variant) * 1.14) / config['K433H_EFFICIENCY']
    elif 0 <= X1 <= 6500 and 4 <= X2 <= 10:
        base = 0.58806 + 0.00034964 * X1 + 0.21324 * X2
        if variant == 1:
            return (base + 0.69196 * 1) / config['K433H_EFFICIENCY']
        elif variant == 2:
            return (base + 0.69196 * 2) / config['K433H_EFFICIENCY']
        elif variant == 3:
            return (base + 0.69196 * 3) / config['K433H_EFFICIENCY']
        elif variant == 4:
            return (base + 0.69196 * 1 + 0.19592) / config['K433H_EFFICIENCY']
        elif variant == 5:
            return (base + 0.69196 * 1 + 0.39184) / config['K433H_EFFICIENCY']
        elif variant == 6:
            return (base + 0.69196 * 1 + 0.58776) / config['K433H_EFFICIENCY']
    return np.nan

def calculate_K355(row):
    config = get_config()
    X1 = row['WireLength']
    X2 = row['Variant_Numeric']
    X3 = X2  
    CS = row['CrossSection']
    if pd.isna(X1) or pd.isna(X2) or pd.isna(CS):
        return np.nan
    
    if 0 <= X1 <= 6000 and 0 <= CS <= 2.5:
        return (0.85757 + 0.00023248 * X1 + 0.25262 * X2) / config['K355_EFFICIENCY']
    elif 6000 < X1 <= 12000 and 0 <= CS <= 2.5:
        return ((0.85757 + 0.00023248 * X1 + 0.25262 * X2) * 1.14) / config['K355_EFFICIENCY']
    elif 0 <= X1 <= 12000 and 2.5 < CS <= 4:
        return (((0.70874 + 0.00019362 * X1 + 0.20878 * X3) * 1.2) * 1.14) / config['K355_EFFICIENCY']
    elif 0 <= X1 <= 12000 and 4 < CS <= 8:
        return (((0.70874 + 0.00019362 * X1 + 0.20878 * X3) * 1.3) * 1.14) / config['K355_EFFICIENCY']
    return np.nan

def calculate_AC91(row):
    config = get_config()
    X1 = row['WireLength']
    X2 = row['Variant_Numeric']
    CS = row['CrossSection']
    if pd.isna(X1) or pd.isna(X2) or pd.isna(CS):
        return np.nan
    
    if 0 <= X1 <= 10000 and 0.13 <= CS <= 2.5:
        return (0.55614 + 0.00029519 * X1 + 0.83713 * np.sqrt(X2)) / config['AC91_EFFICIENCY']
    elif 0 <= X1 <= 5000 and 2.5 <= CS <= 5:
        if X2 in [1, 2, 3]:
            return (1.26642 + 0.00040195 * X1 + 0.1 * X2) / config['AC91_EFFICIENCY']
        elif X2 in [4, 5, 6]:
            return (2.00064 + 0.00039759 * X1 + 0.055396 * X2) / config['AC91_EFFICIENCY']
    elif 5000 < X1 <= 10000 and 2.5 <= CS <= 5:
        if X2 in [1, 2, 3]:
            return (1.47020 + 0.00040389 * X1 + 0.15 * X2) / config['AC91_EFFICIENCY']
        elif X2 in [4, 5, 6]:
            return (1.60353 + 0.00040389 * X1 + 0.2 * X2) / config['AC91_EFFICIENCY']
    elif row['Variant'] == 'Strip' and 0 <= X1 <= 10000 and 0.22 <= CS <= 2.5:
        return (4.61567 + 0.00039914 * X1) / config['AC91_EFFICIENCY']
    return np.nan

def calculate_AC96(row):
    config = get_config()
    X1 = row['WireLength']
    variant = row['Variant_Numeric']
    CS = row['CrossSection']
    if pd.isna(X1) or pd.isna(variant) or pd.isna(CS):
        return np.nan
    
    if 0 <= X1 <= 500 and 0.25 <= CS <= 2.5:
        return [1.79, 1.99, 2.09, 2.8, 3.05, 4.85][int(variant) - 1] / config['AC96_EFFICIENCY']
    elif 501 <= X1 <= 4500 and 0.25 <= CS <= 2.5:
        if variant == 1:
            return (1.69577 + 0.000000044385419 * X1**2 + 0.0859537037037037) / config['AC96_EFFICIENCY']
        elif variant == 2:
            return (1.90364 + 0.0000000000101267 * X1**3 + 0.0859537037037037) / config['AC96_EFFICIENCY']
        elif variant == 3:
            return (2.00107 + 2.12285E-15 * X1**4 + 0.0859537037037037) / config['AC96_EFFICIENCY']
        elif variant == 4:
            return (2.7091 + 9.74994E-16 * X1**4 + 0.0859537037037037) / config['AC96_EFFICIENCY']
        elif variant == 5:
            return (2.93662 + 0.000064494 * X1 + 0.0859537037037037) / config['AC96_EFFICIENCY']
        elif variant == 6:
            return (5.4227 - 329.6057 * 1/X1 + 0.0859537037037037) / config['AC96_EFFICIENCY']
    return np.nan

def calculate_K560(row):
    config = get_config()
    X1 = row['WireLength']
    variant = row['Variant_Numeric']
    CS = row['CrossSection']
    if pd.isna(X1) or pd.isna(variant) or pd.isna(CS):
        return np.nan
    
    if 0 <= X1 <= 12000 and 0.35 <= CS <= 4:
        base = 1.03178 + 0.00015469 * X1
        return (base + 0.204 * (variant - 1)) / config['K560_EFFICIENCY']
    return np.nan


def combine_terms(row):
    if pd.isna(row['Terminal1Key']) and pd.isna(row['Terminal2Key']):
        if pd.isna(row['Seal1Key']) and pd.isna(row['Seal2Key']):
            return 'ST-ST'
    elif not pd.isna(row['Terminal1Key']) and not pd.isna(row['Terminal2Key']):
        if pd.isna(row['Seal1Key']) and not pd.isna(row['Seal2Key']):
            return 'CS-C'
        elif not pd.isna(row['Seal1Key']) and pd.isna(row['Seal2Key']):
            return 'CS-C'
        elif not pd.isna(row['Seal1Key']) and not pd.isna(row['Seal2Key']):
            return 'CS-CS'
        elif pd.isna(row['Seal1Key']) and pd.isna(row['Seal2Key']):
            return 'C-C'
    elif not pd.isna(row['Terminal1Key']) and pd.isna(row['Terminal2Key']):
        if pd.isna(row['Seal1Key']) and pd.isna(row['Seal2Key']):
            return 'C-ST'
        if not pd.isna(row['Seal1Key']) and pd.isna(row['Seal2Key']):
            return 'CS-ST'
    elif pd.isna(row['Terminal1Key']) and not pd.isna(row['Terminal2Key']):
        if pd.isna(row['Seal1Key']) and pd.isna(row['Seal2Key']):
            return 'C-ST'        
        if pd.isna(row['Seal1Key']) and not pd.isna(row['Seal2Key']):
            return 'CS-ST'
    return 'ST-ST'

def variant_to_numeric(variant):
    variant_mapping = {
        'ST-ST': 1, 'C-ST': 2, 'C-C': 3, 'CS-ST': 4, 'CS-C': 5, 'CS-CS': 6
    }
    if pd.isna(variant):
        return np.nan
    if isinstance(variant, (int, float)):
        return variant
    if isinstance(variant, str):
        return variant_mapping.get(variant, np.nan)
    return np.nan

##---------------------- Wire Classification ----------------------------##

def Classify_Wire_Section(CrossSection):
    if 0.13 <= CrossSection <= 1:
        return 'Type A'
    elif 1 < CrossSection <= 2.5:
        return 'Type B'
    elif 3 <= CrossSection <= 6:
        return 'Type C'
    elif 6 < CrossSection <= 10:
        return 'Type D'
    else:
        return 'Type E'

def classify_wire_length(WireLength):
    if WireLength <= 4000:
        return 'Type 1'
    elif 4000 < WireLength <= 5000:
        return 'Type 2'
    elif 5000 < WireLength <= 8000:
        return 'Type 3'
    else:
        return 'Type 4'

def swap_seals_and_terminals(row):
    if pd.isna(row['Seal1Key']) and not pd.isna(row['Seal2Key']):
        row['Seal1Key'], row['Seal2Key'] = row['Seal2Key'], row['Seal1Key']
        row['Terminal1Key'], row['Terminal2Key'] = row['Terminal2Key'], row['Terminal1Key']
    elif not pd.isna(row['Seal1Key']) and not pd.isna(row['Seal2Key']) and row['Seal1Key'] > row['Seal2Key']:
        row['Seal1Key'], row['Seal2Key'] = row['Seal2Key'], row['Seal1Key']
        row['Terminal1Key'], row['Terminal2Key'] = row['Terminal2Key'], row['Terminal1Key']
    return row

def swap(row):
    
    row['Seal1Key'], row['Seal2Key'] = row['Seal2Key'], row['Seal1Key']
    row['Terminal1Key'], row['Terminal2Key'] = row['Terminal2Key'], row['Terminal1Key']
    
    return row
