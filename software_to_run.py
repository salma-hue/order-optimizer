import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import base64
from PIL import Image
from typing import Dict, List, Tuple, Any
from PreProcess2 import (calculate_K433, calculate_K433H, calculate_K355, calculate_AC91, calculate_AC96, calculate_K560, 
                         variant_to_numeric, Classify_Wire_Section, classify_wire_length, combine_terms, 
                         swap_seals_and_terminals, swap)
from config import init_config, show_config_page, get_config
st.set_page_config(page_title="Order Distribution Process Optimizer", page_icon="🔧", layout="wide")
init_config()

MACHINES = {}
PE_SETTINGS = {}

def get_constants():
    config = get_config()
    return {
        'MAX_DAILY_MINUTES': config['MAX_DAILY_MINUTES'],
        'BREAK_TIME_PERCENTAGE': config['BREAK_TIME_PERCENTAGE'],
        'DOWN_TIME_PERCENTAGE': config['DOWN_TIME_PERCENTAGE'],
        'INITIAL_MAX_TERMINALS': config['INITIAL_MAX_TERMINALS'],
        'MAX_SEALS_AC91': config['MAX_SEALS_AC91'],
        'MAX_SEALS_AC96': config['MAX_SEALS_AC96'],
        'MAX_SEALS_KOMAX': config['MAX_SEALS_KOMAX'],
        'WIRE_CHANGE_TIME': config['WIRE_CHANGE_TIME'],
        'AC_SEAL_CHANGE_TIME': config['AC_SEAL_CHANGE_TIME'],
        'AC_TERMINAL_CHANGE_TIME': config['AC_TERMINAL_CHANGE_TIME'],
        'KOMAX_SEAL_CHANGE_TIME': config['KOMAX_SEAL_CHANGE_TIME'],
        'KOMAX_TERMINAL_CHANGE_TIME': config['KOMAX_TERMINAL_CHANGE_TIME']
    }

def load_machines_from_csv(file):
    machines_df = pd.read_csv(file)
    return {row['MachineName']: {'type': row['MachineType']} for _, row in machines_df.iterrows()}

def assign_ranges_to_machines(machines: Dict[str, Dict], orders_df: pd.DataFrame) -> Dict[str, Dict]:
    machine_types = set(machine['type'] for machine in machines.values())
    
    ranges = {
        'K433': [(0.35, 3), (3, 7)],
        'K433H': [(3, 10)],
        'K560': [(3, 10)],
        'AC91': [(0.13, 0.35), (0.35, 2.5), (2.5, 6)],
        'AC96': [(0.13, 0.35), (0.35, 2.5), (2.5, 6)],
        'K355': [(0.13, 0.35), (0.35, 3), (3, 6)]
    }
    
    range_demand = {machine_type: {range: 0 for range in ranges[machine_type]} for machine_type in machine_types}
    for _, order in orders_df.iterrows():
        cs = order['CrossSection']
        for machine_type in machine_types:
            for range in ranges[machine_type]:
                if range[0] <= cs < range[1]:
                    range_demand[machine_type][range] += 1
    
    assigned_ranges = {machine_type: [] for machine_type in machine_types}
    
    for machine_name, machine_info in machines.items():
        machine_type = machine_info['type']
        available_ranges = [r for r in ranges[machine_type] if r not in assigned_ranges[machine_type]]
        
        if not available_ranges:
            # If all ranges are assigned, choose the range with the highest demand
            chosen_range = max(ranges[machine_type], key=lambda r: range_demand[machine_type][r])
        else:
            # Choose the available range with the highest demand
            chosen_range = max(available_ranges, key=lambda r: range_demand[machine_type][r])
        
        machine_info['min_cross_section'] = chosen_range[0]
        machine_info['max_cross_section'] = chosen_range[1]
        
        assigned_ranges[machine_type].append(chosen_range)
        range_demand[machine_type][chosen_range] -= 1
    
    return machines


def dynamically_assign_ranges(machines: Dict[str, Dict], orders_df: pd.DataFrame, use_whole_data: bool) -> Dict[str, Dict]:
    machine_types = set(machine['type'] for machine in machines.values())
    
    ranges = {
        'K433': [(0.35, 3), (3, 7)],
        'K433H': [(3, 10)],
        'K560': [(3, 10)],
        'AC91': [(0.13, 0.35), (0.35, 2.5), (2.5, 6)],
        'AC96': [(0.13, 0.35), (0.35, 2.5), (2.5, 6)],
        'K355': [(0.13, 0.35), (0.35, 3), (3, 6)]
    }
    
    range_demand = {machine_type: {range: 0 for range in ranges[machine_type]} for machine_type in machine_types}
    for _, order in orders_df.iterrows():
        cs = order['CrossSection']
        for machine_type in machine_types:
            for range in ranges[machine_type]:
                if range[0] <= cs < range[1]:
                    range_demand[machine_type][range] += 1
    
    assigned_ranges = {machine_type: [] for machine_type in machine_types}
    
    for machine_name, machine_info in machines.items():
        machine_type = machine_info['type']
        available_ranges = [r for r in ranges[machine_type] if r not in assigned_ranges[machine_type]]
        
        if use_whole_data or not available_ranges:
            chosen_range = max(ranges[machine_type], key=lambda r: range_demand[machine_type][r])
        else:
            valid_ranges = [r for r in available_ranges if range_demand[machine_type][r] > 0]
            if valid_ranges:
                chosen_range = max(valid_ranges, key=lambda r: range_demand[machine_type][r])
            else:
                chosen_range = max(available_ranges, key=lambda r: range_demand[machine_type][r])
        
        machine_info['min_cross_section'] = chosen_range[0]
        machine_info['max_cross_section'] = chosen_range[1]
        
        assigned_ranges[machine_type].append(chosen_range)
        range_demand[machine_type][chosen_range] -= 1
    
    return machines

def initialize_machines(machines: Dict[str, Dict]) -> Dict[str, Dict]:
    return {machine: {
        'type': info['type'],
        'min_cross_section': info.get('min_cross_section', 0),
        'max_cross_section': info.get('max_cross_section', float('inf')),
        'processing_time': 0,
        'setup_time': 0,
        'break_time': 0,
        'down_time': 0,
        'total_time': 0,
        'orders': [],
        'seals_left': set(),
        'seals_right': set(),
        'terminals': set(),
        'current_wire': None,
        'last_setup': {'seals_left': set(), 'seals_right': set(), 'terminals': set(), 'current_wire': None}
    } for machine, info in machines.items()}

def preprocess_orders(orders_df):
    orders_df['Variant'] = orders_df.apply(combine_terms, axis=1)
    orders_df = orders_df.apply(swap_seals_and_terminals, axis=1)
    orders_df['Variant_Numeric'] = orders_df['Variant'].apply(variant_to_numeric)
    
    orders_df['WireLength'] = pd.to_numeric(orders_df['WireLength'], errors='coerce')
    orders_df['CrossSection'] = pd.to_numeric(orders_df['CrossSection'], errors='coerce')

    orders_df['Time_K433'] = orders_df.apply(calculate_K433, axis=1)
    orders_df['Time_K433H'] = orders_df.apply(calculate_K433H, axis=1)
    orders_df['Time_K355'] = orders_df.apply(calculate_K355, axis=1)
    orders_df['Time_AC91'] = orders_df.apply(calculate_AC91, axis=1)
    orders_df['Time_AC96'] = orders_df.apply(calculate_AC96, axis=1)
    orders_df['Time_K560'] = orders_df.apply(calculate_K560, axis=1)
    orders_df['CrossSectionType'] = orders_df['CrossSection'].apply(Classify_Wire_Section) 
    orders_df['WireLengthType'] = orders_df['WireLength'].apply(classify_wire_length)
    
    if 'Quantity' in orders_df.columns:
        for col in ['Time_K433', 'Time_K433H', 'Time_K355', 'Time_AC91', 'Time_AC96', 'Time_K560']:
            orders_df[col] = orders_df[col] * orders_df['Quantity']/100
    else:
        st.warning("'Quantity' column not found. Assuming quantity of 1 for all orders.")
    orders_df.to_csv('hh.csv')
    return orders_df

def calculate_setup_time(current_setup: Dict, new_setup: Dict, machine_type: str, wire_changed: bool,order_change) -> float:
    constants = get_constants()
    setup_time = order_change*0.3
    
    if machine_type.startswith('AC'):
        if current_setup['seals_left'] != new_setup['seals_left']:
            setup_time += constants['AC_SEAL_CHANGE_TIME']
        if current_setup['seals_right'] != new_setup['seals_right']:
            setup_time += constants['AC_SEAL_CHANGE_TIME']
        
        terminal_changes = len(current_setup['terminals'] ^ new_setup['terminals'])
        setup_time += terminal_changes * constants['AC_TERMINAL_CHANGE_TIME']
    else:  # KOMAX machines
        if current_setup['seals_left'] != new_setup['seals_left']:
            setup_time += constants['KOMAX_SEAL_CHANGE_TIME']
        if current_setup['seals_right'] != new_setup['seals_right']:
            setup_time += constants['KOMAX_SEAL_CHANGE_TIME']
        
        terminal_changes = len(current_setup['terminals'] ^ new_setup['terminals'])
        setup_time += terminal_changes * constants['KOMAX_TERMINAL_CHANGE_TIME']
    
    if wire_changed:
        setup_time += constants['WIRE_CHANGE_TIME']
    return setup_time

def calculate_total_time(processing_time: float, setup_time: float) -> float:
    constants = get_constants()
    base_time = processing_time + setup_time
    break_time = base_time * constants['BREAK_TIME_PERCENTAGE']
    down_time = base_time * constants['DOWN_TIME_PERCENTAGE']
    return base_time + break_time + down_time

def calculate_assignment_score(machine_info: Dict, group: pd.DataFrame, new_total_time: float, max_terminals: int, avg_machine_time: float) -> float:
    new_seals_left = set(group['Seal1Key']) - {-1}
    new_seals_right = set(group['Seal2Key']) - {-1}
    new_terminals = set(group['Terminal1Key']) | set(group['Terminal2Key']) - {-1}
    
    seal_changes = len(new_seals_left - machine_info['seals_left']) + len(new_seals_right - machine_info['seals_right'])
    terminal_changes = len(new_terminals - machine_info['terminals'])
    
    setup_change_score = (seal_changes*5 + terminal_changes) / (max_terminals + get_constants()['MAX_SEALS_AC96'] * 2)
    balance_score = abs(new_total_time - avg_machine_time) / get_constants()['MAX_DAILY_MINUTES']

    if machine_info['type'] == 'AC96':
        score= (setup_change_score*2 + balance_score)
    else:
        score=(setup_change_score*2 + balance_score)
    return score

def can_assign_to_machine(order: pd.DataFrame, machine_info: Dict, max_terminals: int = get_constants()['INITIAL_MAX_TERMINALS'], 
                          avg_machine_time: float = 0) -> Tuple[bool, float, str]:
    machine_type = machine_info['type']
    time_column = f'Time_{machine_type}'
    
    cross_section = order['CrossSection'].iloc[0]
    if not (machine_info['min_cross_section'] <= cross_section < machine_info['max_cross_section']):
        return False, float('inf'), f"Cross-section {cross_section} out of range {machine_info['min_cross_section']}-{machine_info['max_cross_section']} for this machine"
    
    if order[time_column].isnull().all():
        return False, float('inf'), "Incompatible machine type"
    
    processing_time = order[time_column].sum()
    
    seals_left = set(order['Seal1Key'].unique()) - {-1}
    seals_right = set(order['Seal2Key'].unique()) - {-1}
    terminals = set(order['Terminal1Key'].unique()) | set(order['Terminal2Key'].unique()) - {-1}
    
    new_setup = {
        'seals_left': seals_left,
        'seals_right': seals_right,
        'terminals': terminals,
        'current_wire': order['Wire1Key'].iloc[0]  
    }
    wire_changed = new_setup['current_wire'] != machine_info['current_wire']  
    setup_time = calculate_setup_time(machine_info['last_setup'], new_setup, machine_type, wire_changed,order_change=len(order)) 
    
    new_total_time = machine_info['total_time'] + calculate_total_time(processing_time, setup_time)
    
    if new_total_time > get_constants()['MAX_DAILY_MINUTES']:
        return False, float('inf'), "Exceeds daily time limit"
    
    if len(machine_info['terminals'] | terminals) > max_terminals:
        return False, float('inf'), "Exceeds terminal limit"
    
    if machine_type.startswith('AC'):
        new_seals_left = machine_info['seals_left'] | seals_left
        new_seals_right = machine_info['seals_right'] | seals_right
        max_seals = get_constants()['MAX_SEALS_AC91'] if machine_type == 'AC91' else get_constants()['MAX_SEALS_AC96']
        if len(new_seals_left) > max_seals or len(new_seals_right) > max_seals:
            return False, float('inf'), f"Exceeds seal limit for {machine_type}"
    else:  # KOMAX machines
        new_seals = (machine_info['seals_left'] | machine_info['seals_right'] | seals_left | seals_right) - {-1}
        if len(new_seals) > get_constants()['MAX_SEALS_KOMAX']:
            return False, float('inf'), f"Exceeds seal limit for Komax ({len(new_seals)} > {get_constants()['MAX_SEALS_KOMAX']})"

    score = calculate_assignment_score(machine_info, order, new_total_time, max_terminals, avg_machine_time)
    
    return True, score, "Assignable"

def assign_group_to_machine(group: pd.DataFrame, machine_info: Dict, machine: str):
    machine_type = machine_info['type']
    time_column = f'Time_{machine_type}'
    
    new_setup = {
        'seals_left': set(group['Seal1Key'].unique()) - {-1},
        'seals_right': set(group['Seal2Key'].unique()) - {-1},
        'terminals': set(group['Terminal1Key'].unique()) | set(group['Terminal2Key'].unique()) - {-1},
        'current_wire': group['Wire1Key'].iloc[0]
    }
    print("the len grp :", len(group))
    wire_changed = new_setup['current_wire'] != machine_info['current_wire']
    order_change=len(group)
    new_setup_time = calculate_setup_time(machine_info['last_setup'], new_setup, machine_type, wire_changed,order_change)
    new_setup_t = machine_info['setup_time']+calculate_setup_time(machine_info['last_setup'], new_setup, machine_type, wire_changed,order_change)
    new_processing_time = machine_info['processing_time'] + group[time_column].sum()
    new_total_time = calculate_total_time(new_processing_time, new_setup_t)
        # Check if adding this group would exceed the daily time limit
    if new_total_time > get_constants()['MAX_DAILY_MINUTES']:
        return False  # Indicate that the assignment failed
    machine_info['processing_time'] = new_processing_time
    machine_info['setup_time'] += new_setup_time
    machine_info['break_time'] = new_total_time * get_constants()['BREAK_TIME_PERCENTAGE']
    machine_info['down_time'] = new_total_time * get_constants()['DOWN_TIME_PERCENTAGE']
    machine_info['total_time'] = new_total_time
    machine_info['orders'].extend(group['N'].tolist())
    
    machine_info['seals_left'].update(new_setup['seals_left'])
    machine_info['seals_right'].update(new_setup['seals_right'])
    machine_info['terminals'].update(new_setup['terminals'])
    machine_info['current_wire'] = new_setup['current_wire']
    machine_info['last_setup'] = new_setup
    return True

def assign_orders_with_utilization(machine_data: Dict[str, Dict], sorted_groups: List[Tuple[Any, pd.DataFrame]], 
                                   max_terminals: int = get_constants()['INITIAL_MAX_TERMINALS']) -> Tuple[Dict[str, Dict], List[int], Dict[int, str]]:
    unassigned_orders = []
    unassigned_reasons = {}
    active_machines = []

    for _, group in sorted_groups:
        assigned = False
        best_machine = None
        best_score = float('inf')
        best_reason = "No compatible machine found"
        
        total_time = sum(info['total_time'] for info in machine_data.values())
        avg_machine_time = total_time / len(machine_data) if machine_data else 0
        
        compatible_machines = list(machine_data.keys())
        
        compatible_machines.sort(key=lambda m: sum(group[f'Time_{machine_data[m]["type"]}'].notna()))
        
        for machine in compatible_machines:
            machine_info = machine_data[machine]
            can_assign, score, reason = can_assign_to_machine(group, machine_info, max_terminals, avg_machine_time)
            if can_assign and score < best_score:
                best_score = score
                best_machine = machine
                best_reason = reason
            elif not can_assign and reason != "Incompatible machine type":
                best_reason = reason

        if best_machine:
            machine_info = machine_data[best_machine]
            assign_group_to_machine(group, machine_info, best_machine)
            if best_machine not in active_machines:
                active_machines.append(best_machine)
            assigned = True
        
        if not assigned:
            unassigned_orders.extend(group['N'].tolist())
            for order in group['N']:
                unassigned_reasons[order] = best_reason

    return machine_data, unassigned_orders, unassigned_reasons

def reassign_order(order: int, from_machine: Dict, to_machine: Dict, orders_df: pd.DataFrame):
    order_data = orders_df[orders_df['N'] == order]
    time_column = f"Time_{to_machine['type']}"

    # Calculate new setup for to_machine
    new_setup = {
        'seals_left': set(order_data['Seal1Key']) - {-1},
        'seals_right': set(order_data['Seal2Key']) - {-1},
        'terminals': set(order_data['Terminal1Key']) | set(order_data['Terminal2Key']) - {-1},
        'current_wire': order_data['Wire1Key'].iloc[0]
    }
    wire_changed = new_setup['current_wire'] != to_machine['current_wire']
    new_setup_time = calculate_setup_time(to_machine['last_setup'], new_setup, to_machine['type'], wire_changed,order_change=1)

    new_processing_time = to_machine['processing_time'] + order_data[time_column].sum()
    new_total_time = calculate_total_time(new_processing_time, to_machine['setup_time'] + new_setup_time)

    if new_total_time > get_constants()['MAX_DAILY_MINUTES']:
        return False

    if from_machine is not None:
        from_machine['orders'].remove(order)
        update_machine_after_reassign(from_machine, order_data, remove=True)

    to_machine['orders'].append(order)
    update_machine_after_reassign(to_machine, order_data, remove=False, new_setup=new_setup,
                                  new_setup_time=new_setup_time)
    return True

def update_machine_after_reassign(machine: Dict, order_data: pd.DataFrame, remove: bool, new_setup: Dict = None,
                                  new_setup_time: float = 0):
    time_column = f"Time_{machine['type']}"
    time_change = order_data[time_column].sum()

    if remove:
        machine['processing_time'] -= time_change
        machine['seals_left'] -= set(order_data['Seal1Key']) - {-1}
        machine['seals_right'] -= set(order_data['Seal2Key']) - {-1}
        machine['terminals'] -= set(order_data['Terminal1Key']) | set(order_data['Terminal2Key']) - {-1}
        # Recalculate setup time for the machine after removal
        machine['setup_time'] = calculate_setup_time(machine['last_setup'], {
            'seals_left': machine['seals_left'],
            'seals_right': machine['seals_right'],
            'terminals': machine['terminals'],
            'current_wire': machine['current_wire']
        }, machine['type'], False,order_change=1)
    else:
        machine['processing_time'] += time_change
        machine['seals_left'].update(new_setup['seals_left'])
        machine['seals_right'].update(new_setup['seals_right'])
        machine['terminals'].update(new_setup['terminals'])
        machine['setup_time'] += new_setup_time
        machine['current_wire'] = new_setup['current_wire']
        machine['last_setup'] = new_setup

    # Recalculate total time
    machine['total_time'] = calculate_total_time(machine['processing_time'], machine['setup_time'])

    # Update break time and down time
    machine['break_time'] = machine['total_time'] * get_constants()['BREAK_TIME_PERCENTAGE']
    machine['down_time'] = machine['total_time'] * get_constants()['DOWN_TIME_PERCENTAGE']


def resolve_unassigned_orders(machine_data: Dict[str, Dict], unassigned_orders: List[int], orders_df: pd.DataFrame):
    still_unassigned = []
    
    for order in unassigned_orders:
        order_data = orders_df[orders_df['N'] == order]
        assigned = False
        
        # Try to assign the order in its original orientation
        for machine_name, machine in machine_data.items():
            can_assign, score, reason = can_assign_to_machine(order_data, machine)
            if can_assign:
                if reassign_order(order, None, machine, orders_df):
                    assigned = True
                    break
        
        # If not assigned, try with swapped orientation
        if not assigned:
            swapped_order_data = order_data.apply(swap, axis=1)
            for machine_name, machine in machine_data.items():
                can_assign, score, reason = can_assign_to_machine(swapped_order_data, machine)
                if can_assign:
                    if reassign_order(order, None, machine, swapped_order_data):
                        # Update the original orders_df with the swapped data
                        orders_df.loc[orders_df['N'] == order] = swapped_order_data
                        assigned = True
                        break
        
        if not assigned:
            still_unassigned.append(order)
    
    return machine_data, still_unassigned


def optimize_with_load_balancing(sorted_groups: List[Tuple[Any, pd.DataFrame]], orders_df: pd.DataFrame, min_machines: int):
    all_solutions = []
    available_machines = MACHINES.copy()

    for num_machines in range(min_machines, len(available_machines) + 1):
        print(f"\nTrying with {num_machines} machines")
        
        machine_list = list(available_machines.keys())
        machine_subset = machine_list[:num_machines]
        
        machine_data = initialize_machines({machine: available_machines[machine] for machine in machine_subset})
        
        # Step 1: Assign initial ranges based on total data
        machine_data = assign_ranges_to_machines(machine_data, orders_df)
        
        # Step 2: Attempt to assign orders
        machine_data, unassigned_orders, unassigned_reasons = assign_orders_with_utilization(machine_data, sorted_groups)
        
        # Calculate efficiency and create solution
        total_time_used = sum(info['total_time'] for info in machine_data.values())
        total_available_time = len(machine_data) * get_constants()['MAX_DAILY_MINUTES']
        efficiency = total_time_used / total_available_time

        initial_solution = {
            'num_machines': num_machines,
            'efficiency': efficiency,
            'unassigned_orders': len(unassigned_orders),
            'machine_data': machine_data,
            'unassigned': unassigned_orders,
            'unassigned_reasons': unassigned_reasons,
            'constraint_type': 'initial'
        }
        all_solutions.append(initial_solution)

        print(f"Initial assignment - Efficiency: {efficiency:.2f}, Unassigned orders: {len(unassigned_orders)}")

        # Step 3: If there are unassigned orders, try to accommodate them within existing machines
        if unassigned_orders:
            unassigned_df = orders_df[orders_df['N'].isin(unassigned_orders)]
            
            # Try to assign unassigned orders to existing machines without changing their ranges
            machine_data, still_unassigned_orders = reassign_unassigned_orders(machine_data, unassigned_df)
            
            # Calculate efficiency and create solution after reassignment attempt
            total_time_used = sum(info['total_time'] for info in machine_data.values())
            efficiency = total_time_used / total_available_time

            reassigned_solution = {
                'num_machines': num_machines,
                'efficiency': efficiency,
                'unassigned_orders': len(still_unassigned_orders),
                'machine_data': machine_data,
                'unassigned': still_unassigned_orders,
                'unassigned_reasons': {order: "Unable to assign within existing constraints" for order in still_unassigned_orders},
                'constraint_type': 'reassigned'
            }
            all_solutions.append(reassigned_solution)

            print(f"Reassignment attempt - Efficiency: {efficiency:.2f}, Unassigned orders: {len(still_unassigned_orders)}")

            unassigned_orders = still_unassigned_orders

        # If all orders are assigned, end the process
        if not unassigned_orders:
            print(f"Found optimal solution with {num_machines} machines and efficiency {efficiency:.2f}")
            break

    return all_solutions

def reassign_unassigned_orders(machine_data: Dict[str, Dict], unassigned_df: pd.DataFrame) -> Tuple[Dict[str, Dict], List[int]]:
    still_unassigned = []
    
    for _, order in unassigned_df.iterrows():
        assigned = False
        for machine_name, machine_info in machine_data.items():
            if (machine_info['min_cross_section'] <= order['CrossSection'] < machine_info['max_cross_section'] and
                machine_info['total_time'] + calculate_total_time(order[f"Time_{machine_info['type']}"], 0) <= get_constants()['MAX_DAILY_MINUTES']):
                
                # Check if adding this order would exceed terminal or seal limits
                new_terminals = set(machine_info['terminals']) | {order['Terminal1Key'], order['Terminal2Key']} - {-1}
                new_seals_left = set(machine_info['seals_left']) | {order['Seal1Key']} - {-1}
                new_seals_right = set(machine_info['seals_right']) | {order['Seal2Key']} - {-1}
                
                if (len(new_terminals) <= get_constants()['INITIAL_MAX_TERMINALS'] and
                    len(new_seals_left) <= get_constants()['MAX_SEALS_KOMAX'] and
                    len(new_seals_right) <= get_constants()['MAX_SEALS_KOMAX']):
                    
                    # Assign the order to this machine
                    machine_info['orders'].append(order['N'])
                    machine_info['total_time'] += calculate_total_time(order[f"Time_{machine_info['type']}"], 0)
                    machine_info['terminals'] = new_terminals
                    machine_info['seals_left'] = new_seals_left
                    machine_info['seals_right'] = new_seals_right
                    assigned = True
                    break
        
        if not assigned:
            still_unassigned.append(order['N'])
    
    return machine_data, still_unassigned

def analyze_unassigned_orders(unassigned_orders: List[int], orders_df: pd.DataFrame) -> Dict:
    unassigned_df = orders_df[orders_df['N'].isin(unassigned_orders)]
    analysis = {
        'cross_section_range': (unassigned_df['CrossSection'].min(), unassigned_df['CrossSection'].max()),
        'wire_length_range': (unassigned_df['WireLength'].min(), unassigned_df['WireLength'].max()),
        'seal_count': len(set(unassigned_df['Seal1Key']) | set(unassigned_df['Seal2Key']) - {-1}),
        'terminal_count': len(set(unassigned_df['Terminal1Key']) | set(unassigned_df['Terminal2Key']) - {-1}),
        'order_count': len(unassigned_orders)
    }
    return analysis

def select_machine_for_unassigned(unassigned_analysis: Dict, available_machines: Dict[str, Dict]) -> str:
    best_machine = None
    best_score = float('inf')

    for machine_name, machine_info in available_machines.items():
        score = 0
        if unassigned_analysis['cross_section_range'][0] < machine_info['min_cross_section'] or \
           unassigned_analysis['cross_section_range'][1] > machine_info['max_cross_section']:
            score += 1000  # High penalty for cross-section mismatch

        if machine_info['type'].startswith('AC'):
            max_seals = get_constants()['MAX_SEALS_AC91'] if machine_info['type'] == 'AC91' else get_constants()['MAX_SEALS_AC96']
            if unassigned_analysis['seal_count'] > max_seals:
                score += 500  # Penalty for exceeding seal capacity
        elif unassigned_analysis['seal_count'] > get_constants()['MAX_SEALS_KOMAX']:
            score += 500  # Penalty for exceeding Komax seal capacity

        if unassigned_analysis['terminal_count'] > get_constants()['INITIAL_MAX_TERMINALS']:
            score += 250  # Penalty for exceeding initial terminal capacity

        if score < best_score:
            best_score = score
            best_machine = machine_name

    return best_machine

def add_machine_for_unassigned(machine_data: Dict[str, Dict], unassigned_orders: List[int], orders_df: pd.DataFrame, available_machines: Dict[str, Dict]):
    if not unassigned_orders:
        return machine_data, unassigned_orders

    unassigned_analysis = analyze_unassigned_orders(unassigned_orders, orders_df)
    best_machine_name = select_machine_for_unassigned(unassigned_analysis, available_machines)

    if best_machine_name:
        new_machine = initialize_machines({best_machine_name: available_machines[best_machine_name]})[best_machine_name]
        machine_data[best_machine_name] = new_machine

        # Reassign unassigned orders to the new machine
        machine_data, unassigned_orders = resolve_unassigned_orders(machine_data, unassigned_orders, orders_df)

    return machine_data, unassigned_orders

def create_solution_csv(solution, orders_df):
    results = []
    for machine, info in solution['machine_data'].items():
        machine_orders = orders_df[orders_df['N'].isin(info['orders'])].copy()
        machine_orders['MachineAssigned'] = machine
        results.append(machine_orders)
    
    if solution['unassigned']:
        unassigned_df = orders_df[orders_df['N'].isin(solution['unassigned'])].copy()
        unassigned_df['MachineAssigned'] = 'Unassigned'
        results.append(unassigned_df)
    
    results_df = pd.concat(results, ignore_index=True)
    
    columns_to_export = ['N', 'Leadset', 'Quantity', 'Wire1Key', 'CrossSection', 'WireLength', 
                         'Terminal1Key', 'Seal1Key', 'Terminal2Key', 'Seal2Key', 'MachineAssigned']
    results_df = results_df[columns_to_export]
    
    return results_df

def calculate_cph_and_pe(machine_info, orders_df):
    machine_orders = orders_df[orders_df['N'].isin(machine_info['orders'])]
    total_quantity = machine_orders['Quantity'].sum()

    ae = machine_info['total_time'] / get_constants()['MAX_DAILY_MINUTES']
    pe = PE_SETTINGS[machine_info['type']]
    working_hour = machine_info['processing_time']/60
    cph_2 = (total_quantity / (machine_info['total_time']/60)) * ae
    return cph_2, ae, pe * 100, total_quantity
def recalculate_machine_analysis(solution, orders_df):
    machine_analysis = []
    for machine, info in solution['machine_data'].items():
        if info['total_time'] > 0:
            machine_orders = orders_df[orders_df['N'].isin(info['orders'])]
            total_quantity = machine_orders['Quantity'].sum()
            
            cph_2, ae, pe, _ = calculate_cph_and_pe(info, orders_df)
            oee = ae * (pe / 100)
            avg_unit_time = info['processing_time'] / total_quantity if total_quantity > 0 else 0
            
            machine_analysis.append({
                'Machine': machine,
                'Machine Type': info['type'],
                'Min CrossSection': machine_orders['CrossSection'].min(),
                'Max CrossSection': machine_orders['CrossSection'].max(),
                'Availability Efficiency (AE)': ae,
                'Performance Efficiency (PE)': pe / 100,
                'OEE': oee,
                'CPH': cph_2,
                'Total Time (min)': info['total_time'],
                'Processing Time (min)': info['processing_time'],
                'Setup Time (min)': info['setup_time'],
                'Break Time (min)': info['break_time'],
                'Down Time (min)': info['down_time'],
                'Orders Assigned': len(info['orders']),
                'Total Quantity': total_quantity,
                'Unique Seals Left': len(set(machine_orders['Seal1Key']) - {-1}),
                'Unique Seals Right': len(set(machine_orders['Seal2Key']) - {-1}),
                'Unique Terminals': len(set(machine_orders['Terminal1Key']) | set(machine_orders['Terminal2Key']) - {-1}),
                'Avg Unit Time': avg_unit_time,
            })
    return pd.DataFrame(machine_analysis)
def analyze_solution(solution, orders_df):
    st.subheader(f"Solution Analysis for {solution['num_machines']} machines")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Machines", solution['num_machines'])
    col2.metric("Efficiency", f"{solution['efficiency']:.2%}")
    col3.metric("Unassigned Orders", solution['unassigned_orders'])
    col4.metric("Constraint Type", solution['constraint_type'].capitalize())

    machine_df = recalculate_machine_analysis(solution, orders_df)

    tab1, tab2 = st.tabs(["Machine Analysis", "Component Analysis"])

    with tab1:
        st.subheader("Machine Utilization Details")
        st.dataframe(machine_df.style.format({
            'Availability Efficiency (AE)': '{:.2%}',
            'Performance Efficiency (PE)': '{:.2%}',
            'OEE': '{:.2%}',
            'CPH': '{:.2f}',
            'Total Time (min)': '{:.2f}',
            'Processing Time (min)': '{:.2f}',
            'Setup Time (min)': '{:.2f}',
            'Break Time (min)': '{:.2f}',
            'Down Time (min)': '{:.2f}',
            'Total Quantity': '{:,.0f}',
            'Avg Unit Time': '{:.2f}',
        }).background_gradient(subset=['Availability Efficiency (AE)', 'Performance Efficiency (PE)', 'OEE'], cmap='RdYlGn', vmin=0, vmax=1))

        # Download button for machine analysis CSV
        csv = machine_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="machine_analysis.csv">Download Machine Analysis CSV</a>'
        st.markdown(href, unsafe_allow_html=True)


        # Efficiency, OEE, CPH, and chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=machine_df['Machine'], y=machine_df['Availability Efficiency (AE)'], name='AE', yaxis='y', offsetgroup=1))
        fig.add_trace(go.Bar(x=machine_df['Machine'], y=machine_df['Performance Efficiency (PE)'], name='PE', yaxis='y', offsetgroup=2))
        fig.add_trace(go.Bar(x=machine_df['Machine'], y=machine_df['OEE'], name='OEE', yaxis='y', offsetgroup=3))
        fig.add_trace(go.Scatter(x=machine_df['Machine'], y=machine_df['CPH'], name='CPH', yaxis='y2'))

        fig.update_layout(
            title='Machine Efficiency, OEE, and Cuts Per Hour',
            yaxis=dict(title='Efficiency / OEE', tickformat='.0%', range=[0, 1]),
            yaxis2=dict(title='Cuts Per Hour', overlaying='y', side='right'),
            barmode='group'
        )
        st.plotly_chart(fig)

        # Time breakdown chart
        time_data = machine_df[['Processing Time (min)', 'Setup Time (min)', 'Break Time (min)', 'Down Time (min)']].sum()
        fig = px.pie(values=time_data.values, names=time_data.index, title='Overall Time Breakdown')
        st.plotly_chart(fig)

        # Machine type distribution
        machine_type_counts = machine_df['Machine Type'].value_counts()
        fig = px.pie(values=machine_type_counts.values, names=machine_type_counts.index, title='Machine Type Distribution')
        st.plotly_chart(fig)

        # Orders assigned per machine
        fig = px.bar(machine_df, x='Machine', y='Orders Assigned', title='Orders Assigned per Machine')
        st.plotly_chart(fig)

        # Quantity per machine
        fig = px.bar(machine_df, x='Machine', y='Total Quantity', title='Quantity per Machine')
        st.plotly_chart(fig)

        # Unassigned orders analysis
        if solution['unassigned']:
            st.subheader("Unassigned Orders Analysis")
            unassigned_df = orders_df[orders_df['N'].isin(solution['unassigned'])]
            st.write(f"Total unassigned orders: {len(solution['unassigned'])}")
            
            # Reasons for unassigned orders
            reason_counts = pd.Series(solution['unassigned_reasons']).value_counts()
            fig = px.bar(x=reason_counts.index, y=reason_counts.values, title='Reasons for Unassigned Orders')
            fig.update_layout(xaxis_title='Reason', yaxis_title='Count')
            st.plotly_chart(fig)

            # Cross-section distribution of unassigned orders
            fig = px.histogram(unassigned_df, x='CrossSection', nbins=20, title='Cross-section Distribution of Unassigned Orders')
            st.plotly_chart(fig)

    with tab2:
        st.subheader("Component Analysis")
        analyze_components(solution, orders_df)

def analyze_components(solution, data):
    cs_wl_analysis = []
    terminal_usage = {}
    seal_usage = {}
    seal_use = {}
    terminal_use = {}

    for machine, info in solution['machine_data'].items():
        machine_orders = data[data['N'].isin(info['orders'])]
        
        # CrossSection and WireLength Analysis
        cs_wl_analysis.append({
            'Machine': machine,
            'Machine Type': info['type'],
            'Min CrossSection': machine_orders['CrossSection'].min(),
            'Max CrossSection': machine_orders['CrossSection'].max(),
            'Min WireLength': machine_orders['WireLength'].min(),
            'Max WireLength': machine_orders['WireLength'].max()
        })

        # Terminal Analysis
        terminals = set(machine_orders['Terminal1Key']) | set(machine_orders['Terminal2Key']) - {-1}
        for terminal in terminals:
            if terminal not in terminal_use:
                terminal_use[terminal] = set()
            terminal_use[terminal].add(machine)

        # Seal Analysis
        seals = set(machine_orders['Seal1Key']) | set(machine_orders['Seal2Key']) - {-1}
        for seal in seals:
            if seal not in seal_use:
                seal_use[seal] = set()
            seal_use[seal].add(machine)

    terminal_dfs = pd.DataFrame([{'Terminal': terminal, 'Machines': ', '.join(machines), 'Machine Count': len(machines)} 
                                for terminal, machines in terminal_use.items()])
    st.write("### Terminal Analysis")
    st.dataframe(terminal_dfs)
    st.markdown(download_csv(terminal_dfs, "terminal_analysis.csv"), unsafe_allow_html=True)

    seal_dfs = pd.DataFrame([{'Seal': seal, 'Machine Count': len(machines), 'Machines': ', '.join(machines)}
                            for seal, machines in seal_use.items()])
    st.write("### Seal Analysis")
    st.dataframe(seal_dfs)
    st.markdown(download_csv(seal_dfs, "seal_analysis.csv"), unsafe_allow_html=True)

    cs_wl_df = pd.DataFrame(cs_wl_analysis)
    st.write("### CrossSection and WireLength Analysis")
    st.dataframe(cs_wl_df.style.format({
        'Min CrossSection': '{:.2f}',
        'Max CrossSection': '{:.2f}',
        'Min WireLength': '{:.2f}',
        'Max WireLength': '{:.2f}',
    }))
    st.markdown(download_csv(cs_wl_df, "crossection_wirelength_analysis.csv"), unsafe_allow_html=True)

    # Additional visualizations
    st.subheader("Component Distribution Visualizations")

    # Terminal distribution across machines
    terminal_counts = terminal_dfs['Machine Count'].value_counts().sort_index()
    fig = px.bar(x=terminal_counts.index, y=terminal_counts.values, 
                 labels={'x': 'Number of Machines', 'y': 'Number of Terminals'},
                 title='Terminal Distribution Across Machines')
    st.plotly_chart(fig)

    # Seal distribution across machines
    seal_counts = seal_dfs['Machine Count'].value_counts().sort_index()
    fig = px.bar(x=seal_counts.index, y=seal_counts.values, 
                 labels={'x': 'Number of Machines', 'y': 'Number of Seals'},
                 title='Seal Distribution Across Machines')
    st.plotly_chart(fig)

    # CrossSection range per machine
    fig = px.bar(cs_wl_df, x='Machine', y=['Min CrossSection', 'Max CrossSection'],
                 title='CrossSection Range per Machine', barmode='group')
    st.plotly_chart(fig)

    # WireLength range per machine
    fig = px.bar(cs_wl_df, x='Machine', y=['Min WireLength', 'Max WireLength'],
                 title='WireLength Range per Machine', barmode='group')
    st.plotly_chart(fig)

def download_csv(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Main application logic
def main():
    st.title("Order Distribution Process Optimizer")
    sidebar_logo = Image.open("yazaki_logo.png")
    st.sidebar.image(sidebar_logo, use_column_width=True)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select a page", ["Configuration", "Optimization"])

    if page == "Configuration":
        show_config_page()
    else:
        optimization_page()

    # Add information about the tool
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "Welcome to the Order Distribution Process Optimizer! "
        "This application streamlines the process of distributing orders "
        "across multiple machines in a wire harness production environment. "
        "It uses a sophisticated algorithm to ensure efficient utilization of resources "
        "and minimize setup times, resulting in optimized production schedules."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown(
        "1. Start with the Configuration page to set up parameters.\n"
        "2. Move to the Optimization page to upload files and run the optimizer.\n"
        "3. Analyze results in the provided visualizations and download reports."
    )
    st.sidebar.markdown("---")
    st.sidebar.info("Developed by: Haytham Zelmatte (intern)")

def optimization_page():
    st.header("Optimization")

    machines_file = st.file_uploader("Upload Machines CSV", type="csv")
    if machines_file is not None:
        global MACHINES, PE_SETTINGS
        MACHINES = load_machines_from_csv(machines_file)
        st.success(f"Loaded {len(MACHINES)} machines")
        
        # Display machine types and their ranges
        ranges = {
            'K433': [(0.35, 3), (3, 7)],
            'K433H': [(3, 10)],
            'K560': [(3, 10)],
            'AC91': [(0.13, 0.35), (0.35, 2.5), (2.5, 6)],
            'AC96': [(0.13, 0.35), (0.35, 2.5), (2.5, 6)],
            'K355': [(0.13, 0.35), (0.35, 3), (3, 6)]
        }
        machine_types = set(info['type'] for info in MACHINES.values())
        for machine_type in machine_types:
            st.write(f"{machine_type} ranges: {ranges[machine_type]}")
        
        st.subheader("Performance Efficiency (PE) Settings")
        PE_SETTINGS = {machine_type: st.slider(f"PE% for {machine_type}", 0.0, 1.0, 0.8, 0.01) 
                       for machine_type in set(info['type'] for info in MACHINES.values())}

        min_machines = st.number_input("Minimum Number of Machines", 
                                       min_value=1, 
                                       max_value=len(MACHINES), 
                                       value=min(17, len(MACHINES)))

        st.header("Upload Orders Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Raw Data Preview:")
            st.write(data.head())

            # Assign ranges to machines based on order demand
            MACHINES = assign_ranges_to_machines(MACHINES, data)

            # Preprocess the data
            with st.spinner('Preprocessing data...'):
                preprocessed_data = preprocess_orders(data)
            
            st.write("Preprocessed Data Preview:")
            st.write(preprocessed_data.head())

            st.session_state.preprocessed_data = preprocessed_data

            if st.button("Run Optimization"):
                run_optimization(preprocessed_data, min_machines)

    else:
        st.warning("Please upload a machines CSV file to start.")

def run_optimization(data, min_machines):
    seal_terminal_cols = ['Seal1Key', 'Seal2Key', 'Terminal1Key', 'Terminal2Key']
    data[seal_terminal_cols] = data[seal_terminal_cols].fillna(-1)
    data['has_seals_terminals'] = data[seal_terminal_cols].ne(-1).any(axis=1)
    
    data['group'] = data.apply(lambda row: tuple(list(row[seal_terminal_cols]) + [row['CrossSectionType'], row['WireLengthType']]), axis=1)
    grouped_orders = data.groupby('group')

    def sort_key(group):
        has_seals = group[1]['Seal1Key'].ne(-1).any() or group[1]['Seal2Key'].ne(-1).any()
        has_terminals = group[1]['Terminal1Key'].ne(-1).any() or group[1]['Terminal2Key'].ne(-1).any()
        machine_type_count = sum(group[1][f'Time_{machine_type}'].notna().any() for machine_type in set(info['type'] for info in MACHINES.values()))
        group_size = len(group[1])
        return (-int(has_seals), -int(has_terminals), machine_type_count, -group_size)

    sorted_groups = sorted(grouped_orders, key=sort_key)
    
    with st.spinner('Running optimization...'):
        all_solutions = optimize_with_load_balancing(sorted_groups, data, min_machines)
    
    if all_solutions:
        st.success("Optimization completed successfully!")
        st.header("Optimization Results")
        
        best_solution = min(all_solutions, key=lambda x: (x['unassigned_orders'], -x['efficiency']))
        
        tab_names = ["All Solutions"] + [f"Solution {i+1}" for i in range(len(all_solutions))]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            st.subheader("All Solutions Summary")
            summary_data = [{
                "Solution": idx + 1,
                "Machines": solution['num_machines'],
                "Efficiency": solution['efficiency'],
                "Unassigned Orders": solution['unassigned_orders'],
                "Constraint Type": solution['constraint_type']
            } for idx, solution in enumerate(all_solutions)]
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.format({"Efficiency": "{:.2%}"})
                         .background_gradient(subset=['Efficiency'], cmap='RdYlGn', vmin=0, vmax=1)
                         .background_gradient(subset=['Unassigned Orders'], cmap='RdYlGn_r'))
            
            for idx, solution in enumerate(all_solutions):
                solution_df = create_solution_csv(solution, data)
                csv = solution_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="solution_{solution["num_machines"]}_machines.csv">Download Solution {idx+1} CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        # Detailed analysis for each solution
        for idx, solution in enumerate(all_solutions):
            with tabs[idx + 1]:
                analyze_solution(solution, data)
    else:
        st.error("No valid solution found. Please check your data and constraints.")

if __name__ == "__main__":
    main()