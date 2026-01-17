#Bloc d'importation
from tabnanny import verbose
import pandas as pd
import numpy as np
import math
import os

##### Constantes #####
rho = 6.371E6
phi_0 = 48.764246

# Chargement véhicules
data_vehicles = pd.read_csv("vehicles.csv")

# Chargement instances
instances = []
for k in range(1, 11):
    file_path = f"instance_{k:02d}.csv"
    df = pd.read_csv(file_path)
    instances.append(df.to_dict(orient="records"))

# Coordonnees
def yj_yi(phij, phii): #yj - yi
    return rho * 2 * np.pi * (phij - phii) / 360

def xj_xi(lambdaj, lambdai): #xj - xi
    return rho * math.cos(2 * np.pi * phi_0 / 360) * 2 * np.pi * (lambdaj - lambdai) / 360

def distM(i, j, instance_idx): #distance de manhattan entre i et j du fichier A
    deltax = xj_xi(instances[instance_idx][j]["longitude"], instances[instance_idx][i]["longitude"])
    deltay = yj_yi(instances[instance_idx][j]["latitude"], instances[instance_idx][i]["latitude"])
    return abs(deltax) + abs(deltay)

def gamma_f_max(row_f):
    # majorant simple
    return sum(
        np.sqrt(row_f[f"fourier_cos_{n}"]**2 + row_f[f"fourier_sin_{n}"]**2)
        for n in range(4)
    )

def temps_max(i, j, family, instance_idx):    
    """
    Returns the maximum travel time for a given family and instance index from i to j.
    tau_f(i,j|max) = (distM(i,j)/s_f + p_f) * gamma_f_max
    Args:
        family (str): The family of vehicles
        i (int): The starting node index.
        j (int): The ending node index.
        instance_idx (int): The index of the instance file.
    """
    row_f = data_vehicles.loc[data_vehicles["family"] == family].iloc[0]
    speed_f = row_f["speed"]
    p_f = row_f["parking_time"]
    gamma_f = gamma_f_max(row_f)
    distance = distM(i, j, instance_idx)
    return (distance / speed_f + p_f) * gamma_f




def is_route_possible(family, sequence, instance_idx):
    """
    Determines if a given sequence of nodes can form a valid route
    for the specified instance index and family vehicle.
    
    Args:
        family (str): The family of vehicles
        sequence (list): A list of node indices representing the route.
        instance_idx (int): The index of the instance file.
    """
    if not isinstance(sequence, (list, tuple)):
        print("sequence must be a list or tuple.")
        return False
    if not (0 <= instance_idx < len(instances)):
        print("instance_idx out of range.")
        return False

    # Pas de dépôt dans la séquence (implicite au début)
    if 0 in sequence:
        print("depot in sequence")
        return False
    
    #Noeuds valides
    inst_size = len(instances[instance_idx])
    for node in sequence:
        if not (0 <= node < inst_size):
            print("invalid node in sequence")
            return False
    
    # Pas de doublons dans la séquence
    if len(sequence) != len(set(sequence)):
        print("duplicates in sequence")
        return False
    
    inst = instances[instance_idx]
    row_f = data_vehicles.loc[data_vehicles["family"] == family].iloc[0]
    capacity = row_f["max_capacity"]  

    # Capacitée du véhicule respéctée
    total_weight = 0.0
    for node in sequence:
        w = inst[node].get("order_weight", None)
        if w is None or (isinstance(w, float) and np.isnan(w)):
            return False
        total_weight += w

    if total_weight > capacity:
        print("capacity exceeded")
        return False
    
    # Contraine de temps ( fenêtres de temps )
    
    current_time = 0.0    # temps actuel = départ dépôt
    prev = 0   # dépôt implicite

    for j in sequence: 
        i = prev
        # temps de trajet majoré
        travel = temps_max(i, j, family, instance_idx)
        # arrivée brute à j
        arrival = current_time + travel

        tmin_j = inst[j]["window_start"]
        tmax_j = inst[j]["window_end"]
        lj = inst[j]["delivery_duration"]

        # attente autorisée
        start_service = max(arrival, tmin_j)

        # faisabilité fenêtre
        if start_service > tmax_j:
            print("time window violated")
            return False

        # départ après service
        current_time = start_service + lj
        prev = j

    return True
