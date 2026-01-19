#Bloc d'importation
from tabnanny import verbose
import pandas as pd
import numpy as np
import math
import pulp
import os

##### Constantes #####
rho = 6.371E6
phi_0 = 48.764246


#########################################
#           CHARGEMENT DONNÉES
#########################################

# Chargement véhicules
data_vehicles = pd.read_csv("vehicles.csv")

# Chargement instances
instances = []
for k in range(1, 11):
    file_path = f"instance_{k:02d}.csv"
    df = pd.read_csv(file_path)
    instances.append(df.to_dict(orient="records"))



#########################################
#           FONCTIONS AUXILIAIRES
#########################################

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


#Fonction distance euclidienne
def distE(i, j, A):
    deltax = xj_xi(instances[A][j]["longitude"], instances[A][i]["longitude"])
    deltay = yj_yi(instances[A][j]["latitude"], instances[A][i]["latitude"])
    return math.sqrt(deltax**2 + deltay**2)


def get_route_dist_rad(sequence, instance_idx):
    #renvoie la distance totale parcourue et le rayon d'une route
    total_dist = 0
    max_radius = 0
    
    #on parcourt la séquence par paires (i, j)
    for k in range(len(sequence)-1):
        i_idx = sequence[k]
        j_idx = sequence[k+1]
        
        #distance de Manhattan entre deux points consécutifs
        d = distM(i_idx, j_idx, instance_idx)
        total_dist += d
        
    #Calcul du rayon (Moitié du diamètre Euclidien entre les commandes)
    orders_in_route = sequence[1:-1]
    max_euclidean_dist = 0
    if len(orders_in_route) > 1:
        for idx_a in range(len(orders_in_route)):
            for idx_b in range(idx_a + 1, len(orders_in_route)):
                d_e = distE(orders_in_route[idx_a], orders_in_route[idx_b], instance_idx)
                if d_e > max_euclidean_dist:
                    max_euclidean_dist = d_e
    max_radius = 0.5 * max_euclidean_dist
    return total_dist, max_radius


def get_best_vehicle(total_weight, total_dist, max_radius):
    best_family = None
    min_total_cost = float('inf')
    
    for index, v in data_vehicles.iterrows(): #on parcourt tous les véhicules
        if v['max_capacity'] >= total_weight:
            #calcul du coût
            current_cost = v['rental_cost'] + v['fuel_cost']*total_dist + v['radius_cost']*max_radius    
            #on minimise le coût
            if current_cost < min_total_cost:
                min_total_cost = current_cost
                best_family = v['family']   
    return best_family

def get_best_vehicle(sequence, instance_idx, df_inst):
    #calcul des caractéristiques de la route "sequence"
    total_weight = sum(df_inst.loc[df_inst['id'] == node, 'order_weight'].values[0] 
                      for node in sequence if node != 0)
    total_dist, max_radius = get_route_dist_rad(sequence, instance_idx)
    
    best_family = None
    min_total_cost = float('inf')
    
    for index, v in data_vehicles.iterrows():
        family = v['family']
        #vérification capacité (évite appel is_route_possible lourd)
        if v['max_capacity'] >= total_weight:
            #vérification temps 
            if is_route_possible(family, sequence[1:-1], instance_idx):
                #minimiser le coût
                current_cost = v['rental_cost'] + v['fuel_cost']*total_dist + v['radius_cost']*max_radius    
                if current_cost < min_total_cost:
                    min_total_cost = current_cost
                    best_family = family
    return best_family, min_total_cost


#coût d'une route
def route_cost(sequence, instance_idx, df_inst):
    #Poids de la route
    total_w = sum(df_inst.loc[df_inst['id'] == order, 'order_weight'].values[0] for order in sequence if order != 0)
    
    #Distance et rayon de la route
    d_tot, r_max = get_route_dist_rad(sequence, instance_idx)
    
    #Trouver le véhicule qui minimise rental + fuel + radius
    best_family = None
    min_cost = float('inf')
    
    for _, v in data_vehicles.iterrows():
        if v['max_capacity'] >= total_w:
            cost = v['rental_cost'] + (v['fuel_cost']*d_tot) + (v['radius_cost']*r_max)
            if cost < min_cost:
                min_cost = cost
                best_family = v['family']         
    return min_cost, best_family

#########################################
#          BOUCLE SUR LES INSTANCES
#########################################
for A in range(10):
    print(f"Instance {A+1:02d}...")
    df_inst = pd.read_csv(f"instance_{A+1:02d}.csv")
    orders = df_inst[df_inst['order_weight'].notna()].copy() #commandes (on exclut le dépôt)
    orders_id = orders['id'].astype(int).tolist()
    
    #Heuristique Clarke and Wright

    #initialisation: 1 client = 1 route
    routes_simples = {o_id: [0, o_id, 0] for o_id in orders_id}

    #calcul des savings
    #initialisation des routes et calcul de leur coût individuel
    routes_simples = {o_id: [0, o_id, 0] for o_id in orders_id}
    costs_indiv = {}
    for o_id in orders_id:
        _, cost = get_best_vehicle([0, o_id, 0], A, df_inst)
        costs_indiv[o_id] = cost

    #calcul des savings
    savings = []
    for i in range(len(orders_id)):
        for j in range(i+1, len(orders_id)):
            id_i = orders_id[i]
            id_j = orders_id[j]
            
            #si fusion
            merged_seq = [0, id_i, id_j, 0]
            best_f, cost_merged = get_best_vehicle(merged_seq, A, df_inst)
            
            #gain
            if best_f is not None:
                gain = (costs_indiv[id_i] + costs_indiv[id_j]) - cost_merged
                savings.append((gain, id_i, id_j))
    
    #Trier par gain décroissant
    savings.sort(key=lambda x: x[0], reverse=True)

    #on essaie de fusionner les routes
    for s, i, j in savings:
        #trouver les routes contenant i et j
        route_i = None
        route_j = None
        for r_id, r_seq in routes_simples.items():
            if r_seq[-2] == i: #dans sa route, i est le dernier client visité avant le dépôt
                route_i = r_id
            if r_seq[1] == j: #dans sa route, j est le premier client visité après le dépôt
                route_j = r_id
        
        #conditions pour fusionner : i et j dans des routes différentes
        if route_i is not None and route_j is not None and route_i != route_j:
            new_sequence = routes_simples[route_i][:-1] + routes_simples[route_j][1:]
            
            #vérifier si un véhicule peut faire cette route (poids + temps)
            can_merge = False
            best_f = None
            
            #on récupère le poids total de la route fusionnée
            total_w = sum(df_inst.loc[df_inst['id'] == client_id, 'order_weight'].values[0] for client_id in new_sequence if client_id != 0)
            
            for i, v in data_vehicles.iterrows():
                if total_w <= v['max_capacity']: 
                    if is_route_possible(v['family'], new_sequence[1:-1], A):
                        can_merge = True
                        break
            
            if can_merge:
                #si on peut, on fusionne et on supprime donc la route j
                routes_simples[route_i] = new_sequence
                del routes_simples[route_j]

    final_routes = []
    for r_id, sequence in routes_simples.items():
        total_w = sum(df_inst.loc[df_inst['id'] == client_id, 'order_weight'].values[0] for client_id in sequence if client_id != 0)
        d_tot, r_max = get_route_dist_rad(sequence, A)
        family = get_best_vehicle(total_w, d_tot, r_max)

        if family:
            final_routes.append({"family":family, "sequence":sequence})
    

    #########################################
    #      EXPORT CSV (Format Flexible)
    #########################################
    
    if not final_routes: continue

    #Calcul de N (nombre max de sommets dans une route)
    max_nodes = max(len(r["sequence"][1:-1]) for r in final_routes)
    
    #Colonnes : family, v_0, v_1, v_2... SANS LE DEPOT
    sol_columns = ["family"] + [f"order_{i+1}" for i in range(max_nodes)]
    
    sol_rows = []
    for r in final_routes:
        orders_without_0 = r["sequence"][1:-1]
        #On fusionne la famille et la séquence dans une seule liste
        new_row = [r["family"]] + orders_without_0
        #Remplissage par du vide pour atteindre la largeur maximale du CSV
        while len(new_row) < len(sol_columns):
            new_row.append("")
        sol_rows.append(new_row)

    df_sol = pd.DataFrame(sol_rows, columns=sol_columns)
    df_sol.to_csv(f"solution_{A+1:02d}.csv", index=False)
