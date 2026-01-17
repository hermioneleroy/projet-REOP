#Bloc d'importation
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
    
    for index, v in vehicles.iterrows(): #on parcourt tous les véhicules
        if v['max_capacity'] >= total_weight:
            #calcul du coût
            current_cost = v['rental_cost'] + v['fuel_cost']*total_dist + v['radius_cost']*max_radius    
            #on minimise le coût
            if current_cost < min_total_cost:
                min_total_cost = current_cost
                best_family = v['family']   
    return best_family

# Chargement instances
instances = []
for k in range(1, 11):
    file_path = f"instance_{k:02d}.csv"
    df = pd.read_csv(file_path)
    instances.append(df.to_dict(orient="records"))


#########################################
#          FONCTIONS GÉOMÉTRIQUES
#########################################
# Coordonnees
def yj_yi(phij, phii): #yj - yi
    return rho * 2 * np.pi * (phij - phii) / 360

def xj_xi(lambdaj, lambdai): #xj - xi
    return rho * math.cos(2 * np.pi * phi_0 / 360) * 2 * np.pi * (lambdaj - lambdai) / 360

def distM(i, j, A): #distance de manhattan entre i et j du fichier A
    deltax = xj_xi(instances[A][j]["longitude"], instances[A][i]["longitude"])
    deltay = yj_yi(instances[A][j]["latitude"], instances[A][i]["latitude"])
    return abs(deltax) + abs(deltay)


#########################################
#          BOUCLE SUR LES INSTANCES
#########################################
for A in range(10):
    print(f"Instance {A+1:02d}...")
    df_inst = pd.read_csv(f"instance_{A+1:02d}.csv")
    orders = df_inst[df_inst['order_weight'].notna()].copy() #commandes (on exclut le dépôt)
    
    #liste de dictionnaires: {famille de véhicules 'family':X, suite de sommets 'sequence':[0, ..., 0]}
    final_routes = []

    #Heuristique
    #1 commande = 1 route
    for index, row in orders.iterrows():
        order_id = int(row['id'])
        order_weight = row['order_weight']
        
        #Étape 1 : Définir la suite de sommets visités
        sequence = [0, order_id, 0]
        
        #Étape 2 : Calculer la distance et le rayon
        d_tot, r_max = get_route_dist_rad(sequence, A)
        
        #Étape 3 : Trouver le meilleur véhicule pour cette route
        family = get_best_vehicle(order_weight, d_tot, r_max)
        
        if family:
            final_routes.append({"family": family, "sequence": sequence})

    #########################################
    #      EXPORT CSV (Format Flexible)
    #########################################
    
    if not final_routes: continue

    # Calcul de N (le nombre max de sommets dans une route)
    max_nodes = max(len(r["sequence"]) for r in final_routes)
    
    # Colonnes : family, v_0, v_1, v_2...
    sol_columns = ["family"] + [f"v_{i}" for i in range(max_nodes)]
    
    sol_rows = []
    for r in final_routes:
        orders_without_0 = r["sequence"][1:-1]
        # On fusionne la famille et la séquence dans une seule liste
        new_row = [r["family"]] orders_without_0
        # Remplissage par du vide pour atteindre la largeur maximale du CSV
        while len(new_row) < len(sol_columns):
            new_row.append("")
        sol_rows.append(new_row)

    df_sol = pd.DataFrame(sol_rows, columns=sol_columns)
    df_sol.to_csv(f"solution_{A+1:02d}.csv", index=False)
