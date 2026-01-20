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
SAFETY_MARGIN = 300 #marge sécurité de 5 minutes

#########################################
#           CHARGEMENT DONNÉES
#########################################

# Chargement véhicules
data_vehicles = pd.read_csv("vehicles.csv")
VEHICLES_DATA = data_vehicles.to_dict(orient="records") #gain de temps complexité
VEHICLES_DICT = {v['family']: v for v in VEHICLES_DATA}

# Chargement instances
instances = []
for k in range(1, 11):
    file_path = f"instance_{k:02d}.csv"
    df = pd.read_csv(file_path)
    instances.append(df.to_dict(orient="records"))



#########################################
#           FONCTIONS AUXILIAIRES
#########################################

### FONCTIONS GEOGRAPHIQUES (coordonnées, distances)
# Coordonnées
def matrice_distance(df_inst):
    #calcul des distances (en une seule fois, pour la complexité)
    n = len(df_inst)
    
    #vecteurs x et y
    x_coords = np.zeros(n)
    y_coords = np.zeros(n)
    
    cos_phi0 = math.cos(2 * np.pi * phi_0 / 360)
    
    for i in range(n):
        lat = df_inst.iloc[i]['latitude']
        lon = df_inst.iloc[i]['longitude']
        x_coords[i] = rho * cos_phi0 * 2 * np.pi * lon / 360
        y_coords[i] = rho * 2 * np.pi * lat / 360

    matrix_M = np.zeros((n, n))
    matrix_E = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dx = x_coords[j] - x_coords[i]
            dy = y_coords[j] - y_coords[i]
            
            #distance de Manhattan
            matrix_M[i, j] = abs(dx) + abs(dy)
            
            #distance euclidienne
            matrix_E[i, j] = math.sqrt(dx**2 + dy**2) 
    return matrix_M, matrix_E, x_coords, y_coords


def distM(i, j, instance_idx): 
    return distance_mat_M[i, j]

def distE(i, j, instance_idx):
    return distance_mat_E[i, j]

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

### FONCTIONS TEMPORELLES
def get_gamma_f_t(row_f, t_sec):
    #gamma qui dépend de la ligne de famille et du temps (en seconde)
    omega = 2*np.pi/(24*3600)  #pulsation -> cycle de 24h
    gamma = 0 #initialisation
    for n in range(4):
        alpha_n = row_f[f"fourier_cos_{n}"]
        beta_n = row_f[f"fourier_sin_{n}"]
        gamma += alpha_n*math.cos(n*omega*t_sec) + beta_n*math.sin(n*omega*t_sec)
    return gamma


def is_time_possible(row_f, sequence, instance_idx):
    inst = instances[instance_idx]
    current_time = 0.0 #départ du dépôt à t=0
    prev_order = 0

    for j in sequence[1:]: #ne pas calculer le trajet du dépôt au dépôt
        d = distM(prev_order, j, instance_idx) #distance
        
        #Calcul de gamma: temps de départ = current_time
        gamma_f_t = get_gamma_f_t(row_f, current_time)
        
        #temps de trajet = (d/v + temps parking)*Gamma(f,t)
        travel_time = (d/row_f["speed"] + row_f["parking_time"])*gamma_f_t
        
        #Arrivée au client j
        arrival_time = current_time + travel_time
        
        #vérification fenêtre de temps
        t_min = inst[j]["window_start"]
        t_max = inst[j]["window_end"]
        if arrival_time > t_max - SAFETY_MARGIN:
            return False
            
        #calcul du temps de départ de j
        start_service = max(arrival_time, t_min)
        current_time = start_service + inst[j]["delivery_duration"] 
        prev_order = j
    return True


### OPTIMISATION
def get_best_vehicle(sequence, instance_idx):
    #calcul des caractéristiques de la route "sequence"
    total_weight = sum(weights_dict[node] for node in sequence if node != 0)
    total_dist, max_radius = get_route_dist_rad(sequence, instance_idx)
    
    best_family = None
    min_total_cost = float('inf')
    
    for v in VEHICLES_DATA:
        #vérification capacité (évite appel is_time_possible lourd)
        if v['max_capacity'] >= total_weight:
            #vérification temps 
            if is_time_possible(v, sequence, instance_idx):
                #minimiser le coût
                current_cost = v['rental_cost'] + v['fuel_cost']*total_dist + v['radius_cost']*max_radius    
                if current_cost < min_total_cost:
                    min_total_cost = current_cost
                    best_family = v['family']
    return best_family, min_total_cost



def optimize_route_permut(sequence, instance_idx, family_id):
    row_f = VEHICLES_DICT[family_id]
    best_seq = list(sequence)
    #on ne touche pas au premier et dernier (dépôts : 0)
    gain = True
    while gain:
        gain = False
        for i in range(1, len(best_seq) - 2):
            for j in range(i + 1, len(best_seq) - 1):
                #on inverse le segment entre i et j
                new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                
                #vérification: faisable avec la famille actuelle?
                if is_time_possible(row_f, new_seq, instance_idx):
                    #compare la distance
                    d_old, _ = get_route_dist_rad(best_seq, instance_idx)
                    d_new, _ = get_route_dist_rad(new_seq, instance_idx)
                    
                    if d_new < d_old:
                        best_seq = new_seq
                        gain = True
        if not gain: break
    return best_seq


def get_route_centers(routes):
    #liste des coordonnées des centres des routes
    centers = []
    for r in routes:
        seq = r["sequence"]
        x_moy = np.mean([x_coords[order] for order in seq if order != 0])
        y_moy = np.mean([y_coords[order] for order in seq if order != 0])
        centers.append((x_moy, y_moy))
    return centers

def get_closest_routes(target_route_id, centers, k=5):
    #renvoie les k plus proches routes de la route target_route_id
    distances = []
    for i, center in enumerate(centers):
        if i == target_route_id: continue
        #Distance entre le centre de la route "target" à laquelle on s'intéresse et la route 'i'
        d = math.sqrt((centers[target_route_id][0] - center[0])**2 + (centers[target_route_id][1] - center[1])**2)
        distances.append((i, d))

    #on trie par distance et on garde les K plus proches
    distances.sort(key=lambda x: x[1])
    return [x[0] for x in distances[:k]]


def eliminer_petites_routes(routes, instance_idx):
    indices_tries = sorted(range(len(routes)), key=lambda i: len(routes[i]['sequence']))
    routes_a_supprimer = []
    
    for i in indices_tries:
        route_source = routes[i]
        clients_a_deplacer = route_source["sequence"][1:-1]
        
        #que les petites routes
        if len(clients_a_deplacer) > 5 or len(clients_a_deplacer) == 0:
            continue

        #copies locales pour pouvoir annuler si l'élimination échoue
        temp_sequences = {j: list(routes[j]["sequence"]) for j in range(len(routes))}
        temp_families = {j: routes[j]["family"] for j in range(len(routes))}
        reussite_pour_tous = True
        
        for client in clients_a_deplacer:
            place_trouvee = False
            #on cherche une cible parmi les routes non supprimées et différente de la source
            for j in range(len(routes)):
                if i == j or j in routes_a_supprimer: continue
                
                #on teste sur la séquence déjà modif de la cible
                seq_cible = temp_sequences[j]
                for pos in range(1, len(seq_cible)):
                    nouvelle_seq = seq_cible[:pos] + [client] + seq_cible[pos:]
                    best_f, _ = get_best_vehicle(nouvelle_seq, instance_idx)
                    
                    if best_f is not None:
                        #mise à jour de la copie locale pour le prochain client
                        temp_sequences[j] = nouvelle_seq
                        temp_families[j] = best_f
                        place_trouvee = True
                        break
                if place_trouvee: break
            
            if not place_trouvee:
                reussite_pour_tous = False
                break
        
        if reussite_pour_tous:
            #validation des changements dans la liste principale
            for j, seq in temp_sequences.items():
                routes[j]["sequence"] = seq
                routes[j]["family"] = temp_families[j]
            routes_a_supprimer.append(i)
            #print(f"Route {i} éliminée !")
    return [r for idx, r in enumerate(routes) if idx not in routes_a_supprimer]



#########################################
#          BOUCLE SUR LES INSTANCES
#########################################
for A in range(10):
    print(f"Instance {A+1:02d}...")
    df_inst = pd.read_csv(f"instance_{A+1:02d}.csv")
    distance_mat_M, distance_mat_E, x_coords, y_coords = matrice_distance(df_inst)
    weights_dict = df_inst.set_index('id')['order_weight'].to_dict() #gain de temps
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
        _, cost = get_best_vehicle([0, o_id, 0], A)
        costs_indiv[o_id] = cost

    #calcul des savings
    savings = []
    for i in range(len(orders_id)):
        for j in range(i+1, len(orders_id)):
            id_i = orders_id[i]
            id_j = orders_id[j]
            
            #si fusion
            merged_seq = [0, id_i, id_j, 0]
            best_f, cost_merged = get_best_vehicle(merged_seq, A)
            
            #gain
            if best_f is not None:
                gain = (costs_indiv[id_i] + costs_indiv[id_j]) - cost_merged
                savings.append((gain, id_i, id_j))
    
    #Trier par gain décroissant
    savings.sort(key=lambda x: x[0], reverse=True)

    #on essaie de fusionner les routes
    for s, i, j in savings:
        route_i = next((rid for rid, rseq in routes_simples.items() if rseq[-2] == i), None)
        route_j = next((rid for rid, rseq in routes_simples.items() if rseq[1] == j), None)
        if route_i is not None and route_j is not None and route_i != route_j:
            new_seq = routes_simples[route_i][:-1] + routes_simples[route_j][1:]
            best_f, _ = get_best_vehicle(new_seq, A)
            if best_f:
                routes_simples[route_i] = new_seq
                del routes_simples[route_j]
    final_routes = []
    for r_id, sequence in routes_simples.items():
        family, cost = get_best_vehicle(sequence, A)

        if family:
            #optimized_seq = optimize_route_permut(sequence, A, family)
            #final_family, final_cost = get_best_vehicle(optimized_seq, A)
            #final_routes.append({"family":final_family, "sequence":optimized_seq})
            final_routes.append({"family":family, "sequence": sequence})
    
    #if final_routes:
        #final_routes = eliminer_petites_routes(final_routes, A)



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