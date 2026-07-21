# Copyright 2026 [Juhen Fashikha Wildan / Universitas Airlangga]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference: https://www.kaggle.com/code/jamesmcguigan/ant-colony-optimization-algorithm

# ==========================================
# IMPORTS & TYPE HINTING
# ==========================================
import os
import ast
import time
import math
import random
import argparse
from itertools import chain
from typing import Any, Callable, List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image

# ==========================================
# GLOBAL PATH SETTINGS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# VISUALIZATION & HELPER FUNCTIONS
# ==========================================
def load_coordinates(city_name: str) -> Dict[str, Tuple[float, float]]:
    """Membaca koordinat dari file txt di folder data/coordinates/"""
    file_path = os.path.join(BASE_DIR, '..', 'data', 'coordinates', f'District_Office_{city_name}.txt')
    with open(file_path, 'r') as file:
        content = file.read()
        dict_string = content.replace('Kecamatan = ', '').strip()
        return ast.literal_eval(dict_string)

def load_city_map(city_name: str):
    """Smart Detector untuk Path Gambar Peta"""
    map_base_path = os.path.join(BASE_DIR, '..', 'assets', 'maps', f'Location_District_Office_{city_name}')
    
    if os.path.exists(f"{map_base_path}.png"):
        return mpimg.imread(f"{map_base_path}.png")
    elif os.path.exists(f"{map_base_path}.jpg"):
        return mpimg.imread(f"{map_base_path}.jpg")
    else:
        raise FileNotFoundError(f"ERROR: Peta untuk {city_name} tidak ditemukan di folder assets/maps/")

def show_Kecamatan(path: Union[Dict, List], map_img: Any, w: int = 8, h: int = 8) -> None:
    """Plot a TSP path overlaid on a map of the Kantor Kecamatan."""
    if isinstance(path, dict):      
        path = list(path.values())
    if isinstance(path[0][0], str): 
        path = [ item[1] for item in path ]    
    
    plt.imshow(map_img)    
    for x0, y0 in path:
        plt.plot(x0, y0, 'y*', markersize=5)      
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])

    save_dir = os.path.join(BASE_DIR, '..', 'assets', 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'Kecamatan.png')
    
    plt.savefig(save_path)
    # plt.show() # Dinonaktifkan agar tidak menghentikan CLI
    plt.close(fig) # Mencegah memory leak

def show_path(path: Union[Dict, List], map_img: Any, starting_kecamatan: Optional[Tuple[float, float]] = None, w: int = 8, h: int = 8) -> None:
    """Plot a TSP path overlaid on a map."""
    if isinstance(path, dict):      
        path = list(path.values())
    if isinstance(path[0][0], str): 
        path = [ item[1] for item in path ]
    
    starting_kecamatan = starting_kecamatan or path[0]
    x, y = list(zip(*path))
    (x0, y0) = starting_kecamatan
    
    plt.imshow(map_img)
    plt.plot(x0, y0, 'y*', markersize=5) 
    plt.plot(x + x[:1], y + y[:1]) 
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])

    save_dir = os.path.join(BASE_DIR, '..', 'assets', 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'path.png')
    
    plt.savefig(save_path)
    # plt.show() # Dinonaktifkan agar tidak menghentikan CLI
    plt.close(fig) # Mencegah memory leak

def distance(xy1: Any, xy2: Any) -> float:
    if isinstance(xy1[0], str): xy1 = xy1[1]; xy2 = xy2[1]
    return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

def path_distance(path: Union[Dict, List]) -> int:
    if isinstance(path, dict):      path = list(path.values())
    if isinstance(path[0][0], str): path = [ item[1] for item in path ]
    return int(sum(
        [ distance(path[i],  path[i + 1]) for i in range(len(path) - 1) ] +
        [ distance(path[-1], path[0]) ]  
    ))

# ==========================================
# ALGORITHM: ANT COLONY SOLVER
# ==========================================
class AntColonySolver:
    def __init__(self,
                 cost_fn: Callable[[Any,Any], Union[float,int]],                         
                 time=0, min_time=0, timeout=0, stop_factor=2, min_round_trips=10, max_round_trips=0,                
                 min_ants=0, max_ants=0, ant_count=64, ant_speed=1, distance_power=1,        
                 pheromone_power=1.25, decay_power=0, reward_power=0, best_path_smell=2,                  
                 start_smell=0, verbose=False):
        
        assert callable(cost_fn)        
        self.cost_fn         = cost_fn
        self.time            = int(time)
        self.min_time        = int(min_time)
        self.timeout         = int(timeout)
        self.stop_factor     = float(stop_factor)
        self.min_round_trips = int(min_round_trips)
        self.max_round_trips = int(max_round_trips)
        self.min_ants        = int(min_ants)
        self.max_ants        = int(max_ants)
        self.ant_count       = int(ant_count)
        self.ant_speed       = int(ant_speed)
        self.distance_power  = float(distance_power)     
        self.pheromone_power = float(pheromone_power)
        self.decay_power     = float(decay_power)
        self.reward_power    = float(reward_power)
        self.best_path_smell = float(best_path_smell)
        self.start_smell     = float(start_smell or 10**self.distance_power)
        self.verbose         = int(verbose)
        self._initalized     = False
        
        if self.min_round_trips and self.max_round_trips: 
            self.min_round_trips = min(self.min_round_trips, self.max_round_trips)
        if self.min_ants and self.max_ants:               
            self.min_ants        = min(self.min_ants, self.max_ants)

    def solve_initialize(self, problem_path: List[Any]) -> None:
        self.distances = { source: { dest: self.cost_fn(source, dest) for dest in problem_path } for source in problem_path }
        self.distance_cost = { source: { dest: 1 / (1 + self.distances[source][dest]) ** self.distance_power for dest in problem_path } for source in problem_path }
        self.pheromones = { source: { dest: self.start_smell for dest in problem_path } for source in problem_path }
        
        if self.ant_count <= 0: self.ant_count = len(problem_path)
        if self.ant_speed <= 0: self.ant_speed = np.median(list(chain(*[ d.values() for d in self.distances.values() ]))) // 5
        self.ant_speed = int(max(1,self.ant_speed))
        
        self.ants_used   = 0
        self.epochs_used = 0
        self.round_trips = 0
        self._initalized = True        

    def solve(self, problem_path: List[Any], restart=False) -> List[Tuple[int,int]]:
        if restart or not self._initalized:
            self.solve_initialize(problem_path)

        ants = {
            "distance":    np.zeros((self.ant_count,)).astype('int32'),
            "path":        [ [ problem_path[0] ]   for n in range(self.ant_count) ],
            "remaining":   [ set(problem_path[1:]) for n in range(self.ant_count) ],
            "path_cost":   np.zeros((self.ant_count,)).astype('int32'),
            "round_trips": np.zeros((self.ant_count,)).astype('int32'),
        }

        best_path       = None
        best_path_cost  = np.inf
        best_epochs     = []
        epoch           = 0
        time_start      = time.perf_counter()

        while True:
            epoch += 1
            ants_travelling = (ants['distance'] > self.ant_speed)
            ants['distance'][ ants_travelling ] -= self.ant_speed
            if all(ants_travelling): continue 
            
            ants_arriving       = np.invert(ants_travelling)
            ants_arriving_index = np.where(ants_arriving)[0]
            for i in ants_arriving_index:
                this_node = ants['path'][i][-1]
                next_node = self.next_node(ants, i)
                ants['distance'][i]  = self.distances[ this_node ][ next_node ]
                ants['remaining'][i] = ants['remaining'][i] - {this_node}
                ants['path_cost'][i] = ants['path_cost'][i] + ants['distance'][i]
                ants['path'][i].append( next_node )

                if not ants['remaining'][i] and ants['path'][i][0] == ants['path'][i][-1]:
                    self.ants_used  += 1
                    self.round_trips = max(self.round_trips, ants["round_trips"][i] + 1)

                    was_best_path = False
                    if ants['path_cost'][i] < best_path_cost:
                        was_best_path  = True
                        best_path_cost = ants['path_cost'][i]
                        best_path      = ants['path'][i]
                        best_epochs   += [ epoch ]
                        if self.verbose:
                            print({"path_cost": int(ants['path_cost'][i]), "ants_used": self.ants_used, "epoch": epoch, "round_trips": ants['round_trips'][i] + 1, "clock": int(time.perf_counter() - time_start)})

                    reward = 1
                    if self.reward_power: reward *= ((best_path_cost / ants['path_cost'][i]) ** self.reward_power)
                    if self.decay_power:  reward *= (self.round_trips ** self.decay_power)
                    for path_index in range( len(ants['path'][i]) - 1 ):
                        this_node = ants['path'][i][path_index]
                        next_node = ants['path'][i][path_index+1]
                        self.pheromones[this_node][next_node] += reward
                        self.pheromones[next_node][this_node] += reward
                        if was_best_path:
                            self.pheromones[this_node][next_node] *= self.best_path_smell
                            self.pheromones[next_node][this_node] *= self.best_path_smell

                    ants["distance"][i]     = 0
                    ants["path"][i]         = [ problem_path[0] ]
                    ants["remaining"][i]    = set(problem_path[1:])
                    ants["path_cost"][i]    = 0
                    ants["round_trips"][i] += 1

            if not len(best_epochs): continue 
            if self.time or self.min_time or self.timeout:
                clock = time.perf_counter() - time_start
                if self.time:
                    if clock > self.time: break
                    else:                 continue
                if self.min_time and clock < self.min_time: continue
                if self.timeout  and clock > self.timeout:  break
            
            if self.min_round_trips and self.round_trips <  self.min_round_trips: continue        
            if self.max_round_trips and self.round_trips >= self.max_round_trips: break
            if self.min_ants and self.ants_used <  self.min_ants: continue        
            if self.max_ants and self.ants_used >= self.max_ants: break            
            if self.stop_factor and epoch > (best_epochs[-1] * self.stop_factor): break
            if True: continue
            
        self.epochs_used = epoch
        self.round_trips = np.max(ants["round_trips"])
        return best_path

    def next_node(self, ants, index):
        this_node   = ants['path'][index][-1]
        weights     = []
        weights_sum = 0
        if not ants['remaining'][index]: return ants['path'][index][0]  
        for next_node in ants['remaining'][index]:
            if next_node == this_node: continue
            reward = (self.pheromones[this_node][next_node] ** self.pheromone_power * self.distance_cost[this_node][next_node])
            weights.append( (reward, next_node) )
            weights_sum   += reward
        rand = random.random() * weights_sum
        for (weight, next_node) in weights:
            if rand > weight: rand -= weight
            else:             break
        return next_node

def AntColonyRunner(Kecamatan, verbose=False, map_img=None, label={}, algorithm=AntColonySolver, **kwargs):
    solver     = algorithm(cost_fn=distance, verbose=verbose, **kwargs)
    start_time = time.perf_counter()
    result     = solver.solve(Kecamatan)
    stop_time  = time.perf_counter()
    if label: kwargs = { **label, **kwargs }
        
    print("N={:<3d} | {:5.0f} -> {:4.0f} | {:4.0f}s | ants: {:5d} | trips: {:4d} | ".format(
          len(Kecamatan), path_distance(Kecamatan), path_distance(result), 
          (stop_time - start_time), solver.ants_used, solver.round_trips) + 
          " ".join([ f"{k}={v}" for k,v in kwargs.items() if k not in ['min_time', 'max_time'] ]))
    
    if map_img is not None:
        show_path(result, map_img=map_img)
    return result

# ==========================================
# MAIN EXECUTION (CLI BOUNDARY)
# ==========================================
if __name__ == "__main__":
    # 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Optimasi Rute Antar Kantor Kecamatan (TSP) dengan ACO")
    parser.add_argument('--city', type=str, default='Surabaya', help="Nama kota yang ingin diproses (Contoh: Surabaya, Sidoarjo, Depok)")
    parser.add_argument('--ants', type=int, default=64, help="Jumlah semut dalam simulasi (default: 64)")
    parser.add_argument('--iterations', type=int, default=100, help="Jumlah maksimum round trips/iterasi (default: 100)")
    parser.add_argument('--alpha', type=float, default=1.25, help="Bobot feromon / pheromone_power (default: 1.25)")
    parser.add_argument('--beta', type=float, default=1.0, help="Bobot visibilitas/jarak / distance_power (default: 1.0)")
    parser.add_argument('--stats', action='store_true', help="Jalankan komputasi statistik Pandas di akhir")
    
    args = parser.parse_args()

    print(f"\n--- Menjalankan Optimasi untuk Kota: {args.city.upper()} ---")
    print(f"Parameter: {args.ants} Semut, {args.iterations} Iterasi Maks, Alpha={args.alpha}, Beta={args.beta}\n")

    # 2. Pemuatan Data Dinamis Berdasarkan CLI
    try:
        Kecamatan_Dict = load_coordinates(args.city)
        Kecamatan = list(sorted(Kecamatan_Dict.items()))
        map_img = load_city_map(args.city)
        print("Banyak Kecamatan = ", len(Kecamatan))
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # 3. Proses Pemetaan Awal
    show_Kecamatan(Kecamatan, map_img=map_img)
    
    # 4. Pemecahan Masalah TSP Utama
    print("\n[Mencari Rute Terbaik...]")
    results = AntColonyRunner(
        Kecamatan, 
        verbose=True, 
        map_img=map_img, 
        ant_count=args.ants, 
        max_round_trips=args.iterations, 
        pheromone_power=args.alpha, 
        distance_power=args.beta
    )

    print("\n✅ Eksekusi selesai. Silakan cek folder assets/results/ untuk melihat gambar hasil optimasi (path.png dan Kecamatan.png).")

    # 5. Blok Statistik Opsional (Gunakan flag --stats di terminal)
    if args.stats:
        print("\n--- Memulai Variasi Acak Statistik (Ini mungkin memakan waktu) ---")
        results_converged = [ AntColonyRunner(Kecamatan) for i in range(10) ]
        results_timed = [ AntColonyRunner(Kecamatan, time=5) for i in range(10) ]

        results_converged_stats = pd.Series([ path_distance(path) for path in results_converged ]).describe()
        results_timed_stats     = pd.Series([ path_distance(path) for path in results_timed     ]).describe()
        difference_stats = results_converged_stats - results_timed_stats

        df = pd.DataFrame({
            "results_converged": results_converged_stats,
            "results_timed":     results_timed_stats,
            "difference":        difference_stats,
        }).T.round(1)

        print("\n=== HASIL STATISTIK ===")
        print(df)