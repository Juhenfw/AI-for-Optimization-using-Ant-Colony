# Define Peta Kecamatan
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image

# Load Gambar Peta
surabaya_map = mpimg.imread(".vscode\Optimization\Tugas_AI_Optimization\Lokasi_Kantor_Kecamatan_Surabaya.png")

def show_Kecamatan(path, w=8, h=8):
    """Plot a TSP path overlaid on a map of the Kantor Kecamatan in Surabaya."""
    if isinstance(path, dict):      
        path = list(path.values())
    if isinstance(path[0][0], str): 
        path = [ item[1] for item in path ]    
    plt.imshow(surabaya_map)    
    for x0, y0 in path:
        plt.plot(x0, y0, 'y*', markersize=5)  # y* = yellow star for starting point        
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])
    plt.savefig('Kecamatan.png')  # Save the figure
    plt.show()  # Display the figure

def show_path(path, starting_kecamatan=None, w=8, h=8):
    """Plot a TSP path overlaid on a map of the Kantor Kecamatan in Surabaya."""
    if isinstance(path, dict):      
        path = list(path.values())
    if isinstance(path[0][0], str): 
        path = [ item[1] for item in path ]
    
    starting_kecamatan = starting_kecamatan or path[0]
    x, y = list(zip(*path))
    (x0, y0) = starting_kecamatan
    plt.imshow(surabaya_map)
    plt.plot(x0, y0, 'y*', markersize=5)  # y* = yellow star for starting point
    plt.plot(x + x[:1], y + y[:1])  # sertakan titik awal di akhir jalur
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])
    plt.savefig('path.png')  # Save the figure
    plt.show()  # Display the figure

###########################################################################################################################

def distance(xy1, xy2) -> float:
    if isinstance(xy1[0], str): xy1 = xy1[1]; xy2 = xy2[1];  # if xy1 == ("Name", (x,y))
    return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

def path_distance(path) -> int:
    if isinstance(path, dict):      path = list(path.values())  # if path == {"Name": (x,y)}
    if isinstance(path[0][0], str): path = [ item[1] for item in path ]  # if path == ("Name", (x,y))
    return int(sum(
        [ distance(path[i],  path[i + 1]) for i in range(len(path) - 1) ] +
        [ distance(path[-1], path[0]) ]  # include cost of return journey
    ))

##################################################################################################################################

# Define the Kecamatan and their coordinates
# Reference Map: 
# https://www.arcgis.com/home/webmap/viewer.html?webmap=7461583abab341b888603657e4a41ccf
# https://www.surabaya.go.id/id/berita/58988/daftar-nama--alamat-camat-kota
# List Kecamatan di Surabaya
Kecamatan = {
    "Semampir": (539, 294),
    "Kenjeran": (616.4, 300),
    "Simokerto": (565, 354),
    "Bulak": (656, 316),
    "Tambaksari": (569, 397.8),
    "Mulyorejo": (646.4, 410),
    "Krembangan": (468, 312.8),
    "Bubutan": (490.4, 361),
    "Genteng": (542.2, 402),
    "Gubeng": (563, 439),
    "Sukolilo": (607.5, 521.4),
    "Rungkut": (613.7, 595),
    "Gunung Anyar": (640.8, 651.3),
    "Tenggilis Mejoyo": (562.8, 569.8),
    "Wonocolo": (517.4, 587),
    "Gayungan": (443.8, 639),
    "Jambangan": (435, 597),
    "Karang Pilang": (391.9, 625),
    "Wiyung": (378.4, 574.2),
    "Dukuh Pakis": (418.2, 459.1),
    "Lekarsantri": (228, 552),
    "Sawahan": (438.7, 486),
    "Wonokromo": (488.4, 501),
    "Tegalsari": (511.5, 485),
    "Sukomanunggal": (421.8, 430.8),
    "Sambikerep": (244, 464),
    "Tandes": (328.8, 403),
    "Asemrowo": (434.7, 381),
    "Benowo": (207.7, 383),
    "Pakal": (172, 349.8),
    "Pabean Cantian": (484.9, 268.6)
}


######################################## TESTING ##################################################
# List Kecamatan di Sidoarjo
'''Kecamatan = {
    "Balongbendo": (173, 210.3),
    "Buduran": (521, 219),
    "Candi": (503, 323),
    "Gedangan": (522.6, 171.1),
    "Jabon": (514, 429),
    "Krembung": (358, 373),
    "Krian": (283, 192.7),
    "Prambon": (259, 303.7),
    "Porong": (458.3, 422.5),
    "Sedati": (596, 177.3),
    "Sidoarjo": (500, 273.3),
    "Sukodono": (440.8, 208),
    "Taman": (474, 108),
    "Tanggulangin": (504, 372),
    "Tarik": (191, 274),
    "Tulangan": (399, 324.5),
    "Waru": (562.5, 104),
    "Wonoayu": (366, 257)
}'''

# List Kecamatan di Depok
'''Kecamatan = {
    "Beji": (449, 303.3),
    "Bojongsari": (193, 365),
    "Cilodong": (559.4, 479),
    "Cimanggis": (650.8, 286),
    "Cinere": (339, 148.4),
    "Cipayung": (375, 485.4),
    "Limo": (301, 266.4),
    "Pancoran Mas": (459, 405.2),
    "Sawangan": (243.9, 401),
    "Sukmajaya": (524, 372.5),
    "Tapos": (703.9, 511.7)
}'''

Kecamatan = list(sorted(Kecamatan.items()))
print("Banyak Kecamatan = ", len(Kecamatan))

# Pemetaan Kecamatan pada Peta
show_Kecamatan(Kecamatan)

###################################################################################################################################

# Rute Pada Peta
show_path(Kecamatan)
path_distance(Kecamatan)

###################################################################################################################################

# Hitung total jarak jalur
total_distance = path_distance(Kecamatan)
print("Total jarak jalur:", total_distance, "units")

# Tampilkan gambar yang disimpan di VSCode
display(Image(filename='Kecamatan.png'))
display(Image(filename='path.png'))


###################################################################################################################################

import time
from itertools import chain
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import random

class AntColonySolver:
    def __init__(self,
                 cost_fn:                 
                 
                 Callable[[Any,Any], Union[float,int]],                         
                 
                 time=0,                  # berjalan untuk jangka waktu tertentu
                 min_time=0,              # waktu proses minimum
                 timeout=0,               # waktu maksimum dalam detik untuk menjalankan
                 stop_factor=2,           # berapa kali menggandakan usaha setelah menemukan jalur terbaik yang baru
                 min_round_trips=10,      # jumlah minimum perjalanan pulang-pergi sebelum berhenti
                 max_round_trips=0,       # jumlah maksimum perjalanan pulang-pergi sebelum berhenti                
                 min_ants=0,              # Jumlah total semut yang digunakan
                 max_ants=0,              # Jumlah total semut yang digunakan
                 
                 ant_count=64,            # ini adalah batas bawah dari rentang hampir-optimal untuk kinerja numpy
                 ant_speed=1,             # berapa banyak langkah yang ditempuh semut per epoch

                 distance_power=1,        # kekuatan jarak yang mempengaruhi feromon                 
                 pheromone_power=1.25,    # kekuatan di mana perbedaan dalam feromon diamati
                 decay_power=0,           # seberapa cepat feromon meluruh
                 reward_power=0,          # imbalan feromon relatif berdasarkan best_path_length/path_length 
                 best_path_smell=2,       # pengganda ratu untuk feromon setelah menemukan jalur terbaik baru                  
                 start_smell=0,           # jumlah feromon awal [0 defaults to `10**self.distance_power`]

                 verbose=False,

    ):
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


    def solve_initialize(
            self,
            problem_path: List[Any],
    ) -> None:
        ### Cache jarak antar node
        self.distances = {
            source: {
                dest: self.cost_fn(source, dest)
                for dest in problem_path
            }
            for source in problem_path
        }

        ### Cache biaya jarak antar node - pembagian dalam loop yang ketat itu mahal
        self.distance_cost = {
            source: {
                dest: 1 / (1 + self.distances[source][dest]) ** self.distance_power
                for dest in problem_path
            }
            for source in problem_path
        }

        ### Ini menyimpan jejak feromon yang perlahan terbentuk
        self.pheromones = {
            source: {
                # Mendorong semut untuk mulai menjelajah ke segala arah dan titik terjauh
                dest: self.start_smell
                for dest in problem_path
            }
            for source in problem_path
        }
        
        ### Sanitasi parameter masukan
        if self.ant_count <= 0:
            self.ant_count = len(problem_path)
        if self.ant_speed <= 0:
            self.ant_speed = np.median(list(chain(*[ d.values() for d in self.distances.values() ]))) // 5
        self.ant_speed = int(max(1,self.ant_speed))
        
        ### Ekspor Heuristik
        self.ants_used   = 0
        self.epochs_used = 0
        self.round_trips = 0
        self._initalized = True        


    def solve(self,
              problem_path: List[Any],
              restart=False,
    ) -> List[Tuple[int,int]]:
        if restart or not self._initalized:
            self.solve_initialize(problem_path)

        ### Define Semut
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

            ### Berjalannya semut yang tervektorisasi
            # Pengoptimalan kecil, testing against `> self.ant_speed` rather than `> 0` 
            #       menghindari komputasi ants_arriving di bagian utama dari loop ketat ini
            ants_travelling = (ants['distance'] > self.ant_speed)
            ants['distance'][ ants_travelling ] -= self.ant_speed
            if all(ants_travelling):
                continue  # lewati pemeriksaan terminasi sampai semut berikutnya tiba
            
            ### Pemeriksaan vektorisasi kedatangan semut
            ants_arriving       = np.invert(ants_travelling)
            ants_arriving_index = np.where(ants_arriving)[0]
            for i in ants_arriving_index:

                ### semut telah tiba di next_node
                this_node = ants['path'][i][-1]
                next_node = self.next_node(ants, i)
                ants['distance'][i]  = self.distances[ this_node ][ next_node ]
                ants['remaining'][i] = ants['remaining'][i] - {this_node}
                ants['path_cost'][i] = ants['path_cost'][i] + ants['distance'][i]
                ants['path'][i].append( next_node )

                ### semut telah kembali ke koloninya
                if not ants['remaining'][i] and ants['path'][i][0] == ants['path'][i][-1]:
                    self.ants_used  += 1
                    self.round_trips = max(self.round_trips, ants["round_trips"][i] + 1)

                    ### Kami telah menemukan jalan baru yang terbaik - beri tahu Ratu
                    was_best_path = False
                    if ants['path_cost'][i] < best_path_cost:
                        was_best_path  = True
                        best_path_cost = ants['path_cost'][i]
                        best_path      = ants['path'][i]
                        best_epochs   += [ epoch ]
                        if self.verbose:
                            print({
                                "path_cost":   int(ants['path_cost'][i]),
                                "ants_used":   self.ants_used,
                                "epoch":       epoch,
                                "round_trips": ants['round_trips'][i] + 1,
                                "clock":       int(time.perf_counter() - time_start),
                            })

                    ### meninggalkan jejak feromon
                    # melakukan ini hanya setelah semut tiba di rumah akan meningkatkan eksplorasi awal
                    #  * self.round_trips memiliki efek merusak jejak feromon lama
                    # ** self.reward_power = -3 memiliki efek mendorong semut untuk menjelajahi rute yang lebih panjang
                    #                           yang dikombinasikan dengan penggandaan feromon untuk jalur_terbaik
                    reward = 1
                    if self.reward_power: reward *= ((best_path_cost / ants['path_cost'][i]) ** self.reward_power)
                    if self.decay_power:  reward *= (self.round_trips ** self.decay_power)
                    for path_index in range( len(ants['path'][i]) - 1 ):
                        this_node = ants['path'][i][path_index]
                        next_node = ants['path'][i][path_index+1]
                        self.pheromones[this_node][next_node] += reward
                        self.pheromones[next_node][this_node] += reward
                        if was_best_path:
                            # Ratu memerintahkan untuk menggandakan jumlah semut yang mengikuti jalur terbaik baru ini                            
                            self.pheromones[this_node][next_node] *= self.best_path_smell
                            self.pheromones[next_node][this_node] *= self.best_path_smell


                    ### reset semut
                    ants["distance"][i]     = 0
                    ants["path"][i]         = [ problem_path[0] ]
                    ants["remaining"][i]    = set(problem_path[1:])
                    ants["path_cost"][i]    = 0
                    ants["round_trips"][i] += 1


            ### Do we terminate?
            
            # Selalu menunggu setidaknya 1 solusi (catatan: 2+ solusi tidak dijamin)
            if not len(best_epochs): continue 
            
            # Timer lebih diprioritaskan dibandingkan batasan lainnya
            if self.time or self.min_time or self.timeout:
                clock = time.perf_counter() - time_start
                if self.time:
                    if clock > self.time: break
                    else:                 continue
                if self.min_time and clock < self.min_time: continue
                if self.timeout  and clock > self.timeout:  break
            
            # Epoch pertama hanya memiliki bau awal - pertanyaan: berapa banyak epoch yang diperlukan untuk mendapatkan hasil yang masuk akal?
            if self.min_round_trips and self.round_trips <  self.min_round_trips: continue        
            if self.max_round_trips and self.round_trips >= self.max_round_trips: break

            # Faktor ini paling erat kaitannya dengan kekuatan komputasi                
            if self.min_ants and self.ants_used <  self.min_ants: continue        
            if self.max_ants and self.ants_used >= self.max_ants: break            
            
            # Mari terus lipatgandakan upaya kita hingga kita tidak dapat menemukan apa pun lagi
            if self.stop_factor and epoch > (best_epochs[-1] * self.stop_factor): break
                                
            # Tidak ada lagi yang menghentikan kita: Ratu memerintahkan semut untuk melanjutkan!      
            if True: continue
            
            
            
        ### Kami (mudah-mudahan) telah menemukan jalur yang hampir optimal, lapor kembali ke Ratu
        self.epochs_used = epoch
        self.round_trips = np.max(ants["round_trips"])
        return best_path


    def next_node(self, ants, index):
        this_node   = ants['path'][index][-1]

        weights     = []
        weights_sum = 0
        if not ants['remaining'][index]: return ants['path'][index][0]  # return home
        for next_node in ants['remaining'][index]:
            if next_node == this_node: continue
            reward = (
                    self.pheromones[this_node][next_node] ** self.pheromone_power
                    * self.distance_cost[this_node][next_node]  # Lebih memilih jalan yang lebih pendek
            )
            weights.append( (reward, next_node) )
            weights_sum   += reward

        # Pilih jalur acak yang sebanding dengan berat feromon
        rand = random.random() * weights_sum
        for (weight, next_node) in weights:
            if rand > weight: rand -= weight
            else:             break
        return next_node
            
        
def AntColonyRunner(Kecamatan, verbose=False, plot=False, label={}, algorithm=AntColonySolver, **kwargs):
    solver     = algorithm(cost_fn=distance, verbose=verbose, **kwargs)
    start_time = time.perf_counter()
    result     = solver.solve(Kecamatan)
    stop_time  = time.perf_counter()
    if label: kwargs = { **label, **kwargs }
        
    for key in ['verbose', 'plot', 'animate', 'label', 'min_time', 'max_time']:
        if key in kwargs: del kwargs[key]
    print("N={:<3d} | {:5.0f} -> {:4.0f} | {:4.0f}s | ants: {:5d} | trips: {:4d} | "
          .format(len(Kecamatan), path_distance(Kecamatan), path_distance(result), (stop_time - start_time), solver.ants_used, solver.round_trips)
          + " ".join([ f"{k}={v}" for k,v in kwargs.items() ])
    )
    if plot:
        show_path(result)
    return result

# Pemecahan Masalah TSP #
results = AntColonyRunner(Kecamatan, distance_power=1, verbose=True, plot=True)

# Menggunakan Semut Liar #
# results = AntColonyRunner(Kecamatan, distance_power=0, min_time=30, verbose=True, plot=True)

# VARIASI ACAK STATISTIK #
import pandas as pd
import matplotlib.pyplot as plt

# Misalkan fungsi path_distance dan AntColonyRunner telah didefinisikan sebelumnya
results_converged = [ AntColonyRunner(Kecamatan) for i in range(10) ]
results_timed = [ AntColonyRunner(Kecamatan, time=10) for i in range(10) ]

# Menghitung statistik untuk kedua hasil
results_converged_stats = pd.Series([ path_distance(path) for path in results_converged ]).describe()
results_timed_stats     = pd.Series([ path_distance(path) for path in results_timed     ]).describe()

# Menghitung perbedaan antara hasil konvergen dan hasil terjadwal
difference_stats = results_converged_stats - results_timed_stats

# Menggabungkan hasil dalam satu DataFrame
df = pd.DataFrame({
    "results_converged": results_converged_stats,
    "results_timed":     results_timed_stats,
    "difference":        difference_stats,
}).T.round(1)

# Membuat tabel dengan Matplotlib
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

# Menampilkan tabel
plt.show()
