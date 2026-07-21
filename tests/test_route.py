import os
import sys
import math
import pytest

# Menambahkan folder 'src' ke dalam sistem path agar bisa mengimpor route_optimization.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from route_optimization import distance, load_coordinates

def test_distance_calculation():
    """Menguji apakah fungsi rumus jarak (Euclidean) menghitung dengan akurat"""
    # Kasus 1: Jarak titik ke dirinya sendiri harus 0
    point_a = (0, 0)
    assert distance(point_a, point_a) == 0.0

    # Kasus 2: Jarak Pythagoras 3-4-5
    point_b = (0, 0)
    point_c = (3, 4)
    assert distance(point_b, point_c) == 5.0

def test_distance_with_string_format():
    """Menguji perhitungan jarak jika input berupa tuple dengan string (nama kecamatan)"""
    point_1 = ("Kecamatan_A", (0, 0))
    point_2 = ("Kecamatan_B", (3, 4))
    
    assert distance(point_1, point_2) == 5.0

def test_load_coordinates_error_handling():
    """Menguji apakah program memunculkan error yang benar jika kota tidak ada"""
    with pytest.raises(FileNotFoundError):
        # Mencoba memuat data kota fiktif yang pasti tidak ada file txt-nya
        load_coordinates("KotaFiktifAtlantis")