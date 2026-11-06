import numpy  as np
import time
import os
from concurrent.futures import ThreadPoolExecutor

# 1. LOYIHA SOZLAMALARI

print("=" * 65)
print("IoT SENSORLARINI PARALLEL Z-SCORE STANDARDLASHTIRISH")
print("=" * 65)

ROWS = 1_010_000
CHUNK_SIZE = 100_000

SENSOR_NAMES = [
    "Temperature",
    "Pressure",
    "Vibration",
    "Humidity",
    "Voltage"
]

# 2. IoT SENSOR MA'LUMOTLARINI GENERATSIYA QILISH

print("\n[1] IoT sensor ma'lumotlari yaratilmoqda...")

np.random.seed(42)
# rng = np.random.default_rng(42)

data = np.column_stack([
    np.random.normal(35, 10, ROWS),      # Harorat
    np.random.normal(5, 2, ROWS),        # Bosim
    np.random.normal(0.5, 0.2, ROWS),    # Tebranish
    np.random.normal(60, 15, ROWS),      # Namlik
    np.random.normal(220, 20, ROWS)      # Kuchlanish
]).astype(np.float64)

print(f"Ma'lumotlar soni: {data.shape[0]:,}")
print(f"Sensorlar soni: {data.shape[1]}")
print(f"Ma'lumotlar hajmi: {data.nbytes / (1024**2):.2f} MB")

# 3. Z-SCORE UCHUN STATISTIK KO'RSATKICHLAR

print("\n[2] Statistik ko'rsatkichlar hisoblanmoqda...")

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

# Nolga teng standart og'ishdan himoyalanish
std[std == 0] = 1.0

for i, sensor in enumerate(SENSOR_NAMES):
    print(
        f"{sensor:15} | "
        f"Mean: {mean[i]:10.4f} | "
        f"Std: {std[i]:10.4f}"
    )

# 4. ODDIY (SEQUENTIAL) STANDARDLASHTIRISH

print("\n[3] Oddiy standardlashtirish boshlandi...")

start = time.perf_counter()

zscore_sequential = (data - mean) / std

sequential_time = time.perf_counter() - start

print(f"Sequential vaqt: {sequential_time:.4f} sekund")

# 5. PARALLEL STANDARDLASHTIRISH


print("\n[4] Parallel standardlashtirish boshlandi...")

zscore_parallel = np.empty_like(data)

# Protsessor yadrolari soni
CPU_COUNT = os.cpu_count() or 2

# Juda ko'p thread ochmaslik uchun chegaralash
WORKERS = min(8, CPU_COUNT)

# Ma'lumotlarni qismlarga ajratish
chunks = [
    (start, min(start + CHUNK_SIZE, ROWS))
    for start in range(0, ROWS, CHUNK_SIZE)
]

def process_chunk(bounds):
    start_idx, end_idx = bounds

    # Har bir thread o'ziga tegishli qismni qayta ishlaydi
    zscore_parallel[start_idx:end_idx] = (
        data[start_idx:end_idx] - mean
    ) / std

start = time.perf_counter()

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    list(executor.map(process_chunk, chunks))

parallel_time = time.perf_counter() - start

print(f"Parallel vaqt: {parallel_time:.4f} sekund")
print(f"CPU yadrolari: {CPU_COUNT}")
print(f"Thread soni: {WORKERS}")
print(f"Qismlar soni: {len(chunks)}")

# 6. NATIJALARNI TEKSHIRISH

print("\n[5] Natijalar tekshirilmoqda...")

is_correct = np.allclose(
    zscore_sequential,
    zscore_parallel,
    rtol=1e-10,
    atol=1e-10
)

print(f"Sequential va parallel natijalar: {is_correct}")

print("\nStandardlashtirilgan ma'lumotlar:")

for i, sensor in enumerate(SENSOR_NAMES):
    print(
        f"{sensor:15} | "
        f"Mean: {np.mean(zscore_parallel[:, i]):.6f} | "
        f"Std: {np.std(zscore_parallel[:, i]):.6f}"
    )

# 7. UNUMDORLIK TAHLILI

print("\n[6] Unumdorlik tahlili")

if parallel_time > 0:
    speedup = sequential_time / parallel_time
else:
    speedup = 0

print(f"Sequential: {sequential_time:.4f} sekund")
print(f"Parallel:   {parallel_time:.4f} sekund")
print(f"Speedup:    {speedup:.2f}x")

if parallel_time < sequential_time:
    print("Parallel ishlov berish tezroq natija berdi.")
else:
    print("Ushbu qurilmada sequential ishlov berish tezroq.")

# 8. NATIJALARNI FAYLGA SAQLASH

print("\n[7] Natijalar saqlanmoqda...")

np.save("iot_zscore_result.npy", zscore_parallel)

np.savetxt(
    "iot_zscore_result.csv",
    zscore_parallel,
    delimiter=",",
    header="Temperature,Pressure,Vibration,Humidity,Voltage",
    comments="",
    fmt="%.6f"
)

print("NPY fayl saqlandi: iot_zscore_result.npy")
print("CSV fayl saqlandi: iot_zscore_result.csv")

print("\n" + "=" * 65)
print("LOYIHA MUVAFFAQIYATLI YAKUNLANDI!")
print("=" * 65)
