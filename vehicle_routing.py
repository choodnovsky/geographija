# vehicle_routing.py

"""
Развозка по Москве — Vehicle Routing Problem
=============================================
Вход  : deliveries_input.csv  (колонка 'address', опционально 'lat','lon')
Выход : moscow_delivery.html  — интерактивная карта маршрутов
        courier_stats.png     — графики по курьерам
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import folium
from IPython.display import display

# ================================================================
# 1. ПАРАМЕТРЫ — меняй под свою задачу
# ================================================================

INPUT_CSV     = 'deliveries_input.csv'   # входной файл с адресами
DEPOT_COORDS  = (55.759660, 37.531388)   # координаты склада
DEPOT_ADDRESS = 'Склад Ермакова Роща'
# N_COURIERS рассчитывается автоматически в блоке 6
WORK_START    = 8  * 3600                 # 08:00
WORK_END      = 19 * 3600                # 19:00
MAX_SHIFT     = WORK_END - WORK_START     # 39 600 сек = 11 ч
AVG_SPEED_MS  = 17 * 1000 / 3600         # 17 км/ч → м/с (с учётом пробок)
STOP_TIME     = 15 * 60                   # 15 мин на одну доставку
GRAPHML_PATH  = 'graph/moscow.graphml'          # путь к графу Москвы
MAP_PATH = 'result/moscow_delivery.html'
REPORT_PATH = 'result/report.txt'


# ================================================================
# 2. ЧТЕНИЕ CSV И ГЕОКОДИНГ
# ================================================================

print("=" * 60)
print("  ЧТЕНИЕ АДРЕСОВ")
print("=" * 60)

try:
    df = pd.read_csv(INPUT_CSV, encoding='utf-8').loc[:]
except FileNotFoundError:
    print(f"❌ Файл {INPUT_CSV} не найден.")
    print("   Создайте CSV с колонкой 'address':")
    print("   address")
    print("   Москва, ул. Тверская, 1")
    print("   ...")
    sys.exit(1)

if 'address' not in df.columns:
    print("❌ В CSV нет колонки 'address'")
    sys.exit(1)

print(f"Загружено строк: {len(df)}")

# Определяем какие строки нужно геокодировать
has_coords = 'lat' in df.columns and 'lon' in df.columns
if not has_coords:
    df['lat'] = np.nan
    df['lon'] = np.nan

need_geo = df[df['lat'].isna() | df['lon'].isna()]
print(f"Нужно геокодировать: {len(need_geo)} адресов")

if len(need_geo) > 0:
    print(f"Запрашиваем координаты (~{len(need_geo)//2 + 1} мин)...\n")
    geolocator = Nominatim(user_agent="moscow_vrp_v2")
    geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=0.7)

    for i in need_geo.index:
        addr = str(df.at[i, 'address']).strip()
        query = addr if 'москва' in addr.lower() else f"Москва, {addr}"
        try:
            loc = geocode(query)
            if loc:
                df.at[i, 'lat'] = round(loc.latitude,  6)
                df.at[i, 'lon'] = round(loc.longitude, 6)
                print(f"  ✅ [{i+1}] {addr[:45]:<45} → {loc.latitude:.4f}, {loc.longitude:.4f}")
            else:
                print(f"  ⚠️  [{i+1}] не найден: {addr}")
        except Exception as e:
            print(f"  ❌ [{i+1}] ошибка: {e}")

# Убираем строки без координат
before = len(df)
df = df.dropna(subset=['lat', 'lon']).reset_index(drop=True)
if len(df) < before:
    print(f"\n⚠️  Отброшено {before - len(df)} адресов без координат")

df['lat'] = df['lat'].astype(float)
df['lon'] = df['lon'].astype(float)

N_DELIVERIES = len(df)
print(f"\nИтого точек доставки: {N_DELIVERIES}")

if N_DELIVERIES == 0:
    print("❌ Нет ни одной точки. Проверь CSV.")
    sys.exit(1)

# ================================================================
# 3. ЗАГРУЗКА ГРАФА
# ================================================================

print("\n" + "=" * 60)
print("  ЗАГРУЗКА ГРАФА")
print("=" * 60)

G = ox.load_graphml(GRAPHML_PATH)
print(f"Граф загружен: {len(G.nodes)} узлов, {len(G.edges)} рёбер")

# Оставляем только наибольший strongly connected component —
# гарантирует что между любыми двумя узлами есть путь по дорогам
G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
print(f"Связный граф : {len(G.nodes)} узлов, {len(G.edges)} рёбер")

# ================================================================
# 4. ПРИВЯЗКА К ГРАФУ + ДЕДУПЛИКАЦИЯ
# ================================================================

print("\nПривязываем точки к узлам графа...")

depot_node = ox.distance.nearest_nodes(G, DEPOT_COORDS[1], DEPOT_COORDS[0])

df['node'] = ox.distance.nearest_nodes(G, df['lon'].tolist(), df['lat'].tolist())

# Убираем точки совпавшие со складом или друг с другом
df = df[df['node'] != depot_node]
df = df.drop_duplicates(subset='node').reset_index(drop=True)

if len(df) < N_DELIVERIES:
    print(f"⚠️  После дедупликации: {len(df)} точек (было {N_DELIVERIES})")
    N_DELIVERIES = len(df)

all_nodes = [depot_node] + df['node'].tolist()
N = len(all_nodes)
print(f"Узлов в задаче: {N} (1 склад + {N_DELIVERIES} доставок)")

# ================================================================
# 5. МАТРИЦА ВРЕМЕНИ ПО РЕАЛЬНОМУ ГРАФУ
# ================================================================

print(f"\n" + "=" * 60)
print(f"  МАТРИЦА {N}×{N} ПО РЕАЛЬНЫМ ДОРОГАМ")
print("=" * 60)
print("Подождите ~2-3 мин...\n")

dist_matrix = np.zeros((N, N), dtype=np.int32)
time_matrix = np.zeros((N, N), dtype=np.int32)

for i in range(N):
    try:
        lengths = dict(nx.single_source_dijkstra_path_length(G, all_nodes[i], weight='length'))
    except Exception:
        lengths = {}

    for j in range(N):
        if i == j:
            continue
        d = lengths.get(all_nodes[j])
        if d is None or d <= 0:
            # Fallback — прямое расстояние × 1.35
            n1   = G.nodes[all_nodes[i]]
            n2   = G.nodes[all_nodes[j]]
            dlat = (n1['y'] - n2['y']) * 111_000
            dlon = (n1['x'] - n2['x']) * 111_000 * np.cos(np.radians(n1['y']))
            d    = np.sqrt(dlat**2 + dlon**2) * 1.35

        d         = min(int(d), 80_000)
        drive_sec = int(d / AVG_SPEED_MS)
        dist_matrix[i][j] = d
        time_matrix[i][j] = min(
            drive_sec + (STOP_TIME if j != 0 else 0),
            MAX_SHIFT
        )

    if i % 10 == 0:
        print(f"  {i+1}/{N}...")

print(f"\nМатрица готова ✅")
print(f"Макс. расстояние : {dist_matrix.max()/1000:.1f} км")
print(f"Макс. время пути : {time_matrix.max()/60:.0f} мин")
display(pd.DataFrame(dist_matrix))
display(pd.DataFrame(time_matrix))


# ================================================================
# 6. АВТОРАСЧЁТ ЧИСЛА КУРЬЕРОВ + КЛАСТЕРИЗАЦИЯ ПО ЗОНАМ
# ================================================================

print("\n" + "=" * 60)
print("  АВТОРАСЧЁТ ЧИСЛА КУРЬЕРОВ")
print("=" * 60)

from sklearn.cluster import KMeans

# Реальная оценка нагрузки:
# Запускаем жадный TSP на всех точках как будто один курьер —
# получаем суммарное время езды по всем точкам в хорошем порядке
unvisited  = list(range(1, N))
tsp_order  = [0]
cur        = 0
while unvisited:
    nearest = min(unvisited, key=lambda j: time_matrix[cur][j])
    tsp_order.append(nearest)
    unvisited.remove(nearest)
    cur = nearest
tsp_order.append(0)

# Суммируем время перегонов + стоянки
total_drive = sum(time_matrix[tsp_order[k]][tsp_order[k+1]] - (STOP_TIME if tsp_order[k+1] != 0 else 0)
                  for k in range(len(tsp_order) - 1))
total_drive = max(total_drive, 0)
total_stops = N_DELIVERIES * STOP_TIME
total_work  = total_drive + total_stops

# Делим суммарную нагрузку на смену с запасом 15% на возврат в депо
N_COURIERS = max(1, int(np.ceil(total_work / (MAX_SHIFT * 0.85))))
N_COURIERS = min(N_COURIERS, N_DELIVERIES)

total_h = total_work / 3600
print(f"Суммарная нагрузка (езда+стоянки) : {total_h:.1f} ч")
print(f"Смена с запасом 15%%               : {MAX_SHIFT * 0.85 / 3600:.1f} ч")
print(f"Курьеров потребуется               : {N_COURIERS}")
print(f"Загрузка на курьера                : ~{total_h/N_COURIERS:.1f} ч / {MAX_SHIFT/3600:.0f} ч смены")

# ================================================================
# КЛАСТЕРИЗАЦИЯ — жёсткое деление на непересекающиеся зоны
# ================================================================

print("\n" + "=" * 60)
print("  КЛАСТЕРИЗАЦИЯ ТОЧЕК")
print("=" * 60)

coords_km = df[['lat', 'lon']].values
kmeans    = KMeans(n_clusters=N_COURIERS, random_state=42, n_init=20, max_iter=1000)
df['cluster'] = kmeans.fit_predict(coords_km)

# Выравниваем по нагрузке (суммарное время зоны), а не по числу точек
def zone_load(cid):
    idx = [int(i) + 1 for i in df[df['cluster'] == cid].index]
    if not idx:
        return 0
    # Жадный TSP внутри зоны для оценки реального времени езды
    unvis = idx.copy()
    cur2  = 0
    load  = 0
    while unvis:
        n = min(unvis, key=lambda j: time_matrix[cur2][j])
        load += time_matrix[cur2][n]
        unvis.remove(n)
        cur2 = n
    load += time_matrix[cur2][0]   # возврат на склад
    return load  # уже включает STOP_TIME из time_matrix

# Балансируем строго по MAX_SHIFT:
# перекидываем точки из зон > MAX_SHIFT в зоны с запасом
for _ in range(2000):
    loads = {c: zone_load(c) for c in range(N_COURIERS)}
    over  = [c for c, l in loads.items() if l > MAX_SHIFT]
    under = [c for c, l in loads.items() if l < MAX_SHIFT * 0.92]
    if not over:
        break
    for b in over:
        if not under:
            break
        s = min(under, key=lambda c: loads[c])
        big_pts  = df[df['cluster'] == b]
        s_center = df[df['cluster'] == s][['lat', 'lon']].mean()
        dists    = (big_pts['lat'] - s_center['lat'])**2 + (big_pts['lon'] - s_center['lon'])**2
        df.at[dists.idxmin(), 'cluster'] = s
        loads[b] = zone_load(b)
        loads[s] = zone_load(s)
        if loads[s] >= MAX_SHIFT * 0.92:
            under.remove(s)

print(f"{'Зона':<6} {'Точек':>6}  {'~Нагрузка, ч':>13}  Статус")
all_ok = True
for c in range(N_COURIERS):
    n   = len(df[df['cluster'] == c])
    wl  = zone_load(c) / 3600
    bar = chr(9608) * max(1, int(wl / (MAX_SHIFT / 3600) * 15))
    ok  = 'OK' if wl <= MAX_SHIFT / 3600 else '! ПРЕВЫШЕНИЕ'
    if wl > MAX_SHIFT / 3600:
        all_ok = False
    print(f"  {c+1:<4} {n:>6}     {wl:>5.1f} h  {bar}  {ok}")
if all_ok:
    print("\nВсе зоны укладываются в смену!")
else:
    print("\nОстались превышения - добавь курьера через N_COURIERS вручную")

df = df.reset_index(drop=True)

# ================================================================
# 7. OR-TOOLS TSP ВНУТРИ КАЖДОЙ ЗОНЫ
# ================================================================

print("\n" + "=" * 60)
print("  ОПТИМИЗАЦИЯ МАРШРУТОВ ВНУТРИ ЗОН")
print("=" * 60)

COLORS = [
    '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
    '#42d4f4','#f032e6','#469990','#9A6324','#808000',
    '#ff6b6b','#51cf66','#339af0','#ffa94d','#cc5de8',
    '#22b8cf','#f06595','#20c997','#fab005','#868e96',
]

courier_stats  = []
courier_routes = {}
courier_stops  = {}
courier_addrs  = {}

for vid in range(N_COURIERS):
    zone = df[df['cluster'] == vid]
    if zone.empty:
        continue

    # Индексы зоны в общей матрице (0=склад, 1..N=точки)
    zone_node_idx = [int(i) + 1 for i in zone.index]

    # Строим локальную матрицу: склад + точки зоны
    local_nodes = [0] + zone_node_idx          # 0 = склад
    Nz          = len(local_nodes)
    local_time  = [[time_matrix[local_nodes[i]][local_nodes[j]] for j in range(Nz)] for i in range(Nz)]

    # OR-Tools TSP для этой зоны
    mgr = pywrapcp.RoutingIndexManager(Nz, 1, 0)
    mdl = pywrapcp.RoutingModel(mgr)

    def make_cb(lt):
        def cb(fi, ti):
            return lt[mgr.IndexToNode(fi)][mgr.IndexToNode(ti)]
        return cb

    cb_idx = mdl.RegisterTransitCallback(make_cb(local_time))
    mdl.SetArcCostEvaluatorOfAllVehicles(cb_idx)
    mdl.AddDimension(cb_idx, 0, MAX_SHIFT, True, 'Time')

    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.time_limit.seconds = 15

    sol = mdl.SolveWithParameters(sp)

    # Извлекаем порядок обхода
    if sol:
        idx = mdl.Start(0)
        ordered_local = []
        while not mdl.IsEnd(idx):
            ordered_local.append(mgr.IndexToNode(idx))
            idx = sol.Value(mdl.NextVar(idx))
        # Переводим локальные индексы обратно в глобальные
        route_idx = [local_nodes[li] for li in ordered_local] + [0]
    else:
        # Fallback: жадный ближайший сосед
        unvisited = zone_node_idx.copy()
        route_idx = [0]
        cur = 0
        while unvisited:
            nearest = min(unvisited, key=lambda j: time_matrix[cur][j])
            route_idx.append(nearest)
            unvisited.remove(nearest)
            cur = nearest
        route_idx.append(0)

    stops_idx = [i for i in route_idx if i != 0]

    # Строим реальные пути по графу
    segments    = []
    stop_coords = []
    stop_addrs  = []
    total_dist  = 0

    for k in range(len(route_idx) - 1):
        u      = all_nodes[route_idx[k]]
        v      = all_nodes[route_idx[k + 1]]
        path   = nx.shortest_path(G, u, v, weight='length')
        length = nx.shortest_path_length(G, u, v, weight='length')

        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
        segments.append(coords)
        total_dist += length

        if route_idx[k + 1] != 0:
            stop_coords.append(coords[-1])
            stop_addrs.append(str(df.iloc[route_idx[k + 1] - 1]['address']))

    drive_sec  = total_dist / AVG_SPEED_MS
    total_sec  = drive_sec + len(stops_idx) * STOP_TIME
    finish_sec = WORK_START + total_sec

    arrival_times = []
    t        = WORK_START
    prev_idx = 0
    for si in route_idx[1:-1]:
        t += time_matrix[prev_idx][si]
        arrival_times.append(f"{int(t//3600):02d}:{int((t%3600)//60):02d}")
        prev_idx = si

    courier_routes[vid] = segments
    courier_stops[vid]  = stop_coords
    courier_addrs[vid]  = list(zip(stop_addrs, arrival_times))

    courier_stats.append({
        'Курьер'     : vid + 1,
        'Доставок'   : len(stops_idx),
        'Пробег, км' : round(total_dist / 1000, 1),
        'В пути, ч'  : round(drive_sec / 3600, 1),
        'Стоянки, ч' : round(len(stops_idx) * STOP_TIME / 3600, 1),
        'Итого, ч'   : round(total_sec / 3600, 1),
        'Финиш'      : f"{int(finish_sec//3600):02d}:{int((finish_sec%3600)//60):02d}",
        'В смену'    : '✅' if total_sec <= MAX_SHIFT else '❌',
    })

    print(f"  Курьер {vid+1}: {len(stops_idx)} точек, {round(total_dist/1000,1)} км, финиш {courier_stats[-1]['Финиш']}")

# ================================================================
# 8. ТАБЛИЦА ОТЧЁТ + ПУТЕВЫЕ ЛИСТЫ
# ================================================================

print("\n" + "=" * 60)
print("  ИТОГИ")
print("=" * 60)

stats_df = pd.DataFrame(courier_stats).set_index('Курьер')
print(stats_df.to_string())
print("=" * 60)

total_visited = stats_df['Доставок'].sum()
print(f"Итого доставок  : {total_visited} / {N_DELIVERIES}", end="  ")
print("✅ Все посещены!" if total_visited == N_DELIVERIES else f"⚠️  Не хватает {N_DELIVERIES - total_visited}")
print(f"Общий пробег    : {stats_df['Пробег, км'].sum():.0f} км")
print(f"Средний пробег  : {stats_df['Пробег, км'].mean():.1f} км/курьер")
print(f"Укладываются    : {(stats_df['В смену'] == '✅').sum()} / {len(stats_df)} курьеров")

# Путевые листы
print("\n" + "=" * 60)
print("  ПУТЕВЫЕ ЛИСТЫ")
print("=" * 60)
for vid in sorted(courier_routes.keys()):
    stat = next(s for s in courier_stats if s['Курьер'] == vid + 1)
    print(f"\nКурьер {vid+1}  |  {stat['Доставок']} точек  |  {stat['Пробег, км']} км  |  финиш {stat['Финиш']}")
    print(f"  Старт: {DEPOT_ADDRESS} в 08:00")
    for order, (addr, arrive) in enumerate(courier_addrs[vid], 1):
        print(f"  {order:>2}. {arrive}  {addr}")
    print(f"  Возврат на склад")

# ================================================================
# 9. СОХРАНЕНИЕ ОТЧЁТА В TXT
# ================================================================

import os
os.makedirs('result', exist_ok=True)

from datetime import datetime

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    today = datetime.now().strftime('%d.%m.%Y %H:%M')
    f.write(f"ОТЧЁТ ПО ДОСТАВКЕ — {today}\n")
    f.write("=" * 60 + "\n\n")

    # Сводная таблица
    f.write("ИТОГИ\n")
    f.write("=" * 60 + "\n")
    f.write(stats_df.to_string())
    f.write("\n" + "=" * 60 + "\n")
    total_visited = stats_df['Доставок'].sum()
    f.write(f"Итого доставок  : {total_visited} / {N_DELIVERIES}")
    f.write("  Все посещены!\n" if total_visited == N_DELIVERIES else f"  Не хватает {N_DELIVERIES - total_visited}\n")
    f.write(f"Общий пробег    : {stats_df['Пробег, км'].sum():.0f} км\n")
    f.write(f"Средний пробег  : {stats_df['Пробег, км'].mean():.1f} км/курьер\n")
    f.write(f"Укладываются    : {(stats_df['В смену'] == '✅').sum()} / {len(stats_df)} курьеров\n")

    # Путевые листы
    f.write("\n" + "=" * 60 + "\n")
    f.write("ПУТЕВЫЕ ЛИСТЫ\n")
    f.write("=" * 60 + "\n")
    for vid in sorted(courier_routes.keys()):
        stat = next(s for s in courier_stats if s['Курьер'] == vid + 1)
        f.write(f"\nКурьер {vid+1}  |  {stat['Доставок']} точек  |  {stat['Пробег, км']} км  |  финиш {stat['Финиш']}\n")
        f.write(f"  Старт: {DEPOT_ADDRESS} в 08:00\n")
        for order, (addr, arrive) in enumerate(courier_addrs[vid], 1):
            f.write(f"  {order:>2}. {arrive}  {addr}\n")
        f.write(f"  Возврат на склад\n")

print(f"Отчёт → {REPORT_PATH} ✅")

# ================================================================
# 10. КАРТА МАРШРУТОВ
# ================================================================

print("Строим карту...")

m = folium.Map(location=DEPOT_COORDS, zoom_start=11, tiles='cartodbpositron')

# Склад
folium.Marker(
    DEPOT_COORDS,
    popup=f'<b>🏭 {DEPOT_ADDRESS}</b>',
    tooltip='Склад — старт в 08:00',
    icon=folium.Icon(color='red', icon='home', prefix='fa')
).add_to(m)

# Маршруты и точки
for vid in sorted(courier_routes.keys()):
    color    = COLORS[vid % len(COLORS)]
    stat     = next(s for s in courier_stats if s['Курьер'] == vid + 1)
    segments = courier_routes[vid]
    stops    = courier_stops[vid]
    addrs    = courier_addrs[vid]

    # Линия маршрута
    for seg in segments:
        folium.PolyLine(
            seg, color=color, weight=4, opacity=0.75,
            tooltip=f"Курьер {vid+1} | {stat['Доставок']} точек | {stat['Пробег, км']} км | финиш {stat['Финиш']}"
        ).add_to(m)

    # Точки с адресами и временем прибытия
    for idx, (coord, (addr, arrive)) in enumerate(zip(stops, addrs)):
        folium.CircleMarker(
            coord,
            radius=7,
            color='white',
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            tooltip=f"К{vid+1} · {arrive} · {addr[:30]}",
            popup=(
                f"<b>Курьер {vid+1}</b><br>"
                f"Остановка {idx+1}<br>"
                f"🕐 Прибытие: {arrive}<br>"
                f"📍 {addr}"
            )
        ).add_to(m)

# Легенда
legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:12px 16px; border-radius:8px;
     box-shadow:0 2px 8px rgba(0,0,0,0.3); font-family:Arial; font-size:12px;
     max-height:300px; overflow-y:auto;">
  <b>Маршруты курьеров</b><br><br>
"""
for vid in sorted(courier_routes.keys()):
    stat  = next(s for s in courier_stats if s['Курьер'] == vid + 1)
    color = COLORS[vid % len(COLORS)]
    legend_html += (
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{color};border-radius:50%;margin-right:6px;'
        f'vertical-align:middle;"></span>'
        f'Курьер {vid+1} — {stat["Доставок"]} точек, '
        f'{stat["Пробег, км"]} км, до {stat["Финиш"]}<br>'
    )
legend_html += "</div>"
m.get_root().html.add_child(folium.Element(legend_html))

m.save(MAP_PATH)
# Leaflet добавляет SVG-флаг через JS — скрываем его через CSS
with open(MAP_PATH, 'r', encoding='utf-8') as f:
    html = f.read()
html = html.replace('</head>', '<style>.leaflet-attribution-flag{display:none!important}</style>\n</head>', 1)
with open(MAP_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Карта  → {MAP_PATH} ✅")