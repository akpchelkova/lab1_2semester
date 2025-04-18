import tkinter as tk
from tkinter import ttk
import random
import math
from tkinter import simpledialog
import time


# Глобальные переменные
history = []  # Список для хранения истории изменений (добавление/удаление вершин и рёбер)
vertices = []  # Список для хранения вершин графа
edges = {}  # Словарь для рёбер графа {(u, v): вес}, где u и v — вершины, вес — это длина ребра
selected_vertex = None  # Переменная для хранения выбранной вершины для создания рёбер

# Функция поиска кратчайшего гамильтонова цикла
def hamiltonian_path(graph, start, n):
    visited = set()  # Множество для хранения посещённых вершин
    path = []  # Список для хранения пути (сами вершины пути)

    # Вложенная функция для выполнения поиска в глубину (DFS)
    def dfs(v):
        # Если путь содержит все вершины (n), проверяем, можно ли замкнуть цикл
        if len(path) == n:
            if graph[v][start] != float('inf'):  # Проверяем, есть ли путь от последней вершины обратно в начальную
                path.append(start)  # Замыкаем цикл
                return True  # Успешно найден путь
            return False  # Невозможно замкнуть цикл

        # Перебираем все вершины, чтобы найти путь
        for u in range(n):
            # Если вершина не посещена и есть ребро между вершинами v и u
            if u not in visited and graph[v][u] != float('inf'):
                visited.add(u)  # Отметить вершину как посещённую
                path.append(u)  # Добавляем вершину в путь
                if dfs(u):  # Рекурсивный вызов DFS для вершины u
                    return True  # Если путь найден, возвращаем True
                visited.remove(u)  # Отменить посещение вершины
                path.pop()  # Убираем вершину из пути, так как путь не найден

        return False  # Если путь не найден, возвращаем False

    # Если начальная вершина - "Рандомная вершина", выбираем её случайным образом
    if start == "Рандомная вершина":
        start = random.randint(0, n - 1)

    path.append(start)  # Добавляем начальную вершину в путь
    visited.add(start)  # Отмечаем начальную вершину как посещённую

    # Запускаем поиск в глубину с начальной вершины
    if dfs(start):
        return path  # Возвращаем найденный путь
    return None  # Если путь не найден, возвращаем None

# Функция отрисовки графа
def draw_graph(canvas, vertices, edges, cycle=None):
    canvas.delete("all")  # Очищаем холст от старых рисунков

    # Рисуем вершины графа
    for i, (x, y) in enumerate(vertices):
        canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill="blue")  # Окружности для вершин
        canvas.create_text(x, y, text=str(i), fill="white", font=('Helvetica', 12))  # Номера вершин

    # Рисуем рёбра графа
    for (u, v) in edges:
        x1, y1 = vertices[u]  # Координаты вершины u
        x2, y2 = vertices[v]  # Координаты вершины v
        canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill="black")  # Рисуем линию для рёбер

    # Если передан гамильтонов цикл, рисуем его красным цветом
    if cycle:
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]  # Составляем рёбра для цикла
            x1, y1 = vertices[u]
            x2, y2 = vertices[v]
            canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill="red", width=3)  # Рисуем красные рёбра для цикла

# Функция добавления вершины
def add_vertex(event):
    x, y = event.x, event.y  # Получаем координаты клика мыши (x, y)
    vertices.append((x, y))  # Добавляем новую вершину в список вершин (сохраняем её координаты)
    history.append(("vertex", None))  # Записываем в историю действие добавления вершины
    draw_graph(input_canvas, vertices, edges)  # Перерисовываем граф с учётом новой вершины

    # Обновляем список значений для выпадающего списка выбора начальной вершины
    start_vertex_menu["values"] = ["Рандомная вершина"] + [str(i) for i in range(len(vertices))]

# Функция добавления рёбер
def add_edge(event):
    global selected_vertex  # Используем глобальную переменную для хранения выбранной вершины
    x, y = event.x, event.y  # Получаем координаты клика мыши (x, y)

    clicked_vertex = None  # Инициализируем переменную для хранения выбранной вершины
    for i, (vx, vy) in enumerate(vertices):  # Перебираем все вершины
        if math.sqrt((x - vx) ** 2 + (y - vy) ** 2) < 20:  # Проверяем, попал ли клик в вершину
            clicked_vertex = i  # Если попал, запоминаем индекс вершины
            break

    if clicked_vertex is None:  # Если клик не попал в вершину, выходим из функции
        return

    # Если ни одна вершина ещё не выбрана, выбираем текущую как начало ребра
    if selected_vertex is None:
        selected_vertex = clicked_vertex
    else:
        # Если выбрана другая вершина, создаём ребро между двумя вершинами
        if selected_vertex != clicked_vertex:
            weight = random.randint(1, 10)  # Случайным образом задаём вес рёбра
            edges[(selected_vertex, clicked_vertex)] = weight  # Добавляем ребро в словарь рёбер
            if bidirectional.get():  # Если включен чекбокс для двусторонних рёбер
                edges[(clicked_vertex, selected_vertex)] = random.randint(1, 10)  # Добавляем обратное ребро
            update_table()  # Обновляем таблицу рёбер
        selected_vertex = None  # Сбрасываем выбранную вершину
        history.append(("edge", (selected_vertex, clicked_vertex)))  # Записываем добавление рёбер в историю
        draw_graph(input_canvas, vertices, edges)  # Перерисовываем граф с учётом новых рёбер


# Функция обновления таблицы рёбер
def update_table():
    for row in tree.get_children():  # Для всех строк в таблице
        tree.delete(row)  # Удаляем старые строки из таблицы

    # Перебираем все рёбра и их веса, добавляем их в таблицу
    for (u, v), w in edges.items():
        tree.insert("", "end", values=(u, v, w))  # Вставляем новые строки с вершинами и весами рёбер


def calculate():
    n = len(vertices)
    if n < 2:
        path_label.config(text="Добавьте хотя бы 2 вершины!")
        return

    graph = [[float('inf')] * n for _ in range(n)]
    for (u, v), w in edges.items():
        graph[u][v] = w

    start_vertex = start_vertex_var.get()
    if start_vertex != "Рандомная вершина":
        start_vertex = int(start_vertex)

    start_time = time.time()  # старт замера времени

    if use_ant_colony.get():
        path, total_length = ant_colony_optimization(
            graph, start_vertex, n,
            use_odeyalo=use_odeyalo_mode.get()
        )
    elif use_simulated_annealing.get():
        path, total_length = simulated_annealing(
            graph, start_vertex, n,
            fast_mode=use_fast_annealing.get()
        )
    else:
        path = hamiltonian_path(graph, start_vertex, n)
        if path:
            total_length = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
        else:
            path_label.config(text="Алгоритм зашел в тупик.")
            draw_graph(output_canvas, vertices, edges)
            return

    elapsed = time.time() - start_time  # конец замера времени

    # Проверка на "невозможный маршрут"
    if any(graph[path[i]][path[i + 1]] == float('inf') for i in range(len(path) - 1)):
        path_label.config(text="Путь содержит отсутствующие рёбра (inf). Убедитесь, что граф связный.")
        draw_graph(output_canvas, vertices, edges)
        return

    path_str = " → ".join(map(str, path))
    cycle_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    draw_graph(output_canvas, vertices, {(u, v): graph[u][v] for u, v in cycle_edges}, path)
    path_label.config(
        text=f"Найденный путь: {path_str}\nОбщая длина пути: {total_length}\nВремя расчета: {elapsed:.4f} секунд"
    )



def edit_weight(event):
    selected_item = tree.selection()  # Получаем выбранный элемент из таблицы рёбер
    if not selected_item:  # Если ничего не выбрано, выходим из функции
        return

    item = selected_item[0]  # Получаем первый выбранный элемент (т.к. selection может вернуть несколько элементов)
    values = tree.item(item, "values")  # Получаем значения из выбранной строки (веса рёбер)

    u, v = int(values[0]), int(values[1])  # Преобразуем строки в индексы вершин, образующие ребро

    # Открываем диалоговое окно для ввода нового веса рёбер
    new_weight = simpledialog.askinteger("Изменить вес", f"Введите новый вес для ребра {u} → {v}:", minvalue=1)

    if new_weight is not None:  # Если пользователь ввёл новое значение веса
        edges[(u, v)] = new_weight  # Обновляем вес рёбра в словаре рёбер
        draw_graph(input_canvas, vertices, edges)  # Перерисовываем граф с новым весом рёбер
        update_table()  # Обновляем таблицу рёбер с новым значением


# Настройка интерфейса
root = tk.Tk()  # Создание основного окна приложения
root.title("Ориентированный граф с таблицей рёбер")  # Название окна

bidirectional = tk.BooleanVar(value=True)  # Переменная для чекбокса двусторонних рёбер (по умолчанию включен)

# Фрейм для левой части (графы)
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)  # Размещение фрейма слева с отступами

# Поле для входного графа
input_canvas = tk.Canvas(left_frame, width=500, height=300, bg="white")  # Холст для отображения входного графа
input_canvas.pack()  # Размещение холста в фрейме

# Поле для выходного графа
output_canvas = tk.Canvas(left_frame, width=500, height=300, bg="white")  # Холст для отображения выходного графа
output_canvas.pack()  # Размещение холста в фрейме

# Фрейм для правой части (таблица рёбер)
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)  # Размещение фрейма справа с отступами

# Таблица рёбер
tree = ttk.Treeview(right_frame, columns=("Вершина 1", "Вершина 2", "Длина"), show="headings")  # Создание таблицы с 3 столбцами
tree.heading("Вершина 1", text="Вершина 1")  # Заголовок первого столбца
tree.heading("Вершина 2", text="Вершина 2")  # Заголовок второго столбца
tree.heading("Длина", text="Длина")  # Заголовок третьего столбца
tree.pack()  # Размещение таблицы в фрейме

# Переменная для хранения выбранной начальной вершины
start_vertex_var = tk.StringVar(value="Рандомная вершина")

# Выпадающий список для выбора начальной вершины
start_vertex_menu = ttk.Combobox(right_frame, textvariable=start_vertex_var, state="readonly")
start_vertex_menu["values"] = ["Рандомная вершина"]  # По умолчанию только "Рандомная вершина"
start_vertex_menu.pack(pady=5)  # Размещение выпадающего списка с отступами

# Переменная для выбора алгоритма
use_simulated_annealing = tk.BooleanVar(value=False)

# Чекбокс для переключения алгоритма
annealing_checkbox = tk.Checkbutton(right_frame, text="Использовать имитацию отжига", variable=use_simulated_annealing)
annealing_checkbox.pack(pady=5)

use_fast_annealing = tk.BooleanVar(value=False)

fast_annealing_checkbox = tk.Checkbutton(
    right_frame,
    text="Сверхбыстрый отжиг",
    variable=use_fast_annealing
)
fast_annealing_checkbox.pack(pady=2)

use_ant_colony = tk.BooleanVar(value=False)
use_odeyalo_mode = tk.BooleanVar(value=False)

ant_checkbox = tk.Checkbutton(right_frame, text="Муравьиный алгоритм", variable=use_ant_colony)
ant_checkbox.pack(pady=5)

odeyalo_checkbox = tk.Checkbutton(right_frame, text="Модификация 'Одеяло'", variable=use_odeyalo_mode)
odeyalo_checkbox.pack(pady=2)


# Кнопка для запуска расчёта
calculate_button = tk.Button(right_frame, text="Рассчитать", command=calculate)
calculate_button.pack(pady=10)  # Размещение кнопки с отступами


# Функция очистки графа
def clear_graph():
    global vertices, edges, selected_vertex  # Определяем глобальные переменные, которые будут очищены
    vertices = []  # Очищаем список вершин
    edges = {}  # Очищаем словарь рёбер
    selected_vertex = None  # Сбрасываем выбранную вершину
    draw_graph(input_canvas, vertices, edges)  # Перерисовываем граф на входном холсте
    draw_graph(output_canvas, vertices, edges)  # Перерисовываем граф на выходном холсте
    update_table()  # Обновляем таблицу рёбер, удаляя все строки
    start_vertex_var.set("Рандомная вершина")  # Восстанавливаем начальное состояние выпадающего списка
    start_vertex_menu['menu'].delete(0, 'end')  # Очищаем выпадающий список начальных вершин


def undo_last_action():
    if history:  # Проверяем, есть ли действия в истории
        action, data = history.pop()  # Извлекаем последнее действие и данные из истории
        if action == "vertex":
            vertices.pop()  # Удаляем последнюю вершину из списка
        elif action == "edge":
            del edges[data]  # Удаляем последнее ребро из словаря рёбер
        draw_graph(input_canvas, vertices, edges)  # Перерисовываем граф на входном холсте с актуальными данными
        update_table()  # Обновляем таблицу рёбер, удалив соответствующие строки


path_label = tk.Label(right_frame, text="", font=("Helvetica", 12))  # Создание метки для отображения пути
path_label.pack(pady=5)  # Размещение метки на экране

# Чекбокс для выбора рисования рёбер в обе стороны
bidirectional_checkbox = tk.Checkbutton(right_frame, text="Рисовать вершины в обе стороны", variable=bidirectional)
bidirectional_checkbox.pack(pady=5)  # Размещение чекбокса с отступом

# Кнопка для отмены последнего действия
undo_button = tk.Button(right_frame, text="Отменить последний ход", command=undo_last_action)
undo_button.pack(pady=5)  # Размещение кнопки с отступом

# Кнопка для очистки поля
clear_button = tk.Button(right_frame, text="Очистить поле", command=clear_graph)
clear_button.pack(pady=10)  # Размещение кнопки с отступом


# Обработчики событий
input_canvas.bind("<Button-1>", add_vertex)  # Левый клик мыши на холсте для добавления вершины
input_canvas.bind("<Button-3>", add_edge)  # Правый клик мыши на холсте для добавления рёбер
tree.bind("<Return>", edit_weight)  # Нажатие клавиши Enter на таблице рёбер для редактирования веса

def simulated_annealing(graph, start, n, initial_temp=10000, cooling_rate=0.995, stop_temp=1e-8, max_iter=1000, fast_mode=False):
    current_path = list(range(n))
    if start != "Рандомная вершина":
        start = int(start)
        current_path.remove(start)
        random.shuffle(current_path)
        current_path = [start] + current_path
    else:
        random.shuffle(current_path)
        start = current_path[0]

    def path_length(path):
        length = 0
        for i in range(n - 1):
            if graph[path[i]][path[i + 1]] == float('inf'):
                return float('inf')
            length += graph[path[i]][path[i + 1]]
        length += graph[path[-1]][path[0]]
        return length

    current_cost = path_length(current_path)
    best_path = current_path[:]
    best_cost = current_cost
    temp = initial_temp
    k = 1

    while temp > stop_temp:
        for _ in range(max_iter):
            i, j = sorted(random.sample(range(1, n), 2))
            new_path = current_path[:]
            new_path[i:j+1] = reversed(new_path[i:j+1])

            new_cost = path_length(new_path)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_path = new_path
                current_cost = new_cost
                if current_cost < best_cost:
                    best_path = current_path[:]
                    best_cost = current_cost

        # Снижение температуры:
        if fast_mode:
            temp = initial_temp / (k ** 2)  # сверхбыстрое охлаждение
        else:
            temp *= cooling_rate
        k += 1

    return best_path + [best_path[0]], best_cost

def ant_colony_optimization(graph, start, n, n_iterations=100, alpha=1, beta=5, rho=0.5, q=100, use_odeyalo=False):
    pheromone = [[1 for _ in range(n)] for _ in range(n)]  # Начальный феромон

    def path_cost(path):
        return sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))

    best_path = None
    best_cost = float('inf')

    for iteration in range(n_iterations):
        all_paths = []
        all_costs = []

        if use_odeyalo:
            # Один муравей из каждой вершины
            start_positions = list(range(n))
        else:
            # Один муравей из одного старта
            start_positions = [random.choice(range(n)) if start == "Рандомная вершина" else int(start)]

        for start_pos in start_positions:
            unvisited = set(range(n))
            current = start_pos
            path = [current]
            unvisited.remove(current)

            while unvisited:
                probs = []
                total = 0
                for next_city in unvisited:
                    tau = pheromone[current][next_city] ** alpha
                    eta = (1 / graph[current][next_city]) ** beta if graph[current][next_city] != float('inf') else 0
                    p = tau * eta
                    probs.append((next_city, p))
                    total += p

                if total == 0:
                    break

                r = random.uniform(0, total)
                cumulative = 0
                for next_city, p in probs:
                    cumulative += p
                    if cumulative >= r:
                        break

                path.append(next_city)
                unvisited.remove(next_city)
                current = next_city

            if len(path) == n and graph[path[-1]][path[0]] != float('inf'):
                path.append(path[0])
                cost = path_cost(path)
                all_paths.append(path)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

        # Обновление феромона
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - rho)

        for path, cost in zip(all_paths, all_costs):
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i + 1]] += q / cost

    return best_path, best_cost


# Запуск
root.mainloop()
