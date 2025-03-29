import tkinter as tk
from tkinter import ttk
import random
import math
from tkinter import simpledialog


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


# Функция для поиска и отображения гамильтонова пути
def calculate():
    n = len(vertices)  # Получаем количество вершин в графе
    if n < 2:  # Если вершин меньше 2, выводим сообщение об ошибке
        path_label.config(text="Добавьте хотя бы 2 вершины!")
        return

    # Создаём матрицу смежности с бесконечными значениями для отсутствующих рёбер
    graph = [[float('inf')] * n for _ in range(n)]
    for (u, v), w in edges.items():  # Заполняем матрицу смежности весами рёбер
        graph[u][v] = w

    start_vertex = start_vertex_var.get()  # Получаем выбранную начальную вершину
    if start_vertex != "Рандомная вершина":  # Если начальная вершина выбрана явно
        start_vertex = int(start_vertex)  # Преобразуем индекс в целое число

    # Ищем гамильтонов путь
    path = hamiltonian_path(graph, start_vertex, n)

    if path:  # Если путь найден
        path_str = " → ".join(map(str, path))  # Формируем строку для отображения пути
        path_label.config(text=f"Найденный путь: {path_str}")  # Отображаем путь
        cycle_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]  # Формируем рёбра цикла
        draw_graph(output_canvas, vertices, {(u, v): graph[u][v] for u, v in cycle_edges}, path)  # Рисуем гамильтонов цикл
        total_length = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))  # Считаем длину пути
        path_label.config(text=f"Найденный путь: {' → '.join(map(str, path))}\nОбщая длина пути: {total_length}")  # Выводим длину пути

    else:  # Если путь не найден
        path_label.config(text="Алгоритм зашел в тупик.")  # Сообщаем, что путь не найден
        draw_graph(output_canvas, vertices, edges)  # Отображаем исходный граф



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


# Запуск
root.mainloop()
