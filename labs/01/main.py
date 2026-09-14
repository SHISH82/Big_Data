import math


# 1. Создание векторов x и y
x = list(range(-10, 6))
y = list(range(-5, 11))

print("1. Векторы:")
print("x =", x)
print("y =", y)


# 2. Создание вектора z
z = []

for i in range(len(x)):
    z.append(x[i])
    z.append(y[i])

print("\n2. Вектор z:")
print(z)

z.sort()

print("Отсортированный вектор z:")
print(z)


# 3. Функция вычисления норм вектора
def vector_norms(vector):
    norm_1 = 0
    norm_2 = 0
    norm_inf = 0

    for element in vector:
        norm_1 += abs(element)
        norm_2 += element ** 2

        if abs(element) > norm_inf:
            norm_inf = abs(element)

    norm_2 = math.sqrt(norm_2)

    return norm_1, norm_2, norm_inf


print("\n3. Нормы векторов:")

n1, n2, n_inf = vector_norms(x)
print("Вектор x:")
print("Норма 1 =", n1)
print(f"Норма 2 = {n2:.3f}")
print("Норма inf =", n_inf)

n1, n2, n_inf = vector_norms(y)
print("\nВектор y:")
print("Норма 1 =", n1)
print(f"Норма 2 = {n2:.3f}")
print("Норма inf =", n_inf)

n1, n2, n_inf = vector_norms(z)
print("\nВектор z:")
print("Норма 1 =", n1)
print(f"Норма 2 = {n2:.3f}")
print("Норма inf =", n_inf)


# 4. Функция вычисления факториала
def factorial(number):
    result = 1

    for i in range(1, number + 1):
        result *= i

    return result


print("\n4. Факториал")

number = int(input("Введите число: "))

if number < 0:
    print("Факториал отрицательного числа не определен")
else:
    print("Факториал числа", number, "=", factorial(number))


# 5. Ввод вектора и весов с клавиатуры
print("\n5. Вектор из 5 элементов")

vector = []

for i in range(5):
    value = float(input("Введите элемент вектора: "))
    vector.append(value)

print("Вектор:", vector)


while True:
    weights = []
    print("\nВведите 5 неотрицательных весов с суммой 1:")

    try:
        for i in range(5):
            value = float(input(f"Введите вес {i + 1}: "))
            weights.append(value)
    except ValueError:
        print("Ошибка: вес должен быть числом. Введите веса заново.")
        continue

    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        print("Ошибка: веса должны быть конечными неотрицательными числами. Введите веса заново.")
        continue

    if not math.isclose(sum(weights), 1.0):
        print("Ошибка: сумма весов должна быть равна 1. Введите веса заново.")
        continue

    break

print("Веса:", weights)


# Минимальный и максимальный элементы по модулю
min_element = vector[0]
max_element = vector[0]

for element in vector:
    if abs(element) < abs(min_element):
        min_element = element

    if abs(element) > abs(max_element):
        max_element = element


# Сумма элементов
vector_sum = sum(vector)


# Весовая норма
weighted_norm = 0

for i in range(5):
    weighted_norm += weights[i] * abs(vector[i])


print("\nРезультаты:")
print("Минимальный элемент по модулю:", min_element)
print("Максимальный элемент по модулю:", max_element)
print("Сумма элементов:", vector_sum)
print(f"Весовая норма: {weighted_norm:.2f}")
