import math

import numpy as np

import graph
from prettytable import PrettyTable
import logging
import sys

from approximation import (
    LinearApproximation,
    SquareApproximation,
    ExponentialApproximation,
    LogApproximation,
    PowApproximation,
    CubicApproximation,
    Function,
)

targets = logging.StreamHandler(sys.stdout), logging.FileHandler('output.txt', mode='w', encoding='utf-8')
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)

METHODS = [
    LinearApproximation(),
    SquareApproximation(),
    ExponentialApproximation(),
    LogApproximation(),
    PowApproximation(),
    # CubicApproximation(),
]


def main():
    _append_new()

    function_table = _read_function_table()
    functions = []
    for a in METHODS:
        f = a.get_approximation(function_table)
        if f is not None:
            functions.append(f)

    functions.sort(key=lambda x: x.root_mean_square_deviation)
    results_table = PrettyTable()
    results_table.field_names = ["Функция", "Мера отклонения", "Среднеквадратичное отклонение"]
    for f in functions:
        results_table.add_row([f.text, round(f.s, 3), round(f.root_mean_square_deviation, 3)])
    logging.info(results_table)
    logging.info("Аппроксимирующая функция с наименьшим среднеквадратическим отклонением: " + functions[0].text)

    graph.draw(function_table, functions)


def _read_function_table() -> {float: float}:
    function_table = {}
    while True:
        filename = input("Введите имя файла для загрузки исходных данных "
                         "или пустую строку, чтобы ввести вручную: ")

        if filename == '':
            while True:
                line = input()
                if line == '':
                    if len(function_table) < 10:
                        print('(!) Необходимо не менее 10 точек')
                        continue
                    else:
                        break
                try:
                    if len(line.split()) != 2:
                        print('(!) Нужно ввсести два числа, через пробел')
                        continue
                    x, y = map(float, line.split())
                    function_table[x] = y
                except ValueError:
                    print('(!) Вы ввели не число')
            break
        else:
            try:
                f = open(filename, "r")
                for line in f.readlines():
                    try:
                        if len(line.split()) != 2:
                            continue
                        x, y = map(float, line.split())
                        function_table[x] = y
                    except ValueError:
                        continue
                if len(function_table) < 10:
                    print('(!) В файле менее 10 корректных точек')
                    continue
                break
            except FileNotFoundError:
                print('(!) Файл для загрузки исходных данных не найден.')

    return function_table


def _append_new():
    f = lambda x: 5*x**(-1)
    with open('in6.txt', 'w') as file:
        for e in np.arange(0.2, 3, 0.1):
            file.write(f'{e:.3f} {f(e):.3f}\n')


if __name__ == '__main__':
    main()
