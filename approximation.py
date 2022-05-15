from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union

import logging

import numpy as np
from prettytable import PrettyTable

from numpy.ma import log, sqrt, exp


@dataclass
class Function:
    function: Callable
    text: str
    s: float
    root_mean_square_deviation: float


class BaseApproximation(ABC):
    """
    Базовый класс для методов аппроксимации
    """

    @staticmethod
    def solve_matrix22(a, b):
        delta = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        delta1 = a[0][0] * b[1] - b[0] * a[1][0]
        delta2 = b[0] * a[1][1] - a[0][1] * b[1]
        return (delta1 / delta, delta2 / delta) if delta != 0 else (None, None)

    @staticmethod
    def solve_matrix33(a, b):
        delta = (a[0][0] * a[1][1] * a[2][2] + a[1][0] * a[2][1] * a[0][2] + a[0][1] * a[1][2] * a[2][0]
                 - a[0][2] * a[1][1] * a[2][0] - a[0][1] * a[1][0] * a[2][2] - a[0][0] * a[1][2] * a[2][1])
        delta2 = (a[0][0] * b[1] * a[2][2] + a[1][0] * b[2] * a[0][2] + b[0] * a[1][2] * a[2][0]
                  - a[0][2] * b[1] * a[2][0] - b[0] * a[1][0] * a[2][2] - a[0][0] * a[1][2] * b[2])
        delta1 = (a[0][0] * a[1][1] * b[2] + a[1][0] * a[2][1] * b[0] + a[0][1] * b[1] * a[2][0]
                  - b[0] * a[1][1] * a[2][0] - a[0][1] * a[1][0] * b[2] - a[0][0] * b[1] * a[2][1])
        delta3 = (b[0] * a[1][1] * a[2][2] + b[1] * a[2][1] * a[0][2] + a[0][1] * a[1][2] * b[2]
                  - a[0][2] * a[1][1] * b[2] - a[0][1] * b[1] * a[2][2] - b[0] * a[1][2] * a[2][1])
        return (delta1 / delta, delta2 / delta, delta3 / delta) if delta != 0 else (None, None, None)

    @staticmethod
    def print_approximation_table(
        function_table: dict,
        f: Function,
        function_type: str,
        decimals=3
    ):
        x = np.around(list(function_table.keys()), decimals)
        y = np.around(list(function_table.values()), decimals)
        approximated_y = np.around(list(round(f.function(x), decimals) for x in function_table.keys()), decimals)

        logging.info(function_type)
        approximation_table = PrettyTable()
        approximation_table.field_names = ["", *(i for i in range(1, len(y) + 1))]
        approximation_table.add_row(["x", *x])
        approximation_table.add_row(["y", *y])
        approximation_table.add_row([f.text, *approximated_y])
        approximation_table.add_row(["E", *(round(approximated_y[i] - y[i], decimals) for i in range(len(y)))])
        logging.info(approximation_table)

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_approximation(self, function_table: dict) -> Union[Function, None]:
        pass

    def __str__(self):
        return self.get_name()


class LinearApproximation(BaseApproximation):

    def get_name(self):
        return 'Линейная аппроксимация'

    def get_approximation(self, function_table: dict) -> Union[Function, None]:
        SX = sum(function_table.keys())
        SXX = sum(x * x for x in function_table.keys())
        SY = sum(function_table.values())
        SXY = sum(x * y for x, y in function_table.items())
        n = len(function_table)

        a, b = self.solve_matrix22([[n, SX], [SX, SXX]], [SY, SXY])
        if a is None:
            return None
        fun = lambda x: a * x + b
        s = sum((fun(x) - function_table[x]) ** 2 for x in function_table.keys())
        root_mean_square_deviation = sqrt(s / n)
        f = Function(fun, f'ф = {round(a, 3)}*x {round(b, 3):+}', s, root_mean_square_deviation)
        self.print_approximation_table(function_table, f, self.get_name())

        average_x = SX / n
        average_y = SY / n
        r = (sum((x - average_x) * (y - average_y) for x, y in function_table.items())
             / sqrt(sum((x - average_x) ** 2 for x in function_table.keys()) *
                    sum((y - average_y) ** 2 for y in function_table.values())))
        logging.info(f'Коэффициент корреляции Пирсона равен {round(r, 3)}')
        return f


class PowApproximation(BaseApproximation):

    def get_name(self):
        return 'Степенная аппроксимация'

    def get_approximation(self, function_table: dict) -> Union[Function, None]:
        try:
            SLNX = sum(log(x) for x in function_table.keys())
            SLNXX = sum(log(x) * log(x) for x in function_table.keys())
            SLNY = sum(log(y) for y in function_table.values())
            SLNXY = sum(log(x) * log(y) for x, y in function_table.items())
            n = len(function_table)
        except ValueError:
            return None

        try:
            b, a = self.solve_matrix22([[n, SLNX], [SLNX, SLNXX]], [SLNY, SLNXY])
            if a is None:
                return None
            a = exp(a)
            fun = lambda x: a * (x ** b)
            s = sum((fun(x) - function_table[x]) ** 2 for x in function_table.keys())
            root_mean_square_deviation = sqrt(s / n)
            f = Function(fun, f'ф = {round(a, 3)}*x^({round(b, 3)})', s, root_mean_square_deviation)
            self.print_approximation_table(function_table, f, self.get_name())
            return f
        except TypeError:
            return None


class ExponentialApproximation(BaseApproximation):

    def get_name(self):
        return 'Экспоненциальная фппроксимация'

    def get_approximation(self, function_table: dict) -> Union[Function, None]:
        try:
            SX = sum(function_table.keys())
            SXX = sum(x * x for x in function_table.keys())
            SLNY = sum(log(y) for y in function_table.values())
            SXLNY = sum(x * log(y) for x, y in function_table.items())
            n = len(function_table)
        except ValueError:
            return None

        try:
            a, b = self.solve_matrix22([[n, SX], [SX, SXX]], [SLNY, SXLNY])
            if a is None:
                return None
            fun = lambda x: (exp(a * x + b))
            s = sum((fun(x) - function_table[x]) ** 2 for x in function_table.keys())
            root_mean_square_deviation = sqrt(s / n)
            f = Function(fun, f'ф = e^({round(a, 3)}*x {round(b, 3):+})', s, root_mean_square_deviation)
            self.print_approximation_table(function_table, f, self.get_name())
            return f
        except TypeError:
            return None


class LogApproximation(BaseApproximation):

    def get_name(self):
        return 'Логарифмическая фппроксимация'

    def get_approximation(self, function_table: dict) -> Union[Function, None]:
        try:
            SLNX = sum(log(x) for x in function_table.keys())
            SLNXX = sum(log(x) * log(x) for x in function_table.keys())
            SY = sum(function_table.values())
            SYLNX = sum(log(x) * y for x, y in function_table.items())
            n = len(function_table)
        except ValueError:
            return None

        try:
            a, b = self.solve_matrix22([[n, SLNX], [SLNX, SLNXX]], [SY, SYLNX])
            if a is None:
                return None
            fun = lambda x: a * log(x) + b
            s = sum((fun(x) - function_table[x]) ** 2 for x in function_table.keys())
            root_mean_square_deviation = sqrt(s / n)
            f = Function(fun, f'ф = {round(a, 3)}*ln(x) {round(b, 3):+}', s, root_mean_square_deviation)
            self.print_approximation_table(function_table, f, self.get_name())
            return f
        except TypeError:
            return None


class SquareApproximation(BaseApproximation):

    def get_name(self):
        return 'Полиномиальная аппроксимация второй степени'

    def get_approximation(self, function_table: dict) -> Union[Function, None]:
        SX = sum(function_table.keys())
        SXX = sum(x * x for x in function_table.keys())
        SXXX = sum(x * x * x for x in function_table.keys())
        SXXXX = sum(x * x * x * x for x in function_table.keys())
        SY = sum(function_table.values())
        SXY = sum(x * y for x, y in function_table.items())
        SXXY = sum(x * x * y for x, y in function_table.items())
        n = len(function_table)

        a, b, c = self.solve_matrix33([[n, SX, SXX], [SX, SXX, SXXX], [SXX, SXXX, SXXXX]], [SY, SXY, SXXY])
        if a is None:
            return None
        fun = lambda x: a * x * x + b * x + c
        s = sum((fun(x) - function_table[x]) ** 2 for x in function_table.keys())
        root_mean_square_deviation = sqrt(s / n)
        f = Function(fun, f'ф = {round(a, 3):+}*x^2 {round(b, 3):+}*x {round(c, 3):+}',
                     s, root_mean_square_deviation)
        self.print_approximation_table(function_table, f, self.get_name())
        return f


class CubicApproximation(BaseApproximation):

    def get_name(self):
        return 'Полиномиальная аппроксимация третьей степени'

    def get_approximation(self, function_table: dict) -> Function:
        n = len(function_table)
        SX = np.array([x for x, _ in function_table.items()])
        SY = np.array([y for _, y in function_table.items()])
        a, b, c, d = np.polyfit(SX, SY, 3)
        function = lambda x: a * x ** 3 + b * x ** 2 + c * x + d
        s = sum((function(x) - y) ** 2 for x, y in function_table.items())
        root_mean_square_deviation = sqrt(s / n)
        f = Function(function, f'ф = {round(a, 3):+}*x^3 {round(b, 3):+}*x^2 {round(c, 3):+}*x+{round(d, 3)}', s, root_mean_square_deviation)
        # self.display(points, f, self.get_name())
        return f
