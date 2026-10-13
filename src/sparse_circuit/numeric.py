class ShiftDataVariance:
    def __init__(self):
        self.K = 0.0
        self.n = 0
        self.Ex = 0.0
        self.Ex2 = 0.0

    def add_variable(self, x: float):
        if self.n == 0:
            self.K = x
        self.n += 1
        self.Ex += x - self.K
        self.Ex2 += (x - self.K) ** 2

    def remove_variable(self, x: float):
        self.n -= 1
        self.Ex -= x - self.K
        self.Ex2 -= (x - self.K) ** 2

    def add_variables(self, xs: list[float]):
        # Python uses Neumaier's algorithm for the builtin sum() function,
        # which is more accurate than a simple loop.
        if self.n == 0 and xs:
            self.K = xs[0]
        self.n += len(xs)
        self.Ex += sum(x - self.K for x in xs)
        self.Ex2 += sum((x - self.K) ** 2 for x in xs)

    def get_mean(self) -> float:
        return self.K + self.Ex / self.n

    def get_variance(self) -> float:
        return (self.Ex2 - self.Ex**2 / self.n) / (self.n - 1)
