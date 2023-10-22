from math import sqrt, gcd


class ModularRational:
    inverses = {}  # Modular field element inverses

    @staticmethod
    def modular_invert(n: int, modulus: int) -> int:
        if n == 0:
            return 0

        if modulus not in ModularRational.inverses or n not in ModularRational.inverses[modulus]:
            for i in range(modulus):
                if (i * n) % modulus == 1:
                    subset = ModularRational.inverses.get(modulus, [])
                    subset[n] = i
                    ModularRational.inverses[modulus] = subset
                    break

        return ModularRational.inverses[modulus][n]

    def __init__(self, numerator: int, denominator: int, modulus: int):
        assert type(numerator) is type(denominator) is type(modulus) is int
        assert denominator != 0
        assert modulus > 2  # TODO: enforce modulus primality/binarity

        self.modulus = modulus
        self.numerator = numerator % modulus
        self.denominator = denominator % modulus
        self.resolution = 0 if numerator == 0 else None

    def resolve(self) -> int:
        if self.resolution is None:
            resolved = self.numerator / self.denominator
            if resolved % 1 == 0:
                self.resolution = int(resolved)
            else:
                self.resolution = (
                                      self.numerator * ModularRational.modular_invert(self.denominator, self.modulus)
                                  ) % self.modulus

        return self.resolution


def simplify_rational(a: int, b: int) -> tuple[int, int]:
    d = gcd(a, b)

    a_d = a / d
    b_d = b / d

    # Should both remain integer
    assert a_d % 1 == 0 and b_d % 1 == 0, F'Simplification failed (a={a}, b={b}, a_d={a_d}, b_d={b_d})'

    return int(a_d), int(b_d)


def elliptic_points(a: int, b: int, modulus=5) -> list[tuple[int, int]]:
    assert type(a) is type(b) is type(modulus) is int
    assert modulus > 2
    # TODO: ensure modulus primality

    points = []
    for x in range(modulus):
        for y in range(modulus // 2):
            lhs = (y**2) % modulus
            rhs = ((x**3) + (a * x) + b) % modulus

            if lhs == rhs:
                points.append((x, y))

                if lhs > 0:
                    points.append((x, -y % modulus))

    return points


def modular_inverse(xP: int, yP: int, modulus=5) -> tuple:
    assert type(xP) is type(yP) is type(modulus) is int
    assert modulus > 2

    return xP, -yP % modulus


def elliptic_add(xP: int, yP: int, xQ: int, yQ: int, modulus=5) -> tuple:
    assert type(xP) is type(yP) is type(xQ) is type(yQ) is int is type(modulus) is int
    assert modulus > 2
    assert set([elem < modulus for elem in (xP, yP, xQ, yQ)]) == {True}

    # Try to deal with modular division
    #try:
        #numerator = yQ - yP
        #denominator = xQ - xP
        #numerator, denominator = simplify_rational(yQ - yP, xQ - xP)
        #quotient = modular_rectify(numerator, modulus) / modular_rectify(denominator, modulus)
        #assert quotient % 1 == 0
        #slope = modular_rectify(int(quotient), modulus)
        #assert slope % 1 == 0 #, F'Rational slope (P=({xP}, {yP}), Q=({xQ}, {yQ}), modulus={modulus}, slope=({numerator}/{denominator})%{modulus}={slope})'  # Ensure int slope
        #print('[DBG] Strategy #1 worked!')
    #except AssertionError:
        #numerator = modular_rectify(yQ - yP, modulus)
        #denominator = modular_rectify(xQ - xP, modulus)
        #numerator, denominator = simplify_rational(modular_rectify(yQ - yP, modulus), modular_rectify(xQ - xP, modulus))
        #quotient = numerator / denominator
        #assert quotient % 1 == 0
        #slope = modular_rectify(int(quotient), modulus)
        #assert slope % 1 == 0,\
        #    (F'Rational slope (P=({xP}, {yP}), Q=({xQ}, {yQ}), modulus={modulus}, '
        #     F'slope=({numerator}/{denominator})%{modulus}={slope})')  # Ensure int slope
        #print('[DBG] Strategy #2 required!')

    if xP == xQ and yP == yQ:
        # Point doubling since P = Q
        slope = ModularRational(yQ - yP, xQ - xP, modulus).resolve()
    else:
        # P != Q
        slope = ModularRational(3 * (xP ** 2), yP, modulus).resolve()

    slope = int(slope)
    xR = (slope**2) - xP - xQ
    yR = (slope * (xP - xR)) - yP

    return xR, yR


def main():
    M = 5
    for a in range(-12, 12):
        for b in range(-12, 12):
            points = elliptic_points(a, b, M)
            inverses = {
                (xP, yP): modular_inverse(xP, yP, M)
                for xP, yP in points
            }
            self_inverses = [
                point
                for point, inverse in inverses.items()
                if point == inverse
            ]
            non_singularity = ', non-singular' if sum([
                1
                for xP, yP in points
                if set([
                    # Singularity point check: P + Q = P, P != Q for all P, Q on the curve
                    elliptic_add(xP, yP, xQ, yQ, M)
                    for xQ, yQ in points
                    if xP != xQ and yP != yQ
                ]) == {(xP, yP)}
            ]) > 0 else ''
            print(F'a={a}, b={b}: {len(points)} points, {len(self_inverses)} self-inverses{non_singularity}')


if __name__ == '__main__':
    #main()
    print(elliptic_points(2, 5, 7))
