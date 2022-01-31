dims_table = {
    6: {'r': 12, 'M': 24, 'P': 110, 'd': 7, 'lap': 0.800, 'weight': 0.222},
    8: {'r': 16, 'M': 32, 'P': 115, 'd': 9, 'lap': 0.900, 'weight': 0.395},
    10: {'r': 20, 'M': 40, 'P': 120, 'd': 12, 'lap': 1.000, 'weight': 0.617},
    12: {'r': 24, 'M': 48, 'P': 125, 'd': 14, 'lap': 1.100, 'weight': 0.888},
    16: {'r': 32, 'M': 64, 'P': 130, 'd': 19, 'lap': 1.200, 'weight': 1.58},
    20: {'r': 70, 'M': 140, 'P': 190, 'd': 23, 'lap': 1.300, 'weight': 2.47},
    25: {'r': 87, 'M': 175, 'P': 240, 'd': 29, 'lap': 1.400, 'weight': 3.85},
    32: {'r': 112, 'M': 224, 'P': 305, 'd': 37, 'lap': 1.500, 'weight': 6.31},
    40: {'r': 140, 'M': 280, 'P': 380, 'd': 48, 'lap': 2.000, 'weight': 9.86}
}


class RebarClass:
    def __init__(self, shape_code, size, dims):
        self.ShapeCode = str(shape_code)
        if self.ShapeCode == '0':
            self.ShapeCode = '00'
        self.Size = size
        self.Dims = dims

    def check_dims(self):
        errors = 0
        if self.ShapeCode == '00':
            if self.Dims['A'] > 14000:
                print(f"A [{self.Dims['A']}mm] must be less than 14m")
                errors += 1
        elif self.ShapeCode == '13':
            if self.Dims['B'] < 2 * (dims_table[self.Size]['r'] + dims_table[self.Size]['d']):
                print(f"B [{self.Dims['B']}mm] must not be less than 2(r+d) [{2*(dims_table[self.Size]['r'] + dims_table[self.Size]['d'])}mm]")
                errors += 1
            if self.Dims['A'] < dims_table[self.Size]['P']:
                print(f"A [{self.Dims['A']}mm] must not be less than P [{dims_table[self.Size]['P']}mm]")
                errors += 1
            if self.Dims['C'] < dims_table[self.Size]['P']:
                print(f"C [{self.Dims['C']}mm] must not be less than P [{dims_table[self.Size]['P']}mm]")
                errors += 1
            if self.Dims['A'] < (self.Dims['B']/2 + 5 * dims_table[self.Size]['d']):
                print(f"A [{self.Dims['A']}mm] must not be less than B/2 + 5d [{self.Dims['B']/2 + 5 * dims_table[self.Size]['d']}mm]")
                errors += 1
            if self.Dims['C'] < (self.Dims['B']/2 + 5 * dims_table[self.Size]['d']):
                print(f"C [{self.Dims['C']}mm] must not be less than B/2 + 5d [{self.Dims['B']/2 + 5 * dims_table[self.Size]['d']}mm]")
                errors += 1
        elif self.ShapeCode == '21':
            if self.Dims['A'] < dims_table[self.Size]['P']:
                print(f"A [{self.Dims['A']}mm] must not be less than P [{dims_table[self.Size]['P']}mm]")
                errors += 1
            if self.Dims['C'] < dims_table[self.Size]['P']:
                print(f"C [{self.Dims['C']}mm] must not be less than P [{dims_table[self.Size]['P']}mm]")
                errors += 1
            if self.Dims['B'] < 4 * dims_table[self.Size]['d'] + 2 * dims_table[self.Size]['r']:
                print(f"B [{self.Dims['B']}mm] must not be less than 4d + 2r [{4 * dims_table[self.Size]['d'] + 2 * dims_table[self.Size]['r']}mm]")
                errors += 1
            if self.Size <= 16 and self.Dims['B'] < 10 * dims_table[self.Size]['d']:
                print(f"B [{self.Dims['B']}mm] must not be less than 10d [{10 * dims_table[self.Size]['d']}mm]")
                errors += 1
            if self.Size > 16 and self.Dims['B'] < 13 * dims_table[self.Size]['d']:
                print(f"B [{self.Dims['B']}mm] must not be less than 13d [{13 * dims_table[self.Size]['d']}mm]")
                errors += 1

        print(f"Check Complete: {errors} Errors Found")

    def length(self):
        if self.ShapeCode == '00':
            return self.Dims['A']
        elif self.ShapeCode == '13':
            return self.Dims['A'] + 0.57 * self.Dims['B'] + self.Dims['C'] - 1.6 * dims_table[self.Size]['d']
        elif self.ShapeCode == '21':
            return self.Dims['A'] + self.Dims['B'] + self.Dims['C'] - dims_table[self.Size]['r'] - 2 * dims_table[self.Size]['d']
        else:
            return None

    def __str__(self):
        return f"Shape Code: {self.ShapeCode} Bar Size: {self.Size}"


def get_dims_table():
    return dims_table


