class SwapNumbers:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def swap_using_temp(self):
        temp = self.a
        self.a = self.b
        self.b = temp
        return self.a, self.b


# Example usage
obj = SwapNumbers(10, 20)
print("After swap:", obj.swap_using_temp())
