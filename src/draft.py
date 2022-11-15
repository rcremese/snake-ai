from abc import ABCMeta, abstractmethod
import numpy as np

class foo(metaclass=ABCMeta):
    def __init__(self, a) -> None:
        self.a = a

    @abstractmethod
    def foo(self):
        print(self.a)

    def __repr__(self) -> str:
        return f"{__class__.__name__}"

class bar(foo):
    def foo(self):
        super().foo()
        print('Hello')

    def __repr__(self) -> str:
        return super().__repr__()

b = bar(2)
b.foo()
print(b)


pos = (5,10)
a = np.repeat([pos], 10, axis=0)
print(a)
b = np.zeros(10, dtype=bool)
print(b)
b[5]=True
print(a[b, :])