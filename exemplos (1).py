import numpy as np

print("-------------------------")
print("| Operações Matemáticas |")
print("-------------------------\n")

array1 = np.array([1 , 2 , 3 , 4])
array2 = np.array([10 , 20 , 30 , 40])

print(array1 + array2)
print(array1 - array2)
print(array1 * array2)
print(array1 / array2)
print(array1 ** 2)
print(array2 ** 2)
print(np.sqrt(array1))

print("\n--------------------------")
print("| Operações Estatísticas |")
print("--------------------------\n")

array3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(np.sum(array3)) # soma total → 45
print(np.mean(array3)) # média → 5.0
print(np.std(array3)) # desvio padrão
print(np.min(array3)) # mínimo → 1
print(np.max(array3)) # máximo → 9
print(np.sum(array3, axis=0)) # soma por coluna → [12 15 18]
print(np.sum(array3, axis=1)) # soma por linha → [6 15 24]

print("\n-----------------------")
print("| Funções Matemáticas |")
print("-----------------------\n")

array4 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

print(np.sin(array4)) # seno
print(np.cos(array4)) # cosseno
print(np.exp(array4)) # exponencial
print(np.log(array4[1:])) # logaritmo natural (evita log(0))
print(np.abs([-1, 2, -3])) # valor absoluto

print("\n------------------")
print("| Arredondamento |")
print("------------------\n")

array5 = np.array([1.234, 5.678, 9.999])

print(np.round(array5, 2))   # arredonda para 2 casas → [1.23, 5.68, 10.0]
print(np.floor(array5))      # arredonda pra baixo
print(np.ceil(array5))       # arredonda pra cima

print("\n-----------------------------")
print("| Valores Únicos e Contagem |")
print("-----------------------------\n")

array6 = np.array([1, 2, 2, 3, 3, 3, 4])

print(np.unique(array6))                      # [1, 2, 3, 4]
print(np.unique(array6, return_counts=True))  # valores + quantas vezes aparecem

