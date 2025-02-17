from graphviz import Digraph

# Paso 1: Leer el archivo cadenas.txt
def leer_cadenas():
    with open("cadenas.txt", 'r') as file:
        cadenas = file.read().splitlines()
    return cadenas

# Paso 2: Identificar transiciones compartidas manteniendo el orden
def identificar_transiciones(cadenas):
    transiciones = set()
    dot = Digraph()
    dot.node('q0')  # Estado inicial

    for cadena in cadenas:
        estado_actual = 'q0'
        for i in range(len(cadena) - 1):
            transicion = (estado_actual, cadena[i], cadena[i + 1])
            if transicion not in transiciones:
                transiciones.add(transicion)
                nuevo_estado = f'q{len(transiciones)}'
                dot.node(nuevo_estado)
                dot.edge(estado_actual, nuevo_estado, label=cadena[i])
                estado_actual = nuevo_estado
            else:
                estado_actual = [estado for estado, char, _ in transiciones if estado == estado_actual and char == cadena[i]][0]

    dot.render('afn', format='png')
    return transiciones

# Leer las cadenas del archivo
cadenas = leer_cadenas()

# Identificar transiciones compartidas y construir el AFN
transiciones = identificar_transiciones(cadenas)