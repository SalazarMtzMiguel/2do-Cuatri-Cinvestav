# Paso 1: Leer el archivo cadenas.txt
from collections import defaultdict

from graphviz import Digraph,Graph
from numpy.ma.core import shape, filled


def leer_cadenas():
    with open("cadenas3.txt", 'r') as file:
        cadenas = file.read().splitlines()
    return cadenas

# Paso 2: Identificar transiciones compartidas manteniendo el orden
def identificar_transiciones(cadenas):
    longitud = len(cadenas)
    array_de_listas = [[] for _ in range(longitud)]
    contador = 0
    dot = Digraph()
    dot.attr(rankdir='LR')
    dot.node('q0',shape='octagon',style='filled', fillcolor='lightgreen')
    mi_set=set()
    for i in range(longitud):
        contador = 0
        for j in range(len(cadenas[i]) - 1):
            array_de_listas[i].append((cadenas[i][j], cadenas[i][j + 1]))
            if contador==0:
                if len(mi_set)==0:
                    mi_set.add(('q0',cadenas[i][j]))
                    dot.edge('q0', cadenas[i][j], label=cadenas[i][j])
                dot.node(cadenas[i][j],shape='circle',label="")
                if (('q0',cadenas[i][j])) not in mi_set:
                    dot.edge('q0', cadenas[i][j], label=cadenas[i][j])
            if j+1==len(cadenas[i])-1:
                dot.node(cadenas[i][j + 1], shape='doublecircle',style='filled', fillcolor='lightblue',label="")
            else:
                dot.node(cadenas[i][j + 1], shape='circle',label="")
            if (cadenas[i][j], cadenas[i][j + 1]) not in mi_set:
                mi_set.add((cadenas[i][j], cadenas[i][j + 1]))
                dot.edge(cadenas[i][j], cadenas[i][j + 1], label=cadenas[i][j + 1])

            contador += 1
    print(array_de_listas)

    dot.render('afn', format='png')
    return array_de_listas

identificar_transiciones(leer_cadenas())


def crear_afn_por_cadena(cadenas):
    dot = Digraph()
    dot.attr(rankdir='LR')

    for index, cadena in enumerate(cadenas):
        # Prefijo de la cadena vacía (epsilon) hacia el primer carácter
        estado_inicial = f'q0_{index}'
        dot.node(estado_inicial, shape='point')
        primer_caracter = cadena[0]
        dot.node(primer_caracter, shape='circle', label=primer_caracter)
        dot.edge(estado_inicial, primer_caracter, label='ε')

        # Crear transiciones para cada carácter en la cadena
        for i in range(len(cadena) - 1):
            dot.node(cadena[i], shape='circle', label=cadena[i])
            dot.node(cadena[i + 1], shape='circle', label=cadena[i + 1])
            dot.edge(cadena[i], cadena[i + 1], label=cadena[i + 1])

        # Estado final
        estado_final = cadena[-1]
        dot.node(estado_final, shape='doublecircle', style='filled', fillcolor='lightblue', label=estado_final)

    dot.render('afn_por_cadena', format='png')


# Leer cadenas del archivo y crear AFN por cada cadena
cadenas = leer_cadenas()
crear_afn_por_cadena(cadenas)


def crear_afn_por_cadena(cadenas):
    dot = Digraph()
    dot.attr(rankdir='LR')

    for index, cadena in enumerate(cadenas):
        # Prefijo de la cadena vacía (epsilon) hacia el primer carácter
        estado_inicial = f'q0_{index}'
        dot.node(estado_inicial, shape='point')
        primer_caracter = f'{cadena[0]}_{index}'
        dot.node(primer_caracter, shape='circle', label=cadena[0])
        dot.edge(estado_inicial, primer_caracter, label='ε')

        # Crear transiciones para cada carácter en la cadena
        for i in range(len(cadena) - 1):
            estado_actual = f'{cadena[i]}_{index}'
            estado_siguiente = f'{cadena[i + 1]}_{index}'
            dot.node(estado_actual, shape='circle', label=cadena[i])
            dot.node(estado_siguiente, shape='circle', label=cadena[i + 1])
            dot.edge(estado_actual, estado_siguiente, label=cadena[i + 1])

        # Estado final
        estado_final = f'{cadena[-1]}_{index}'
        dot.node(estado_final, shape='doublecircle', style='filled', fillcolor='lightblue', label=cadena[-1])

    dot.render('afn_por_cadena', format='png')



def GrafoCobertura(cadenas):
    longitud = len(cadenas)
    array_de_listas = [[] for _ in range(longitud)]
    contador = 0
    dot = Digraph()
    dot.attr(rankdir='LR')
    #dot.node('q0',shape='octagon',style='filled', fillcolor='lightgreen')
    mi_set=set()
    for i in range(longitud):
        contador = 0
        for j in range(len(cadenas[i]) - 1):
            array_de_listas[i].append((cadenas[i][j], cadenas[i][j + 1]))
            dot.node(cadenas[i][j + 1], shape='circle',label=cadenas[i][j+1],regular="true")
            if (cadenas[i][j], cadenas[i][j + 1]) not in mi_set:
                mi_set.add((cadenas[i][j], cadenas[i][j + 1]))
                dot.edge(cadenas[i][j], cadenas[i][j + 1], label="")

            contador += 1
    print(array_de_listas)

    dot.render('GC', format='png')
    return array_de_listas

GrafoCobertura(leer_cadenas())


# Leer cadenas del archivo y crear AFN por cada cadena
cadenas = leer_cadenas()
crear_afn_por_cadena(cadenas)

def ordena_cadenas(cadenas):
    new_cadenas=sorted(cadenas,key=len,reverse=True)
    return new_cadenas
def crea_afn(cadenas):
    longitud = len(cadenas)
    array_de_listas = [[] for _ in range(longitud)]
    dot = Digraph()
    dot.attr(rankdir='LR')
    dot.node('q0', shape='octagon', style='filled', fillcolor='lightgreen')

    mi_set = set()
    transiciones = defaultdict(set)  # Almacena transiciones salientes por estado

    # Construir el grafo inicial
    for i in range(longitud):
        for j in range(len(cadenas[i]) - 1):
            origen = cadenas[i][j]
            destino = cadenas[i][j + 1]

            # Crear nodos y transiciones
            if j == 0 and (('q0', origen) not in mi_set):
                mi_set.add(('q0', origen))
                dot.edge('q0', origen, label=origen)

            dot.node(origen, shape='circle', label="")

            if j + 1 == len(cadenas[i]) - 1:
                dot.node(destino, shape='doublecircle', style='filled', fillcolor='lightblue', label="")
            else:
                dot.node(destino, shape='circle', label="")

            if (origen, destino) not in mi_set:
                mi_set.add((origen, destino))
                dot.edge(origen, destino, label=destino)
                transiciones[origen].add((destino, cadenas[i][j + 1]))

    # Función para fusionar estados que llevan a la misma transición final
    def colapsar_estados(transiciones):
        estado_equivalente = {}  # Mapeo de estados a su colapso
        final_transitions = defaultdict(list)

        # Identificar estados que apuntan al mismo destino con la misma transición
        for origen, destinos in transiciones.items():
            for destino, label in destinos:
                if destino == 'D':  # Solo nos importa el estado final
                    final_transitions[label].append(origen)

        # Colapsar estados previos
        for label, estados in final_transitions.items():
            if len(estados) > 1:
                colapsado = estados[0]
                for estado in estados[1:]:
                    estado_equivalente[estado] = colapsado

        return estado_equivalente

    estado_equivalente = colapsar_estados(transiciones)

    # Reconstruir el grafo colapsado
    dot_collapse = Digraph()
    dot_collapse.attr(rankdir='LR')
    dot_collapse.node('q0', shape='octagon', style='filled', fillcolor='lightgreen')

    processed_edges = set()

    for (origen, destino) in mi_set:
        # Resolver estados equivalentes
        origen = estado_equivalente.get(origen, origen)
        destino = estado_equivalente.get(destino, destino)

        # Evitar duplicar transiciones
        if (origen, destino) not in processed_edges:
            processed_edges.add((origen, destino))
            dot_collapse.edge(origen, destino, label=destino)

            # Definir tipo de nodo
            if destino.startswith('q') or destino.isupper():
                dot_collapse.node(destino, shape='doublecircle' if destino == 'D' else 'circle',
                                  style='filled' if destino == 'D' else '',
                                  fillcolor='lightblue' if destino == 'D' else '')

    # Guardar el grafo colapsado
    dot_collapse.render('afn_colapsado', format='png')
    return array_de_listas


crea_afn(leer_cadenas())