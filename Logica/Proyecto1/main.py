import re

# Definir los tokens
TOKENS = [
    ('LETRA', r'[a-zA-Z]'),       # Letras del abecedario
    ('NEGACION', r'¬'),           # Negacion
    ('Y', r'∧'),                  # Conjuncion
    ('O', r'∨'),                  # Disyuncion
    ('IMPLICA', r'⇾'),            # Implicacion
    ('SIYSOLOSI', r'⇿'),          # Doble implicacion
    ('PAREN_IZQ', r'\('),         # Parentesis izquierdo
    ('PAREN_DER', r'\)'),         # Parentesis derecho
    ('ESPACIO', r'\s+'),          # Espacios en blanco
]

def lexer(codigo_entrada):
    tokens = []
    pos = 0

    while pos < len(codigo_entrada):
        match = None
        for tipo_token, patron in TOKENS:
            regex = re.compile(patron)
            match = regex.match(codigo_entrada, pos)
            if match:
                lexema = match.group(0)
                if tipo_token != 'ESPACIO':  # Ignorar espacios en blanco
                    tokens.append((tipo_token, lexema))
                pos = match.end()
                break
        if not match:
            raise SyntaxError(f"Caracter inesperado: {codigo_entrada[pos]}")
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.balance_parentesis = 0  # Contador de parentesis abiertos

    def peek(self):
        """Obtener el token actual sin consumirlo."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, tipo_esperado):
        """Consumir el token actual si coincide con el tipo esperado."""
        token = self.peek()
        if token and token[0] == tipo_esperado:
            if tipo_esperado == "PAREN_IZQ":
                self.balance_parentesis += 1
            elif tipo_esperado == "PAREN_DER":
                self.balance_parentesis -= 1
                if self.balance_parentesis < 0:
                    raise SyntaxError("Parentesis de cierre inesperado.")
            self.pos += 1
            return token
        raise SyntaxError(f"Se esperaba {tipo_esperado}, pero se encontro {token}")

    def parse_proposicion(self):
        """Regla: Proposicion -> LETRA | ( Expresion ) | ¬ Proposicion"""
        token = self.peek()

        if token[0] == 'LETRA':
            return self.consume('LETRA')
        elif token[0] == 'PAREN_IZQ':
            self.consume('PAREN_IZQ')
            self.parse_expresion()
            self.consume('PAREN_DER')
        elif token[0] == 'NEGACION':
            self.consume('NEGACION')
            self.parse_proposicion()
        else:
            raise SyntaxError(f"Proposicion invalida: {token}")

    def parse_expresion(self):
        """Regla: Expresion -> Proposicion operador Proposicion"""
        self.parse_proposicion()

        while True:
            token = self.peek()
            # Validar si el token actual es un operador esperado
            if token and token[0] in {'Y', 'O', 'IMPLICA', 'SIYSOLOSI'}:
                self.consume(token[0])  # Consumir el operador
                self.parse_proposicion()
            elif token and token[0] not in {'PAREN_DER', None}:  # Caso donde falta un operador
                raise SyntaxError(f"Se esperaba un operador entre proposiciones, pero se encontro {token[1]}")
            else:
                break

    def validar_final(self):
        """Verificar que no queden tokens sin procesar."""
        if self.pos < len(self.tokens):
            raise SyntaxError("Tokens adicionales al final de la proposicion.")


def validar_proposicion(codigo_entrada):
    try:
        tokens = lexer(codigo_entrada)
        parser = Parser(tokens)
        parser.parse_expresion()

        # Verificar si hay parentesis abiertos sin cerrar
        if parser.balance_parentesis != 0:
            raise SyntaxError("Faltan parentesis de cierre.")

        # Verificar que no haya tokens adicionales
        parser.validar_final()

        print("¡La proposicion es valida!")
    except SyntaxError as e:
        print(f"Error de sintaxis: {e}")


if __name__ == "__main__":
    print("Ingrese una proposicion logica para validar (use '¬', '∧', '∨', '⇾', '⇿', '(', ')'):")
    proposicion = input().strip()
    validar_proposicion(proposicion)