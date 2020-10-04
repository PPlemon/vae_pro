import base64


base64_dictionary = {
                  'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                  'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                  'X': 23, 'Y': 24, 'Z': 25,
                  'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33, 'i': 34, 'j': 35, 'k': 36,
                  'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44, 't': 45, 'u': 46, 'v': 47,
                  'w': 48, 'x': 49, 'y': 50, 'z': 51,
                  '0': 52, '1': 53, '2': 54, '3': 55, '4': 56, '5': 57, '6': 58, '7': 59, '8': 60, '9': 61, '+': 62,
                  '/': 63}
base64_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9', '+', '/']
base32_charset = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7']
def basevector(smiles):
    num = []
    smiles = smiles.ljust(60)
    compressed = base64.b64encode(smiles.encode())
    for c in compressed:
        i = base64_dictionary[chr(c)]/64
        num.append(i)
    return num

def base64encoder(smiles):
    return base64.b64encode(smiles.encode())

def base32encoder(smiles):
    return base64.b32encode(smiles.encode())

def base64_vector(smiles):
    smiles_vector = []
    smiles = smiles.replace('\n', '')
    compressed = base64.b64encode(smiles.encode())
    # compressed = compressed.ljust(150)
    for c in compressed:
        charset_vector = [0] * 64
        for index, value in enumerate(base64_charset):
            if chr(c) == value:
                charset_vector[index] = 1
        smiles_vector.append(charset_vector)
    return smiles_vector
def base32_vector(smiles):
    smiles_vector = []
    smiles = smiles.replace('\n', '')
    compressed = base64.b32encode(smiles.encode())
    # compressed = compressed.ljust(192)
    for c in compressed:
        charset_vector = [0] * 33
        for index, value in enumerate(base32_charset):
            if chr(c) == value:
                charset_vector[index] = 1
        smiles_vector.append(charset_vector)
    return smiles_vector

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()
def vector1(smiles, charset):
    smiles_vector = []
    smiles = smiles.replace('\n', '')
    smiles = smiles.ljust(150)
    for c in smiles:
        charset_vector = [0] * len(charset)
        for index, value in enumerate(charset):
            if c == value:
                charset_vector[index] = 1
        smiles_vector.append(charset_vector)
    return smiles_vector

def index_vector(smiles, charset):
    smiles_vector = []
    smiles = smiles.replace('\n', '')
    smiles = smiles.ljust(120)
    for c in smiles:
        charset_vector = [0] * len(charset)
        for index, value in enumerate(charset):
            if c == value:
                charset_vector[index] = 1
        smiles_vector.append(charset_vector)
    return smiles_vector