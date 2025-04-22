# DNA base encoding for 2-bit binary pairs (Table 3 in paper: A=01, C=00, G=11, T=10)&#8203;:contentReference[oaicite:25]{index=25}
def binary_to_dna(bin_str):
    dna_str = ""
    for i in range(0, len(bin_str), 2):
        pair = bin_str[i:i+2]
        if pair == '00':
            dna_str += 'C'
        elif pair == '01':
            dna_str += 'A'
        elif pair == '10':
            dna_str += 'T'
        else:
            # '11' or single '1' (if odd length) -> encode as 'T'
            dna_str += 'G'
    return dna_str

# Codon (3-base) to amino acid letter mapping (custom 26-letter code as per paper's scheme)
aminodict = {
    "GCU": "A","GCC": "A","GCA": "A","GCG": "A",      # Alanine (A) - example grouping
    "UAA": "B","UGA": "B","UAG": "B",                # 'B' represents stop codons here (paper's custom mapping)
    "UGU": "C","UGC": "C",                          # Cysteine -> 'C'
    "GAU": "D","GAC": "D",                          # Aspartic acid -> 'D'
    "GAA": "E","GAG": "E",                          # Glutamic acid -> 'E'
    "UUU": "F","UUC": "F",                          # Phenylalanine -> 'F'
    "GGU": "G","GGC": "G","GGA": "G","GGG": "G",      # Glycine -> 'G'
    "CAU": "H","CAC": "H",                          # Histidine -> 'H'
    "AUU": "I","AUC": "I","AUA": "I",                # Isoleucine -> 'I'
    "AAA": "K","AAG": "K",                          # Lysine -> 'K'
    "CUU": "L","CUC": "L","CUA": "L","CUG": "L",      # Leucine -> 'L'
    "AUG": "M",                                    # Methionine (start) -> 'M'
    "AAU": "N","AAC": "N",                          # Asparagine -> 'N'
    "UUA": "O","UUG": "O",                          # 'O' representing Leu (alternative) in custom mapping
    "CCU": "P","CCC": "P","CCA": "P","CCG": "P",      # Proline -> 'P'
    "CAA": "Q","CAG": "Q",                          # Glutamine -> 'Q'
    "CGU": "R","CGC": "R","CGA": "R","CGG": "R",      # Arginine -> 'R'
    "UCU": "S","UCC": "S","UCA": "S","UCG": "S",      # Serine -> 'S'
    "ACU": "T","ACC": "T","ACA": "T","ACG": "T",      # Threonine -> 'T'
    "AGA": "U","AGG": "U",                          # 'U' representing Arginine (AGA/AGG) in custom mapping
    "GUU": "V","GUC": "V","GUA": "V","GUG": "V",      # Valine -> 'V'
    "UGG": "W",                                    # Tryptophan -> 'W'
    "AGU": "X","AGC": "X",                          # 'X' representing Serine (AGU/AGC) in custom mapping
    "UAU": "Y","UAC": "Y"                           # Tyrosine -> 'Y'
    # Note: 'B', 'O', 'U', 'X' are used to represent certain codons to expand to 26-letter alphabet.
}

def dna_to_amino(rna_str):
    """Convert an RNA sequence (string of A,C,G,U) into an amino acid string using the mapping above."""
    amino_str = ""
    # Ensure length is a multiple of 3 for full codons
    for i in range(0, len(rna_str) - (len(rna_str) % 3), 3):
        codon = rna_str[i:i+3]
        if codon in aminodict:
            amino_str += aminodict[codon]
    return amino_str

def worint(wor):
	final=[]
	z=0
	if len(wor)>=3:
		for index in range(0,len(wor)-len(wor)%3,3):
			w=dna_to_amino(wor[index:index+3])
			z+=ord(w)
		z+=ord(wor[len(wor)-1])

	elif len(wor)<3:
		for index in range(0,len(wor)):
			z+=ord(wor[index])
	return(z)

def transform_feature_value(val):
    """
    Apply DNA and amino acid encoding to a single feature value.
    Returns a positive integer representing the transformed feature.
    """
    # 1. Get binary representation of the absolute integer value
    val_int = int(abs(int(float(val))))  # ensure integer type
    # print("step1",val_int)
    bin_str = format(val_int, 'b')
    # print("step2",bin_str)
    # 2. Binary to DNA encoding (2 bits -> 1 base)
    dna_str = binary_to_dna(bin_str)
    # print("step3",dna_str)
    # 3. DNA (with T) to RNA (replace T with U)
    rna_str = dna_str.replace('T', 'U')
    # print("step4",rna_str)
    # 4. RNA to amino acid string
    amino_str = dna_to_amino(rna_str)
    # print("step5",amino_str)
    # 5. Convert amino acid string to a positive integer
    #    Sum the ASCII codes of all amino acid characters and leftover RNA bases.
    total = worint(rna_str)
    # total = 0
    # for ch in amino_str:
    #     total += ord(ch)
    # # Add any leftover bases (if length of RNA not a multiple of 3)
    # converted_len = len(rna_str) - (len(rna_str) % 3)
    # for ch in rna_str[converted_len:]:
    #     total += ord(ch)
    return total

# Quick demonstration of the transformation on a single value
example_val = 53.627  # example feature value
transformed_val = transform_feature_value(example_val)
print(f"Original value: {example_val} -> Transformed (integer) value: {transformed_val}")
