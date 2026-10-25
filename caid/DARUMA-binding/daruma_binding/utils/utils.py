

def read_single_fasta(fasta_file):
    """
    Reads a single sequence from a FASTA file and returns the accession and sequence.
    """
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        if not lines or not lines[0].startswith('>'):
            raise ValueError("Invalid FASTA format: missing header line starting with '>'")
        
        acc = lines[0].strip()[1:]
        seq = ''.join(line.strip() for line in lines[1:])
    return acc, seq

