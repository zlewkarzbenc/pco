import pandas as pd
import numpy as np
import argparse

# Hydropathicity based on https://web.expasy.org/protscale/pscale/Hphob.Doolittle.html
HP_dict = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,
           'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,
           'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}

# Molar mass based on standard amino acid molecular weights (g/mol)
MW_dict = {'A':89.1,'C':121.2,'D':133.1,'E':147.1,'F':165.2,'G':75.1,'H':155.2,
           'I':131.2,'K':146.2,'L':131.2,'M':149.2,'N':132.1,'P':115.1,'Q':146.2,
           'R':174.2,'S':105.1,'T':119.1,'V':117.1,'W':204.2,'Y':181.2}


# Pre-compute feature names based on the registered functions

# helper strings
AAs = 'ACDEFGHIKLMNPQRSTVWY'
q3 = 'CHE'
q8 = 'CHTSEGIB'

# Groups of features to be added to header
Diso = ["Fraction of sequence in disordered", 
        "Number of disordered regions", 
        "Number of disordered regions shorter than 5 AA", 
        "Number of disordered regions shorter between 5 and 10 AA",
        "Number of disordered regions longer than 10 AA"] + [f"{aa} frequency in disordered" for aa in AAs]


SS3_counters = ["Mono_SS3: " + ss1 for ss1 in q3] + ["Di_SS3: " + ss1 + ss2 for ss1 in q3 for ss2 in q3] + ["Tri_SS3: " + ss1 + ss2 + ss3 for ss1 in q3 for ss2 in q3 for ss3 in q3]

SS8_counters = ["Mono_SS8: " + ss1 for ss1 in q8] + ["Di_SS8: " + ss1 + ss2 for ss1 in q8 for ss2 in q8] + ["Tri_SS8: " + ss1 + ss2 + ss3 for ss1 in q8 for ss2 in q8 for ss3 in q8]
Turn_freq = ["Turn Freq"]

AA_counters = [aa1 for aa1 in AAs] + [aa1 + aa2 for aa1 in AAs for aa2 in AAs] + [aa1 + aa2 + aa3 for aa1 in AAs for aa2 in AAs for aa3 in AAs]
Global_feats = ["log(L)", "Log(MW)", "Gravy", "Aliphatic index", "Total Charge"]

FER_at_RSA_cutoffs = [item for tuple in [(f"FER_at_RSA_cutoff {cutoff/100}", f"FER_at_RSA_cutoff {cutoff/100} X HP") for cutoff in range(0, 100, 5)] for item in tuple]






FEATURE_NAMES = Diso + FER_at_RSA_cutoffs + SS3_counters + SS8_counters + Turn_freq +  Global_feats + AA_counters

###
# FEATURE FUNCTIONS
###
def compute_group_diso(sequence_df, threshold=0.5):
    """Diso = ["Fraction of sequence in disordered", 
        "Number of disordered regions", 
        "Number of disordered regions shorter than 5 AA", 
        "Number of disordered regions shorter between 5 and 10 AA",
        "Number of disordered regions longer than 10 AA"] + 
        [f"{aa} frequency in disordered" for aa in AAs]
    """

    sequence = list(sequence_df['seq'])
    disorder_scores = list(sequence_df['disorder'])

    aa_counters = [aa1 for aa1 in AAs]
    aa_counts = {counter: 0 for counter in aa_counters}

    bool_vector = [1 if i>threshold else 0 for i in disorder_scores]

    # count the number of each aminoacid in the cumulative disordered region
    for (res, bool) in zip(sequence, bool_vector):
        aa_counts[res] += bool


    # Initialize counters for disordered regions
    regions = []
    current_region = 0
    
    # Find continuous regions of ones
    for val in bool_vector:
        if val == 1:
            current_region += 1
        elif current_region > 0:
            regions.append(current_region)
            current_region = 0
    
    # Don't forget the last region if it ends with 1
    if current_region > 0:
        regions.append(current_region)

    # Count regions by length
    num_regions = len(regions)
    num_short = sum(1 for r in regions if r < 5)
    num_medium = sum(1 for r in regions if 5 <= r <= 10)
    num_long = sum(1 for r in regions if r > 10)


    # Generate output:
    
    other_feats = [
        sum(bool_vector)/len(sequence),  # fraction disordered
        num_regions,                     # total number of regions
        num_short,                       # regions shorter than 5
        num_medium,                      # regions between 5 and 10
        num_long                         # regions longer than 10
    ]

    # change counts to frequencies by dividing by cumulative disorder region length (sum(bool_vetor))
    if sum(bool_vector) == 0:
        ordered_aa_freqs = [0 for counter in aa_counters]
    else:
        ordered_aa_freqs = [aa_counts[counter]/sum(bool_vector) for counter in aa_counters]

    return other_feats + ordered_aa_freqs


def compute_group_FER(sequence_df):

    def FER_at_cutoff(rsa: list, sequence: list, cutoff: float):
        """returns both FER and FER times hydropathy for a given threshold"""
        bool_vector = [1 if float(i) > cutoff else 0 for i in rsa]
        fer = sum(bool_vector)/len(sequence)

        cumulative_exposed_hp = 0
        for (res, bool) in zip(sequence, bool_vector):
            cumulative_exposed_hp += HP_dict[res]*bool

        fer_x_hp = cumulative_exposed_hp*fer/len(sequence)
        return [fer, fer_x_hp]
    
    sequence_df['rsa'] = sequence_df['rsa'].apply(lambda x: float(str(x).strip()))

    sequence = list(sequence_df['seq'])
    rsa = sequence_df['rsa']
    FER_feats = []
    for cutoff in range(0,100,5):
        cutoff = cutoff/100
        FER_feats += FER_at_cutoff(rsa, sequence, cutoff)
    return FER_feats


def compute_group_SS3(sequence_df):
    q3 = 'CHE'
    sequence = list(sequence_df['q3'])
    counters = [ss1 for ss1 in q3] + [ss1 + ss2 for ss1 in q3 for ss2 in q3] + [ss1 + ss2 + ss3 for ss1 in q3 for ss2 in q3 for ss3 in q3]
    counts = {counter: 0 for counter in counters}
    idx = 0
    for char in sequence:
        # count 1-meres
        counts[char] += 1
        
        # count 2-meres
        if idx > 0:
            di_mere = sequence[idx-1] + char
            counts[di_mere] += 1
            
        # count 3-meres
        if idx > 1:
            tri_mere = sequence[idx-2] + sequence[idx-1] + char
            counts[tri_mere] += 1
        
        idx += 1

    ordered_counts = [counts[counter] for counter in counters]
    return ordered_counts

def compute_group_SS8(sequence_df):
    """ALSO RETURNS TURN FREQUENCY!!"""
        
    q8 = 'CHTSEGIB'
    sequence = list(sequence_df['q8'])
    counters = [ss1 for ss1 in q8] + [ss1 + ss2 for ss1 in q8 for ss2 in q8] + [ss1 + ss2 + ss3 for ss1 in q8 for ss2 in q8 for ss3 in q8]
    counts = {counter: 0 for counter in counters}
    idx = 0
    for char in sequence:
        # count 1-meres
        counts[char] += 1
        
        # count 2-meres
        if idx > 0:
            di_mere = sequence[idx-1] + char
            counts[di_mere] += 1
            
        # count 3-meres
        if idx > 1:
            tri_mere = sequence[idx-2] + sequence[idx-1] + char
            counts[tri_mere] += 1
        
        idx += 1

    ordered_counts = [counts[counter] for counter in counters]
    turn_freq = counts['T']/len(sequence)
    return ordered_counts + [turn_freq]
    

def global_feats_and_AA_counters(sequence_df):

    def logMW(counts: dict):
        mw = 0
        for aa in AAs:
            mw += counts[aa]*MW_dict[aa] 
        return np.log(mw)

    def gravy(sequence):
        return sum([HP_dict[i] for i in sequence])/len(sequence)
    
    def aliphatic_index(counts: dict, length, a = 2.9, b = 3.9):
        # compute mole fraction of Alanine, Valine, Isoleucie and Leucine
        A, V, I, L = counts['A'], counts['V'], counts['I'], counts['L']
        X_a, X_v, X_i, X_l = A/length, V/length, I/length, L/length

        # multiply by empirical coefficients
        return X_a + a*X_v + b*(X_i + X_l)

    def total_charge(counts: dict):
        num_basic = counts['K'] + counts['R']
        num_acidic = counts['D'] + counts['E']

        return num_basic - num_acidic
    

    #AAs = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = list(sequence_df['seq'])
    counters = [aa1 for aa1 in AAs] + [aa1 + aa2 for aa1 in AAs for aa2 in AAs] + [aa1 + aa2 + aa3 for aa1 in AAs for aa2 in AAs for aa3 in AAs]
    counts = {counter: 0 for counter in counters}
    idx = 0
    for char in sequence:
        # count 1-meres
        counts[char] += 1
        
        # count 2-meres
        if idx > 0:
            di_mere = sequence[idx-1] + char
            counts[di_mere] += 1
            
        # count 3-meres
        if idx > 1:
            tri_mere = sequence[idx-2] + sequence[idx-1] + char
            counts[tri_mere] += 1
        
        idx += 1

# ["log(L)", "Log(MW)", "Gravy", "Aliphatic index", "Total Charge"]
    global_feats = [np.log(len(sequence)), logMW(counts), gravy(sequence), aliphatic_index(counts, len(sequence)), total_charge(counts)]
    ordered_counts = [counts[counter] for counter in counters]
    return global_feats + ordered_counts



# Register feature functions in a fixed order to ensure consistency
FEATURE_FUNCTIONS = [
    compute_group_diso,
    compute_group_FER,
    compute_group_SS3,
    compute_group_SS8,
    global_feats_and_AA_counters
]

###
# HIGHER-ORDER FUNCTIONS
###
def compute_features(sequence_df):
    """Compute features for a single sequence."""

    features = []
    for func in FEATURE_FUNCTIONS:
        features.extend(func(sequence_df))
    return features

def process_sequences(input_path, output_path):
    """Process the input CSV and compute features."""
    # Read the input CSV in chunks to handle large files
    chunksize = 100000  # Adjust based on memory availability
    columns = ['id', 'seq', 'n', 'rsa', 'q3', 'q8', 'disorder']
    output_rows = []

    current_id = None
    sequence_df = pd.DataFrame(columns=columns)

    for chunk in pd.read_csv(
            input_path,
            chunksize=chunksize,
            skiprows=0,            # skip the old header
            header=0,           # no header in the file now
            skipinitialspace=True,
    ):
        # Select only the required columns
        chunk = chunk[columns]  # This will ignore any extra columns in the input CSV

        for _, row in chunk.iterrows():
            fasta_id = row['id']

            # On new sequence, flush previous
            if fasta_id != current_id:
                if current_id is not None:
                    features = compute_features(sequence_df)
                    output_rows.append([current_id] + features)
                current_id = fasta_id
                sequence_df = pd.DataFrame(columns=columns)

            # Append the row to the current sequence DataFrame
            if sequence_df.empty:
                sequence_df = pd.DataFrame([row])
            else:
                sequence_df = pd.concat([sequence_df, pd.DataFrame([row])], ignore_index=True)


    # Flush the last sequence
    if current_id is not None:
        features = compute_features(sequence_df)
        output_rows.append([current_id] + features)

    # Write the output CSV
    output_df = pd.DataFrame(output_rows, columns=['id'] + FEATURE_NAMES)
    output_df.to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='Process protein sequences CSV to compute features.')
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Path to output CSV file')
    args = parser.parse_args()

    process_sequences(args.input, args.output)

if __name__ == '__main__':
    main()