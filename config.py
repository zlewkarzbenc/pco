# config.py

data_dir = "data"


train_ratio = 0.8
cal_ratio = 0.2
seed = 42


# Select with features to include in training
feature_groups = {
    'Diso': 1,
    'FER_RSA': 1,
    'SS3_Mono': 1,
    'SS3_Di': 1,
    'SS3_Tri': 1,
    'SS8_Mono': 1,
    'SS8_Di': 1,
    'SS8_Tri': 1,
    'Turn': 1,
    'Global': 1,
    'AA_Mono': 1,
    'AA_Di': 1,
    'AA_Tri': 1
}





# EVERYTHING BELOW IS FOR FEATURE MAINTAINANCE
# =====================================================================================

# helper strings
AAs = 'ACDEFGHIKLMNPQRSTVWY'
q3 = 'CHE'
q8 = 'CHTSEGIB'

# Groups of features
Diso = ["Fraction of sequence in disordered", 
        "Number of disordered regions", 
        "Number of disordered regions shorter than 5 AA", 
        "Number of disordered regions shorter between 5 and 10 AA",
        "Number of disordered regions longer than 10 AA"] +\
        [f"{aa} frequency in disordered" for aa in AAs]

FER_at_RSA_cutoffs = [item for tuple in [(f"FER_at_RSA_cutoff {cutoff/100}", f"FER_at_RSA_cutoff {cutoff/100} X HP") for cutoff in range(0, 100, 5)] for item in tuple]

# SS3 counters
Mono_SS3 = ["Mono_SS3: " + ss1 for ss1 in q3] 
Di_SS3 = ["Di_SS3: " + ss1 + ss2 for ss1 in q3 for ss2 in q3]
Tri_SS3 = ["Tri_SS3: " + ss1 + ss2 + ss3 for ss1 in q3 for ss2 in q3 for ss3 in q3]

# SS8 counters
Mono_SS8 = ["Mono_SS8: " + ss1 for ss1 in q8] 
Di_SS8 = ["Di_SS8: " + ss1 + ss2 for ss1 in q8 for ss2 in q8]
Tri_SS8 = ["Tri_SS8: " + ss1 + ss2 + ss3 for ss1 in q8 for ss2 in q8 for ss3 in q8]

# other features
Turn_freq = ["Turn Freq"]
Global_feats = ["log(L)", "Log(MW)", "Gravy", "Aliphatic index", "Total Charge"]

# Amino-acid counters
Mono_AA = [aa1 for aa1 in AAs]
Di_AA = [aa1 + aa2 for aa1 in AAs for aa2 in AAs]
Tri_AA = [aa1 + aa2 + aa3 for aa1 in AAs for aa2 in AAs for aa3 in AAs]

def get_feature_names():
    features = []
    if feature_groups['Diso']: features.extend(Diso)
    if feature_groups['FER_RSA']: features.extend(FER_at_RSA_cutoffs)
    if feature_groups['SS3_Mono']: features.extend(Mono_SS3)
    if feature_groups['SS3_Di']: features.extend(Di_SS3)
    if feature_groups['SS3_Tri']: features.extend(Tri_SS3)
    if feature_groups['SS8_Mono']: features.extend(Mono_SS8)
    if feature_groups['SS8_Di']: features.extend(Di_SS8)
    if feature_groups['SS8_Tri']: features.extend(Tri_SS8)
    if feature_groups['Turn']: features.extend(Turn_freq)
    if feature_groups['Global']: features.extend(Global_feats)
    if feature_groups['AA_Mono']: features.extend(Mono_AA)
    if feature_groups['AA_Di']: features.extend(Di_AA)
    if feature_groups['AA_Tri']: features.extend(Tri_AA)
    return features

feature_names = get_feature_names()
