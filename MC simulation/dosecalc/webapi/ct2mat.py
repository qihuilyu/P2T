import numpy as np
from scipy.interpolate import interp1d


class G4Material():
    def __init__(self, label='', density=0.0, matid=0):
        self.label = label
        self.density = density
        self.matid = matid

    def __repr__(self):
        return '<Material: "{!s}", {:d}, {:f}>'.format(self.label, self.matid, self.density)

    def get_material_def(self):
        """returns entry suitable for voxel description in geometry.text file (format: [dens, matid] )"""
        return "{:0.3f} {:d}\n".format(self.density, self.matid)

mat_map = {
    "air":              G4Material('Air',               density=0.00129,   matid=0),
    "lung_in":          G4Material('Lungs (inhale)',    density=0.217,     matid=1),
    "lung_ex":          G4Material('Lungs (exhale)',    density=0.508,     matid=2),
    "adipose":          G4Material('Adipose',           density=0.967,     matid=3),
    "breast":           G4Material('Breast',            density=0.99,      matid=4),
    "water":            G4Material('Water',             density=1.0,       matid=5),
    "muscle":           G4Material('Muscle',            density=1.061,     matid=6),
    "liver":            G4Material('Liver',             density=1.071,     matid=7),
    "bone_trab":        G4Material('Bone (trabecular)', density=1.159,     matid=8),
    "bone_comp":        G4Material('Bone (compact)',    density=1.575,     matid=9),
    "Io05":             G4Material('Io05',              density=1.0415,    matid=10),
    "Ba05":             G4Material('Ba05',              density=1.0405,    matid=11),
    "Gd05":             G4Material('Gd05',              density=1.0457,    matid=12),
    "Yb05":             G4Material('Yb05',              density=1.0447,    matid=13),
    "Ta05":             G4Material('Ta05',              density=1.0493,    matid=14),
    "Au05":             G4Material('Au05',              density=1.0498,    matid=15),
    "Bi05":             G4Material('Bi05',              density=1.0470,    matid=16),    
}

lut_ct2dens = [
    (-5000.0, 0.0),
    (-1000.0, 0.01),
    (-400,    0.602),
    (-150,    0.924),
    (100,     1.075),
    (300,     1.145),
    (2000,    1.856),
    (4927,    3.379),
    (66000,   7.8),
]
f_ct2dens = None

lut_dens2mat = [
    (0.0,   mat_map["air"]      ),
    (0.207, mat_map["lung_in"]  ),
    (0.481, mat_map["lung_ex"]  ),
    (0.919, mat_map["adipose"]  ),
    (0.979, mat_map["breast"]   ),
    (1.004, mat_map["water"]    ),
    (1.109, mat_map["muscle"]   ),
    (1.113, mat_map["liver"]    ),
    (1.496, mat_map["bone_trab"]),
    (1.654, mat_map["bone_comp"]),
    (6.0,   mat_map["Io05"]),
    (6.1,   mat_map["Ba05"]),
    (6.2,   mat_map["Gd05"]),
    (6.3,   mat_map["Yb05"]),
    (6.4,   mat_map["Ta05"]),
    (6.5,   mat_map["Au05"]),
    (6.6,   mat_map["Bi05"]),
]
f_dens2matindex = None

def init_lut_interpolators():
    global f_ct2dens, f_dens2matindex
    if f_ct2dens is None:
        lut_ct, lut_dens = zip(*lut_ct2dens)
        f_ct2dens = interp1d(lut_ct, lut_dens, kind='linear', bounds_error=False, fill_value=(np.min(lut_dens), np.max(lut_dens)))

    if f_dens2matindex is None:
        lut_dens, mat_list = zip(*lut_dens2mat)
        f_dens2matindex = interp1d(lut_dens, range(len(mat_list)), kind='previous', bounds_error=False, fill_value=(0, len(mat_list)-1))

def f_dens2mat(densities):
    """use cached interpolator to convert densities array to object array of corresponding materials objects"""
    init_lut_interpolators()
    matindices = f_dens2matindex(densities).astype(int)
    _, mat_list = zip(*lut_dens2mat)
    material_choices = np.array(mat_list, dtype=object)
    materials = material_choices[matindices]
    return materials



def lookup_materials(ctnums=None, densities=None, bulk_density=True):
    """convert either an array of ctnums or an array of densities to an array (of type=np.array(dtype=object)) of material specs
    Promises to be much faster than the per-element version (ct2mat)

    UNTESTED
    """
    assert any(param is not None for param in (ctnums, densities))
    init_lut_interpolators()
    if ctnums is not None:
        densities = f_ct2dens(ctnums)
    print(densities.max())

    materials = f_dens2mat(densities)
    return materials



def convert_ct_to_density(ctvol):
    """convert ctvol (array-like) of ct intensities (HU) to array-like of material densities based on
    standard lookup table"""
    init_lut_interpolators()
    density = f_ct2dens(ctvol)
    return density


