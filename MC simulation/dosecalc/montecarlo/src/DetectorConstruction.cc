#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"

#include "G4Element.hh"
#include "G4Material.hh"

//Solid volumes (shapes)
#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4Trd.hh"

//Logical and Physical volumes (materials, placement)
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"

//Boolean Solids (union, subtraction, intersection etc.)
#include "G4UnionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4IntersectionSolid.hh"

//PVReplica for voxelization,
#include "G4PVReplica.hh"

//G4PVParameterised and nestedparam for CT materials
#include "G4PVParameterised.hh"
#include "NestedParam.hh"

//For Scoring
#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4PSDoseDeposit3D.hh"
#include "PositronAnnihilation3D.hh"
#include "G4SDParticleFilter.hh"
#include "G4PSPassageCellCurrent.hh"
#include "G4UserParticleWithDirectionFilter.hh"
#include "G4PSEnergyDeposit.hh"

//Quality of Life includes
#include "G4NistManager.hh"
#include "G4UnitsTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4RunManager.hh"
#include "G4GlobalMagFieldMessenger.hh"
// #include "G4TransportationManager.hh"
// #include "G4FieldManager.hh"
// #include "G4UniformMagField.hh"
// #include "G4ChordFinder.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <exception>

extern G4String g_geoFname;
extern G4String g_outputdir;
extern bool g_output_positronannihilation;

using namespace std;
DetectorConstruction* DetectorConstruction::instance = 0;
DetectorConstruction* DetectorConstruction::getInstance()
{
    if (instance == 0) instance = new DetectorConstruction();
    return instance;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
DetectorConstruction::DetectorConstruction()
{
    dMess = new DetectorMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction()
{
    delete dMess;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4cout << "Entering DetectorConstruction::Construct()" << G4endl;

    // create base materials from elemental mixtures
    InitialisationOfMaterials();
    // read (density, material_id) pairs from geometry file and store in vector
    ReadPhantom();

    {
        // Create world volume
        G4double wx, wy, wz;
        wx = 300*cm;
        wy = 300*cm;
        wz = 300*cm;

        //Note that manager and air water ... are global to the class
        man = G4NistManager::Instance();
        G4Air = man->FindOrBuildMaterial("G4_AIR");

        G4Box *sWorld = new G4Box("world", wx / 2., wy / 2., wz / 2.);
        lWorld = new G4LogicalVolume(sWorld, G4Air, "World");
    }
    G4VPhysicalVolume *pWorld = new G4PVPlacement(0, G4ThreeVector(), lWorld, "World", 0, false, 0);

    // make unique list of material instances, map each voxel to one instance
    MapMaterials();
    // Create voxelized geometry using mapped material instances
    CreatePhantom();
    // save density array for verification
    SanityCheck();
    // Create ring detector to measure photons from positron annihilation
    CreateRingDetector();

    return pWorld;
}

void DetectorConstruction::ReadPhantom() {
    G4String line;
    std::stringstream ss;

    std::ifstream infile;
    infile.open(g_geoFname);

    if (!infile.is_open()){
        G4cerr << "Failed opening Geometry" << G4endl;
        exit(1);
    }

    // read header
    infile >> nx >> ny >> nz; // nvoxels
    infile >> dx >> dy >> dz; // voxelsize (mm)
    infile >> px >> py >> pz; // position of array center (mm)
    nxyz = nx*ny*nz;
    infile.ignore(1, '\n'); // move to next line

    //sanity check
    G4cout << "Array size: " << nx <<" "<< ny << " " << nz << G4endl <<
              "Voxel size (mm): "<< dx << " " << dy << " " << dz << G4endl <<
              "Center Position (mm): " << px << " " << py << " " << pz << G4endl;

    ss.str("");
    ss.clear();

    // read material specifications
    while (getline(infile, line)){
        if (line.length() < 1) { //last line is empty
            infile.close();
            break;
        }
        ss.str(line);
        matStruct temp;
        ss >> temp.den >> temp.matID;

        // G4cout << "density: " << temp.den << " g/cm^3 | matid:  " << temp.matID << G4endl;
        temp.den *= g/cm3;
        raw_matspecs.push_back(temp);
        ss.str("");
        ss.clear();
    }
    if((G4long)raw_matspecs.size()!=nxyz) { //number of voxel mismatch
        G4cerr << "mismatch between nxyz in header and number of lines" << G4endl;
        throw runtime_error("mismatch between nxyz in header and number of lines in file");
    }
}

void DetectorConstruction::MapMaterials() {
    // //Create, Sort and prune matspec list to contain only unique (matid, density) pairs to prevent multiple material redefinition
    // unique_matspecs = std::list<matStruct>(raw_matspecs.cbegin(), raw_matspecs.cend()); // copy
    // G4cout << "# materials before pruning: " << unique_matspecs.size() << G4endl;
    // unique_matspecs.sort();
    // unique_matspecs.unique();
    // G4cout << "# materials after pruning: " << unique_matspecs.size() << G4endl;

    // //Migrate list to vector for random access
    // std::vector<matStruct>vunique_matspecs.resize(unique_matspecs.size());
    // std::copy(vunique_matspecs.begin(), vunique_matspecs.end(), unique_matspecs.begin());


    //Go through all voxels, find corresponding density and map voxel to that unique density
    G4cout << "Assigning material instance to each voxel" << G4endl;
    G4long idx;
    matMap.resize(raw_matspecs.size());
    for (uint i = 0; i < raw_matspecs.size(); i++) {
        // check if valid material instance already exists
        // for (idx = 0; idx < vunique_matspecs.size(); idx++) {
        //     //match current voxel i's density with a unique density index idx
        //     if (raw_matspecs[i] == vunique_matspecs[idx]) {
        //         //assign this mapping to the material
        //         matMap.push_back(idx);

        //         //create the material if it doesn't already exist
        //         CreateMaterial(i, idx);
        //         break;
            // }
            //

        // }
        // assign same density to all voxels of the same material
        matMap[i] = raw_matspecs[i].matID;
    }
}

void DetectorConstruction::InitialisationOfMaterials()
{
    // Creating elements :
    G4double z, a, density;
    G4String name, symbol;

    G4Element* elC = new G4Element( name = "Carbon",
                                   symbol = "C",
                                   z = 6.0, a = 12.011 * g/mole );
    G4Element* elH = new G4Element( name = "Hydrogen",
                                   symbol = "H",
                                   z = 1.0, a = 1.008  * g/mole );
    G4Element* elN = new G4Element( name = "Nitrogen",
                                   symbol = "N",
                                   z = 7.0, a = 14.007 * g/mole );
    G4Element* elO = new G4Element( name = "Oxygen",
                                   symbol = "O",
                                   z = 8.0, a = 16.00  * g/mole );
    G4Element* elNa = new G4Element( name = "Sodium",
                                    symbol = "Na",
                                    z= 11.0, a = 22.98977* g/mole );
    G4Element* elS = new G4Element( name = "Sulfur",
                                   symbol = "S",
                                   z = 16.0,a = 32.065* g/mole );
    G4Element* elCl = new G4Element( name = "Chlorine",
                                    symbol = "P",
                                    z = 17.0, a = 35.453* g/mole );
    G4Element* elK = new G4Element( name = "Potassium",
                                   symbol = "P",
                                   z = 19.0, a = 30.0983* g/mole );
    G4Element* elP = new G4Element( name = "Phosphorus",
                                   symbol = "P",
                                   z = 30.0, a = 30.973976* g/mole );
    G4Element* elFe = new G4Element( name = "Iron",
                                    symbol = "Fe",
                                    z = 26, a = 56.845* g/mole );
    G4Element* elMg = new G4Element( name = "Magnesium",
                                    symbol = "Mg",
                                    z = 12.0, a = 24.3050* g/mole );
    G4Element* elCa = new G4Element( name="Calcium",
                                    symbol = "Ca",
                                    z = 20.0, a = 40.078* g/mole );
    G4Element* elIo = new G4Element( name="Iodine",
                                    symbol = "Io",
                                    z = 53.0, a = 126.904* g/mole );
    G4Element* elBa = new G4Element( name="Barium",
                                    symbol = "Ba",
                                    z = 56.0, a = 137.327* g/mole );
    G4Element* elGd = new G4Element( name="Gadolinium",
                                    symbol = "Gd",
                                    z = 64.0, a = 157.25* g/mole );
    G4Element* elYb = new G4Element( name="Ytterbium",
                                    symbol = "Yb",
                                    z = 70.0, a = 173.04* g/mole );
    G4Element* elTa = new G4Element( name="Tantalum",
                                    symbol = "Ta",
                                    z = 73.0, a = 180.948* g/mole );
    G4Element* elAu = new G4Element( name="Gold",
                                    symbol = "Au",
                                    z = 79.0, a = 196.967* g/mole );
    G4Element* elBi = new G4Element( name="Bismuth",
                                    symbol = "Bi",
                                    z = 83.0, a = 208.9804* g/mole );


    // Creating Materials :
    G4int numberofElements;

    // Air
    fAir = new G4Material( "Air",
                          1.290*mg/cm3,
                          numberofElements = 2 );
    fAir->AddElement(elN, 0.7);
    fAir->AddElement(elO, 0.3);

    //  Lung Inhale
    G4Material* lunginhale = new G4Material( "LungInhale",
                                            density = 0.217*g/cm3,
                                            numberofElements = 9);
    lunginhale->AddElement(elH,0.103);
    lunginhale->AddElement(elC,0.105);
    lunginhale->AddElement(elN,0.031);
    lunginhale->AddElement(elO,0.749);
    lunginhale->AddElement(elNa,0.002);
    lunginhale->AddElement(elP,0.002);
    lunginhale->AddElement(elS,0.003);
    lunginhale->AddElement(elCl,0.002);
    lunginhale->AddElement(elK,0.003);

    // Lung exhale
    G4Material* lungexhale = new G4Material( "LungExhale",
                                            density = 0.508*g/cm3,
                                            numberofElements = 9 );
    lungexhale->AddElement(elH,0.103);
    lungexhale->AddElement(elC,0.105);
    lungexhale->AddElement(elN,0.031);
    lungexhale->AddElement(elO,0.749);
    lungexhale->AddElement(elNa,0.002);
    lungexhale->AddElement(elP,0.002);
    lungexhale->AddElement(elS,0.003);
    lungexhale->AddElement(elCl,0.002);
    lungexhale->AddElement(elK,0.003);

    // Adipose tissue
    G4Material* adiposeTissue = new G4Material( "AdiposeTissue",
                                               density = 0.967*g/cm3,
                                               numberofElements = 7);
    adiposeTissue->AddElement(elH,0.114);
    adiposeTissue->AddElement(elC,0.598);
    adiposeTissue->AddElement(elN,0.007);
    adiposeTissue->AddElement(elO,0.278);
    adiposeTissue->AddElement(elNa,0.001);
    adiposeTissue->AddElement(elS,0.001);
    adiposeTissue->AddElement(elCl,0.001);

    // Breast
    G4Material* breast = new G4Material( "Breast",
                                        density = 0.990*g/cm3,
                                        numberofElements = 8 );
    breast->AddElement(elH,0.109);
    breast->AddElement(elC,0.506);
    breast->AddElement(elN,0.023);
    breast->AddElement(elO,0.358);
    breast->AddElement(elNa,0.001);
    breast->AddElement(elP,0.001);
    breast->AddElement(elS,0.001);
    breast->AddElement(elCl,0.001);

    // Water
    G4Material* water = new G4Material( "Water",
                                       density = 1.0*g/cm3,
                                       numberofElements = 2 );
    water->AddElement(elH,0.112);
    water->AddElement(elO,0.888);

    // Muscle
    G4Material* muscle = new G4Material( "Muscle",
                                        density = 1.061*g/cm3,
                                        numberofElements = 9 );
    muscle->AddElement(elH,0.102);
    muscle->AddElement(elC,0.143);
    muscle->AddElement(elN,0.034);
    muscle->AddElement(elO,0.710);
    muscle->AddElement(elNa,0.001);
    muscle->AddElement(elP,0.002);
    muscle->AddElement(elS,0.003);
    muscle->AddElement(elCl,0.001);
    muscle->AddElement(elK,0.004);

    // Liver
    G4Material* liver = new G4Material( "Liver",
                                       density = 1.071*g/cm3,
                                       numberofElements = 9);
    liver->AddElement(elH,0.102);
    liver->AddElement(elC,0.139);
    liver->AddElement(elN,0.030);
    liver->AddElement(elO,0.716);
    liver->AddElement(elNa,0.002);
    liver->AddElement(elP,0.003);
    liver->AddElement(elS,0.003);
    liver->AddElement(elCl,0.002);
    liver->AddElement(elK,0.003);

    // Trabecular Bone
    G4Material* trabecularBone = new G4Material( "TrabecularBone",
                                                density = 1.159*g/cm3,
                                                numberofElements = 12 );
    trabecularBone->AddElement(elH,0.085);
    trabecularBone->AddElement(elC,0.404);
    trabecularBone->AddElement(elN,0.058);
    trabecularBone->AddElement(elO,0.367);
    trabecularBone->AddElement(elNa,0.001);
    trabecularBone->AddElement(elMg,0.001);
    trabecularBone->AddElement(elP,0.034);
    trabecularBone->AddElement(elS,0.002);
    trabecularBone->AddElement(elCl,0.002);
    trabecularBone->AddElement(elK,0.001);
    trabecularBone->AddElement(elCa,0.044);
    trabecularBone->AddElement(elFe,0.001);

    // Dense Bone
    G4Material* denseBone = new G4Material( "DenseBone",
                                           density = 1.575*g/cm3,
                                           numberofElements = 11 );
    denseBone->AddElement(elH,0.056);
    denseBone->AddElement(elC,0.235);
    denseBone->AddElement(elN,0.050);
    denseBone->AddElement(elO,0.434);
    denseBone->AddElement(elNa,0.001);
    denseBone->AddElement(elMg,0.001);
    denseBone->AddElement(elP,0.072);
    denseBone->AddElement(elS,0.003);
    denseBone->AddElement(elCl,0.001);
    denseBone->AddElement(elK,0.001);
    denseBone->AddElement(elCa,0.146);

/*
    G4NistManager* man2 = G4NistManager::Instance();    
    // Tumor
    G4Material* Tumor = new G4Material("Tumor", density = 1.0*g / cm3, numberofElements = 2);
    Tumor->AddElement(elH,0.112);
    Tumor->AddElement(elO,0.888);
  
    // Tumor with gold 0.5%
    G4Material* TumorWithAu005 = new G4Material("TumorWithAu005", density = 1.0048*g / cm3, numberofElements = 3);
    TumorWithAu005->AddElement(elH,0.112*0.995);
    TumorWithAu005->AddElement(elO,0.888*0.995);
    TumorWithAu005->AddElement(elAu,0.005);

    // Tumor with gold 2%
    G4Material* TumorWithAu02 = new G4Material("TumorWithAu02", density = 1.0193*g / cm3, numberofElements = 3);
    TumorWithAu02->AddElement(elH,0.112*0.98);
    TumorWithAu02->AddElement(elO,0.888*0.98);
    TumorWithAu02->AddElement(elAu,0.02);

    // Tumor with gold 5%
    G4Material* TumorWithAu05 = new G4Material("TumorWithAu05", density = 1.0498*g / cm3, numberofElements = 3);
    TumorWithAu05->AddElement(elH,0.112*0.95);
    TumorWithAu05->AddElement(elO,0.888*0.95);
    TumorWithAu05->AddElement(elAu,0.05);

    // Tumor with Calcium 0.5%
    G4Material* TumorWithCa005 = new G4Material("TumorWithCa005", density = 1.0018*g / cm3, numberofElements = 3);
    TumorWithCa005->AddElement(elH,0.112*0.995);
    TumorWithCa005->AddElement(elO,0.888*0.995);
    TumorWithCa005->AddElement(elCa,0.005);

    // Tumor with Calcium 2%
    G4Material* TumorWithCa02 = new G4Material("TumorWithCa02", density = 1.0071*g / cm3, numberofElements = 3);
    TumorWithCa02->AddElement(elH,0.112*0.98);
    TumorWithCa02->AddElement(elO,0.888*0.98);
    TumorWithCa02->AddElement(elCa,0.02);

    // Tumor with Calcium 5%
    G4Material* TumorWithCa05 = new G4Material("TumorWithCa05", density = 1.0181*g / cm3, numberofElements = 3);
    TumorWithCa05->AddElement(elH,0.112*0.95);
    TumorWithCa05->AddElement(elO,0.888*0.95);
    TumorWithCa05->AddElement(elCa,0.05);
*/


    G4NistManager* man2 = G4NistManager::Instance();    
    // Tumor
    G4Material* Io05 = new G4Material("Io05", density =  1.0415*g / cm3, numberofElements = 3);
    Io05->AddElement(elH,0.112*0.95);
    Io05->AddElement(elO,0.888*0.95);
    Io05->AddElement(elIo,0.05);
  
    // Tumor with gold 0.5%
    G4Material* Ba05 = new G4Material("Ba05", density = 1.0405*g / cm3, numberofElements = 3);
    Ba05->AddElement(elH,0.112*0.95);
    Ba05->AddElement(elO,0.888*0.95);
    Ba05->AddElement(elBa,0.05);

    // Tumor with gold 0.5%
    G4Material* Gd05 = new G4Material("Gd05", density = 1.0457*g / cm3, numberofElements = 3);
    Gd05->AddElement(elH,0.112*0.95);
    Gd05->AddElement(elO,0.888*0.95);
    Gd05->AddElement(elGd,0.05);
        
    // Tumor with gold 0.5%
    G4Material* Yb05 = new G4Material("Yb05", density = 1.0447*g / cm3, numberofElements = 3);
    Yb05->AddElement(elH,0.112*0.95);
    Yb05->AddElement(elO,0.888*0.95);
    Yb05->AddElement(elYb,0.05);

    // Tumor with gold 0.5%
    G4Material* Ta05 = new G4Material("Ta05", density = 1.0493*g / cm3, numberofElements = 3);
    Ta05->AddElement(elH,0.112*0.95);
    Ta05->AddElement(elO,0.888*0.95);
    Ta05->AddElement(elTa,0.05);

    // Tumor with gold 0.5%
    G4Material* Au05 = new G4Material("Au05", density = 1.0498*g / cm3, numberofElements = 3);
    Au05->AddElement(elH,0.112*0.95);
    Au05->AddElement(elO,0.888*0.95);
    Au05->AddElement(elAu,0.05);

    // Tumor with gold 0.5%
    G4Material* Bi05 = new G4Material("Bi05", density = 1.0470*g / cm3, numberofElements = 3);
    Bi05->AddElement(elH,0.112*0.95);
    Bi05->AddElement(elO,0.888*0.95);
    Bi05->AddElement(elBi,0.05);


    //----- Put the materials in a vector (order must match matid in input geometry file)
    fOriginalMaterials.resize(17);
    fOriginalMaterials[0] = fAir;           // rho = 0.00129
    fOriginalMaterials[1] = lunginhale;     // rho = 0.217
    fOriginalMaterials[2] = lungexhale;     // rho = 0.508
    fOriginalMaterials[3] = adiposeTissue;  // rho = 0.967
    fOriginalMaterials[4] = breast ;        // rho = 0.990
    fOriginalMaterials[5] = water;          // rho = 1.018
    fOriginalMaterials[6] = muscle;         // rho = 1.061
    fOriginalMaterials[7] = liver;          // rho = 1.071
    fOriginalMaterials[8] = trabecularBone; // rho = 1.159
    fOriginalMaterials[9] = denseBone;      // rho = 1.575
  /*
    fOriginalMaterials[10] = Tumor;      // rho = 1.575
    fOriginalMaterials[11] = TumorWithAu005;      // rho = 1.575
    fOriginalMaterials[12] = TumorWithAu02;      // rho = 1.575
    fOriginalMaterials[13] = TumorWithAu05;      // rho = 1.575
    fOriginalMaterials[14] = TumorWithCa005;      // rho = 1.575
    fOriginalMaterials[15] = TumorWithCa02;      // rho = 1.575
    fOriginalMaterials[16] = TumorWithCa05;      // rho = 1.575
      */
    fOriginalMaterials[10] = Io05;      // rho = 1.575
    fOriginalMaterials[11] = Ba05;      // rho = 1.575
    fOriginalMaterials[12] = Gd05;      // rho = 1.575
    fOriginalMaterials[13] = Yb05;      // rho = 1.575
    fOriginalMaterials[14] = Ta05;      // rho = 1.575
    fOriginalMaterials[15] = Au05;      // rho = 1.575
    fOriginalMaterials[16] = Bi05;      // rho = 1.575 
    
    
    G4bool isotopes = false;  
    G4Element*  O = man2->FindOrBuildElement("O" , isotopes); 
    G4Element* Si = man2->FindOrBuildElement("Si", isotopes);
    G4Element* Lu = man2->FindOrBuildElement("Lu", isotopes);  
   
    G4Material* LSO = new G4Material("Lu2SiO5", 7.4*g/cm3, 3);
    LSO->AddElement(Lu, 2);
    LSO->AddElement(Si, 1);
    LSO->AddElement(O , 5); 

 //   G4double A, Z;
 //   A= 207.2 *g/mole;
 //   density= 11.35 *g/cm3;
 //   G4Material* Pb= new G4Material("Lead", Z=82., A, density);


}

// void DetectorConstruction::CreateMaterial(G4long i, G4int idx) {
//     //reminder: i corresponds to specific voxel, idx corresponds to the material we want
//     if (fOriginalMaterials[idx] != 0)
//         return; //this material is already here, nothing to do.  Put this first so we jump out quick

//     //if we're here, then the material doesn't exist, time to create a material
//     G4cout << "i: " << i<< "matidx: "<<idx<< G4endl;
//     throw runtime_error("material undefined");
// }

void DetectorConstruction::CreatePhantom() {
    /////////////////From lecture 8/////////////////
    G4double boxx, boxy, boxz;
    boxx = nx*dx;
    boxy = ny*dy;
    boxz = nz*dz;

    G4Box * sBox = new G4Box("sBox", boxx / 2., boxy / 2., boxz / 2.);
    G4LogicalVolume *lBox = new G4LogicalVolume(sBox, G4Air, "lBox");
    new G4PVPlacement(0, G4ThreeVector(), lBox, "pBox", lWorld, false, 0, true);

    G4VSolid *sRepZ = new G4Box("sRepZ", boxx / 2., boxy / 2., dz / 2.);
    G4LogicalVolume *lRepZ = new G4LogicalVolume(sRepZ, G4Air, "lRepZ");
    new G4PVReplica("pRepZ", lRepZ, lBox, kZAxis, nz, dz);

    G4VSolid *sRepY = new G4Box("sRepY", boxx / 2., dy / 2., dz / 2.);
    G4LogicalVolume *lRepY = new G4LogicalVolume(sRepY, G4Air, "lRepY");
    new G4PVReplica("pRepY", lRepY, lRepZ, kYAxis, ny, dy);

    G4VSolid *sRepX = new G4Box("sRepX", dx / 2., dy / 2., dz / 2.);
    G4LogicalVolume *lRepX = new G4LogicalVolume(sRepX, G4Air, "lRepX");

    /////////////////////////Not from Lecture 8///////////

    NestedParam* param = new NestedParam(matMap, fOriginalMaterials);  //defines material mapping that overrides voxel logvol
    param->SetDimVoxel(dx, dy, dz);
    param->SetNoVoxel(nx, ny, nz);

    new G4PVParameterised("ctVox", lRepX, lRepY, kXAxis, nx, param); //a parameterised pvplacement
}

void DetectorConstruction::SanityCheck() {
    std::ofstream outfile;
    std::ostringstream fname;
    fname << g_outputdir << "/" << "InputDensity.bin";

    outfile.open(fname.str(), std::ios::out | std::ios::binary);
    for (G4long i = 0; i < (G4long)raw_matspecs.size(); i++) {
        float val = float(fOriginalMaterials[matMap[i]]->GetDensity())/(g/cm3);
        outfile.write((char*)(&val), sizeof(float));
    }
    outfile.close();
}

void DetectorConstruction::CreateRingDetector() {
   // Gamma detector Parameters
   //
    G4int nb_cryst = 1440;
    G4int nb_rings = 1;
    G4double ring_R1 = 120*cm;
    G4double ring_R2 = 125*cm;
    G4double cryst_dX = 10*cm;
   
    G4double dPhi = twopi/nb_cryst, half_dPhi = 0.5*dPhi;
    G4double cosdPhi = std::cos(half_dPhi);
    G4double tandPhi = std::tan(half_dPhi);

    G4double cryst_dY = ring_R1*tandPhi*2;    
    G4double cryst_dZ = ring_R2*cosdPhi-ring_R1;

   //
   G4double detector_dZ = nb_rings*cryst_dX;
   //
   G4NistManager* nist = G4NistManager::Instance();
   G4Material* default_mat = nist->FindOrBuildMaterial("G4_AIR");
   G4Material* cryst_mat   = nist->FindOrBuildMaterial("G4_AIR");

   G4Tubs* solidRing =
     new G4Tubs("Ring", ring_R1, ring_R2, 0.5*cryst_dX, 0., twopi);
       
   G4LogicalVolume* logicRing =                         
     new G4LogicalVolume(solidRing,           //its solid
                         default_mat,         //its material
                         "Ring");             //its name

    G4bool  fCheckOverlaps = 0;            
   //     
   // define crystal
   //
   G4double gap = 0.01*mm;        //a gap for wrapping
   G4double dX = cryst_dX - gap, dY = cryst_dY - gap;
   G4Box* solidCryst = new G4Box("crystal", dX/2, dY/2, cryst_dZ/2);
                      
   G4LogicalVolume* logicCryst = 
     new G4LogicalVolume(solidCryst,          //its solid
                         cryst_mat,           //its material
                         "CrystalLV");        //its name
            
   // place crystals within a ring 
   //
   for (G4int icrys = 0; icrys < nb_cryst ; icrys++) {
     G4double phi = icrys*dPhi;
     G4RotationMatrix rotm  = G4RotationMatrix();
     rotm.rotateY(90*deg); 
     rotm.rotateZ(phi);
     G4ThreeVector uz = G4ThreeVector(std::cos(phi),  std::sin(phi),0.);     
     G4ThreeVector position = (ring_R1+0.5*cryst_dZ)*uz;
     G4Transform3D transform = G4Transform3D(rotm,position);
                                     
     new G4PVPlacement(transform,             //rotation,position
                       logicCryst,            //its logical volume
                       "crystalVox",             //its name
                       logicRing,             //its mother  volume
                       false,                 //no boolean operation
                       icrys,                 //copy number
                       fCheckOverlaps);       // checking overlaps 
   }
                                                       
   //
   // full detector
   //
   G4Tubs* solidDetector =
     new G4Tubs("Detector", ring_R1, ring_R2, 0.5*detector_dZ, 0., twopi);
       
   G4LogicalVolume* logicDetector =                         
     new G4LogicalVolume(solidDetector,       //its solid
                         default_mat,         //its material
                         "Detector");         //its name
                                  
   // 
   // place rings within detector 
   //
   G4double OG = -0.5*(detector_dZ + cryst_dX);
   for (G4int iring = 0; iring < nb_rings ; iring++) {
     OG += cryst_dX;
        new G4PVPlacement(0,                     //no rotation
                       G4ThreeVector(0,0,OG), //position
                       logicRing,             //its logical volume
                       "ring",                //its name
                       logicDetector,         //its mother  volume
                       false,                 //no boolean operation
                       iring,                 //copy number
                       fCheckOverlaps);       // checking overlaps 
   }
                        
   //
   // place detector in world
   //                    
   new G4PVPlacement(0,                       //no rotation
                     G4ThreeVector(px,py,pz),         //at (0,0,0)
                     logicDetector,           //its logical volume
                     "Detector",              //its name
                     lWorld,              //its mother  volume
                     false,                   //no boolean operation
                     0,                       //copy number
                     fCheckOverlaps);         // checking overlaps 
                    
}
                                      

void DetectorConstruction::ConstructSDandField() {
///////// GENERATE SENSITIVE DETECTORS AND SCORERS ////////////
    G4SDManager *sdmanager = G4SDManager::GetSDMpointer();

    // setup general purpose voxelized sensitive detector
    G4MultiFunctionalDetector *mfd = new G4MultiFunctionalDetector("mfd");
    G4cout << "Attaching Dose MFD of name " << mfd->GetName() << " to SDmanager" << G4endl;
    sdmanager->AddNewDetector(mfd);
    SetSensitiveDetector("lRepX", mfd);

    // create magnetic field
    G4GlobalMagFieldMessenger* fMagFieldMessenger = new G4GlobalMagFieldMessenger(G4ThreeVector());

    // total dose
    G4PSDoseDeposit3D* dose3d = new G4PSDoseDeposit3D("dose3d", nz, ny, nx);
    G4cout << "Attaching primitive scorer of name " << dose3d->GetName() << " to mfd" << G4endl;
    mfd->RegisterPrimitive(dose3d);

    // positron annilation
    if (g_output_positronannihilation) {
      PositronAnnihilation3D* NofPositronAnni3D = new PositronAnnihilation3D("NofPositronAnni3D", nz, ny, nx);
      G4cout << "Attaching primitive scorer of name " << NofPositronAnni3D->GetName() << " to mfd" << G4endl;
      mfd->RegisterPrimitive(NofPositronAnni3D);
    }

    // declare crystal as a MultiFunctionalDetector scorer    
    G4MultiFunctionalDetector* cryst = new G4MultiFunctionalDetector("crystal");
    sdmanager->AddNewDetector(cryst);
    SetSensitiveDetector("CrystalLV",cryst);

}
