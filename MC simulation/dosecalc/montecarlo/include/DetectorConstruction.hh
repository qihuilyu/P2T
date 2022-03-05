#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>
#include <list>
#include <map>

class DetectorMessenger;
class G4Material;
class G4NistManager;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
struct matStruct {
	G4double den;
	G4int matID;

    bool operator == (const matStruct& other) {
        return std::fabs(den - other.den)<=1e-2 && (matID == other.matID);
    }
    friend bool operator < (const matStruct& lhs, const matStruct& rhs) {
        return (lhs.matID < rhs.matID) && (lhs.den < rhs.matID);
    }
};


class DetectorConstruction : public G4VUserDetectorConstruction
{
	public:
		DetectorConstruction();
		~DetectorConstruction();

		virtual void ConstructSDandField();
		G4VPhysicalVolume*  Construct();
		static DetectorConstruction* getInstance();
		static DetectorConstruction* instance;
		DetectorMessenger			*dMess;


	private:
        G4Material* fAir;
        std::vector<G4Material*> fOriginalMaterials;  // list of original materials

		G4int nx, ny, nz;
		G4long nxyz;
		G4double dx, dy, dz;
		G4double px, py, pz;
		G4LogicalVolume *lWorld;

        void InitialisationOfMaterials();
		void ReadPhantom();
		void MapMaterials();
		void CreateMaterial(G4long, G4int);
		void CreatePhantom();
		void CreateRingDetector();
		void SanityCheck();

		G4NistManager* man;
		G4Material *G4Air;

		//Input
		std::vector<matStruct> raw_matspecs;  // list of raw material specifications for every voxel (as read from file)

		//Intermediate
        std::list<matStruct>   unique_matspecs; // unique-ified list of materials from raw_matspecs
        std::vector<matStruct> vunique_matspecs; // vector

		//Feed to voxelisation
		std::vector<G4int> matMap;		//map voxel number to index of unique material instance in vunique_matspecs

        friend class DetectorMessenger;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
