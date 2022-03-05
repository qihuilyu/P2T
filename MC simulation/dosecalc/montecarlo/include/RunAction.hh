#ifndef RunAction_h
#define RunAction_h 1


#include "G4UserRunAction.hh"
#include "G4String.hh"
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

#include "Run.hh"

// create directory with full path "dir" and permissions "perms", ignore EEXIST error if exist_ok==True
int create_directory(const std::string& dir, const bool exist_ok=true, const mode_t perms=0775);

struct iThreeVector {
    int x, y, z;
    int size() { return x*y*z; }
};

class RunAction : public G4UserRunAction
{
    public:
        RunAction();
        ~RunAction() {}

        virtual void BeginOfRunAction(const G4Run *run);
        virtual void EndOfRunAction(const G4Run *run);
        virtual Run* GenerateRun();

    protected:
        void UpdateOutput(const G4MultiFunctionalDetector* mfd, const std::map<G4String, G4THitsMap<G4double>*>&, G4String fsuffix="");
        std::vector<G4double> process_hitsmap(G4THitsMap<G4double>* hitsmap);
        std::vector<G4double> normalize_vect(std::vector<G4double> vec, unsigned long neventsthisrun);
        std::vector<G4double> calcVariance(const std::vector<G4double> dose, const std::vector<G4double> sqdose, unsigned long nsamples);
        G4double calcAvgRelativeUncertainty(const std::vector<G4double> rel_sd, const std::vector<G4double> dose, G4double thresh=0.5);

    private:
        G4String mfd_name = "mfd";
        G4int fRTally = 0;
        iThreeVector det_size{-1,-1,-1}; // read from file on construction
};
#endif
