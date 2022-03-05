#ifdef G4MULTITHREADED
    #include "G4MTRunManager.hh"
    #include "G4Threading.hh"
	#include <unistd.h>
#else
    #include "G4RunManager.hh"
#endif
#include "Randomize.hh"
#include "G4UImanager.hh"    // enables use of .in files
#include "G4VisExecutive.hh" // enables heprap file output
#include "G4UIExecutive.hh"  // enables input files (UI Commands)
#include "G4SystemOfUnits.hh"

#include "AllActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"              // required
#include "PrimaryGeneratorAction.hh" // just to get NUM_THREADS definition
#include "RunAction.hh"
#include "G4ParallelWorldPhysics.hh"
#include "G4ios.hh"
#include "SteppingAction.hh"

#include <vector>
#include <exception>

#include "os_utils.hh"

#include "cxxopts.hpp"

// keep a count of the number of events that have already been processed; updated after each run by the master thread
long int g_eventsProcessed = 0;
G4String g_geoFname; // set using argv[1]
G4String g_outputdir = "./";
bool g_sparse;
double g_sparse_threshold;
bool g_snapshot_mode;
bool g_cumulative_mode;
bool g_individual_mode;

bool g_output_positronannihilation;

std::vector<std::vector<DetectionEvent> > all_thread_events;

void print_setting(bool setting, char* description) {
    G4cout <<"Setting \"" << description << "\" is ";
    if (setting) {
        G4cout << "enabled";
    } else {
        G4cout << "disabled";
    }
    G4cout << G4endl;
}

int main( int argc, char** argv )
{
    // Parse command line args
    cxxopts::Options options(argv[0], "MC Simulation in CT Geometry");
    options.add_options()
        ("c,cumulative", "accumulate all runs into a single file [default]")
        ("s,snapshot", "retain cumulative snapshots after each run (also enables cumulative mode)")
        ("i,individual", "store results of each run in separate file/dir")
        ("o,outputdir", "Output Directory", cxxopts::value<std::string>()->default_value("./"))
        ("sparse", "Output in sparse format")
        ("sparse-threshold", "Set 'keep data' threshold as fraction of beamlet maximum (0.0: keep all non-zero, 1.0: keep only maximum)", cxxopts::value<double>()->default_value("0.0"))
        ("output-positronannihilation", "also output voxel-wise positron annihilation counts")
        ("h,help", "display this help message")
        ("geometry", "Geometry file", cxxopts::value<std::string>())
        ("inputs", "Geant4 macro/input files", cxxopts::value<std::vector<std::string>>())
        ;
    options.parse_positional({"geometry", "inputs"});
    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("geometry") || !result.count("inputs")) {
        options.positional_help("<geometry-file> <input-file> [<input-file> ...]");
        G4cout << options.help();
        exit(0);
    }

    g_snapshot_mode = result["snapshot"].as<bool>();
    g_cumulative_mode = result["cumulative"].as<bool>();
    g_individual_mode = result["individual"].as<bool>();

    // for now, must enable cumulative if snapshot is enabled
    if (g_snapshot_mode) {
        g_cumulative_mode = true;
    }

    if (!g_snapshot_mode && !g_cumulative_mode && !g_individual_mode) {
        g_cumulative_mode = true;
    }

    print_setting(g_snapshot_mode,    "Result Snapshotting");
    print_setting(g_cumulative_mode,  "Result Accumulation");
    print_setting(g_individual_mode,  "Individual Run-Saving");

    g_geoFname = result["geometry"].as<std::string>();
    g_outputdir = result["outputdir"].as<std::string>();
    g_sparse = result["sparse"].as<bool>();
    g_sparse_threshold = result["sparse-threshold"].as<double>();
    g_output_positronannihilation = result["output-positronannihilation"].as<bool>();

    create_directory(g_outputdir, true);
    G4cout << "Using geometry file: \""<<g_geoFname<<"\""<<G4endl;
    G4cout << "Result destination: \""<<g_outputdir<<"\""<<G4endl;
    G4cout << "Output format: \""<<(g_sparse?"Sparse":"Dense")<<"\""<<G4endl;
    if (g_sparse) {
      G4cout << "Sparse threshold: "<< g_sparse_threshold*100.0 << "% of beamlet maximum"<<G4endl;
    }
    G4cout << "Optional Data Outputs:" << G4endl;
    G4cout << "  - Positron Annihilation Counts: "<<(g_output_positronannihilation?"YES":"NO")<<G4endl;
    G4cout<<G4endl;

    G4cout << "Using input files: {";
    auto& input_files = result["inputs"].as<std::vector<std::string>>();
    for (const auto& in : input_files) {
        G4cout << in << ",";
    }
    G4cout << "}" << G4endl;

    if (g_sparse && g_cumulative_mode) {
      throw std::runtime_error("Sparse output in cumulative output mode is not supported");
    }

    G4cout << "Creating Run Manager ..." << G4endl;
    #ifdef G4MULTITHREADED
        G4cout << "Running multithreaded." << G4endl;
        G4MTRunManager *runManager = new G4MTRunManager{};

        // enable run-level seeding (instead of event-level seeding) in MT;
        // recommended for high event-count runs (like all medical physics dose calculation)
        runManager->SetSeedOncePerCommunication(1);
        G4cout << "Using MT seeding strategy: ";
        switch (runManager->SeedOncePerCommunication()) {
            case 0 :
                G4cout << "event-level" << G4endl
                    << " (warning: event-level seeding is unsuitable for high-run-count simulation, prefer run-level instead)";
                break;
            case 1 :
                G4cout << "run-level";
                break;
            default :
                G4cout << runManager->SeedOncePerCommunication();
        }
        G4cout << G4endl;
        G4int number_of_cores = G4Threading::G4GetNumberOfCores();
        all_thread_events.resize(number_of_cores);

        #ifdef USEPHASESPACE
            // set NUM_THREADS in PrimaryGeneratorAction.hh
            number_of_cores = NUM_THREADS;
            G4cout << "Number of cores/threads in use: "
                << number_of_cores << " of "
                << G4Threading::G4GetNumberOfCores()
                << G4endl;
            runManager->SetNumberOfThreads(number_of_cores);
        #endif


    #else
        G4cout << "Running single threaded." << G4endl;
        G4RunManager* runManager = new G4RunManager;
    #endif

    // prng seed
    G4int t1 = time(NULL);
    G4int seed = t1%900000000; // prevents seed overflow
    // G4Random::setTheEngine(new CLHEP::Ranlux64Engine());
    // G4Random::setTheEngine(new CLHEP::MTwistEngine()); // uses two seeds
    auto *engine = G4Random::getTheEngine();
    G4Random::setTheSeed(seed);
    G4cout << "Psuedo-RNG seed: " << seed << G4endl;
    engine->showStatus();


    /*------------------ Mandatory Init Classes ---------------------------------------*/
    // Geometry - construct
    DetectorConstruction* det = DetectorConstruction::getInstance();
    runManager->SetUserInitialization(det);

    // Physics - register instance of selected physics with runManager
    G4VModularPhysicsList* physics = new PhysicsList;
    runManager->SetUserInitialization(physics);

    // Register all "UserActions": Particle generation, Stepping Actions, Event Actions ... etc
    G4VUserActionInitialization* AAI = new AllActionInitialization();
    runManager->SetUserInitialization(AAI);
    /*---------------------------------------------------------------------------------*/

    // Visualization
    G4VisManager* visManager = new G4VisExecutive;
    visManager->Initialize();

    // Issue runtime commands to program in the form of input files (*.in)
    G4UImanager* UI = G4UImanager::GetUIpointer();

    // std::ofstream myFile("ids.csv");
    // // Send data to the stream
    // myFile << "DetectorID" << ",";
    // myFile << "Time" << ",";
    // myFile << "Energy" << ",";
    // myFile << "EventID" << ",";
    // myFile << "ThreadID" << "\n";
    // // Close the file
    // myFile.close();

    // Parse Input file(s)
    // input file must contain "/run/beamOn ###" for run to start
    G4String command = "/control/execute ";
    for (const auto& in : input_files) {
        G4String macroFileName = in;
        UI->ApplyCommand(command+macroFileName);
    }

    G4int t2 = time(NULL);
    G4cout << "Total Runtime: " << difftime(t2, t1) << " seconds" << G4endl;

    // job termination
    delete runManager;
    return 0;
}
