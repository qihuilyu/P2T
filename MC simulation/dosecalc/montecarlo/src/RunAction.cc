#include "RunAction.hh"

#include "G4RunManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4String.hh"
#include "G4THitsMap.hh"
#include "G4Threading.hh"

#include "Run.hh"
#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "os_utils.hh"
#include "detection.protobuf.pb.h"

// from ../main.cc
extern long int g_eventsProcessed;
extern G4String g_geoFname;
extern G4String g_outputdir;
extern bool     g_sparse;
extern double   g_sparse_threshold;
extern bool     g_snapshot_mode;
extern bool     g_cumulative_mode;
extern bool     g_individual_mode;
extern std::vector<std::vector<DetectionEvent>> all_thread_events;

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <map>
#include <cassert>
#include <sys/stat.h>
#include <unistd.h>

RunAction::RunAction()
{
    // remind detector replica size (z is fastest index)
    G4int nx, ny, nz;
    {
        G4String line;
        std::stringstream ss;

        std::ifstream infile;
        infile.open(g_geoFname);

        if (!infile.is_open()){
            G4cerr << "Failed opening Geometry" << G4endl;
            exit(1);
        }
        getline(infile, line);

        ss.str(line);
        // read header
        ss >> nx >> ny >> nz; // nvoxels
        infile.close();
    }
    det_size = {nx, ny, nz};
}

void RunAction::BeginOfRunAction(const G4Run*)
{
    /*
    From Geant4 Manual:
    This method is invoked before entering the event loop. A typical use of this method would be to initialize and/
    or book histograms for a particular run. This method is invoked after the calculation of the physics tables.
    */

    // each thread is responsible for (re)initializing its detectedevents vector at the start of each run
    all_thread_events[G4Threading::G4GetThreadId()].clear();

    if(IsMaster()){
        fRTally++;
        return;
    }
}

void RunAction::EndOfRunAction(const G4Run *run)
{
    /*
    From Geant4 Manual:
    This method is invoked at the very end of the run processing. It is typically used for a simple analysis of
    the processed run.
    */

    if( ! IsMaster()){
        G4cout<<"End of Run - worker thread terminated"<<G4endl;
        return;
    }
    //If we're here, should be master thread, collect all of the worker tallies
    long int nEventsThisRun = G4RunManager::GetRunManager()->GetCurrentRun()->GetNumberOfEventToBeProcessed();
    g_eventsProcessed += nEventsThisRun;
    G4cout << nEventsThisRun << " events processed in this run ("<<g_eventsProcessed<<" events in processed so far in the simulation)" << G4endl;
    G4cout << "Updating measurement output files..." << G4endl;

    G4SDManager *sdm = G4SDManager::GetSDMpointer();
    G4MultiFunctionalDetector *mfd =static_cast<G4MultiFunctionalDetector*>(sdm->FindSensitiveDetector(mfd_name));
    if (!mfd) { return; }

    //Note, a separate file will be created for each run (equivalently, each /run/beamOn N command in your
    //beamon.in file, which is most likely just one.)
    // again, downcasting is necessary in order to access the detectedEvents member variable of the Run() class
    // I convert to a reference here, becuase pointers annoy me for simple things like this. Now you can use
    // dot notation instead of arrow (->) notation to access members
    auto *_run = static_cast<const Run*>(run);
    // output full beam
    G4cout << "writing results for \"Full Beam\"" << G4endl;
    UpdateOutput(mfd, _run->hitsmaps_by_name, "");


}

std::vector<G4double> RunAction::calcVariance(const std::vector<G4double> dose, const std::vector<G4double> sqdose, unsigned long nsamples) {
    unsigned long n = dose.size();
    std::vector<G4double> var(n, 0.0);
    G4double factor = (1.0/(nsamples-1));
    for (unsigned long ii=0; ii<n; ii++) {
        var[ii] =  factor * (sqdose[ii]/nsamples - std::pow(dose[ii]/nsamples, 2.0));
        var[ii] = sqrt(var[ii])/dose[ii];
    }
    return var;
}

double RunAction::calcAvgRelativeUncertainty(const std::vector<G4double> rel_sd, const std::vector<G4double> dose, G4double thresh) {
    G4double max_dose = *std::max_element(dose.cbegin(), dose.cend());
    G4double sum = 0.0;
    double dose_thresh = thresh * max_dose;
    unsigned long count = 0;
    for (unsigned long int ii=0; ii<rel_sd.size(); ii++) {
        if (dose[ii] > dose_thresh) {
            sum += std::pow(rel_sd[ii], 2.0);
            ++count;
        }
    }
    return sum/count;
}

void update_cumulative_data(std::vector<G4double>& cumdata, const std::vector<G4double>& rundata, unsigned long total_samples, unsigned long run_samples) {
    // updates NORMALIZED cumulative data from NORMALIZED rundata
    unsigned long prev_samples = total_samples - run_samples;
    for (unsigned long ii=0; ii<cumdata.size(); ii++) {
        cumdata[ii] = (cumdata[ii]*prev_samples + rundata[ii]*run_samples) / (double)total_samples;
    }
}

std::vector<G4double> RunAction::process_hitsmap(G4THitsMap<G4double>* hitsmap) {
    std::vector<G4double> rundata(det_size.size(), 0.0);

    if (hitsmap) {
        for (G4int iz = 0; iz < det_size.z; ++iz) {
            for (G4int iy = 0; iy < det_size.y; ++iy) {
                for (G4int ix = 0; ix < det_size.x; ++ix) {
                    G4long copyto = ix + det_size.x*(iy + det_size.y*iz); // ZYX ordering
                    const G4long& copyfrom = copyto;
                    G4double* val = (*hitsmap)[copyfrom];
                    if (val) {
                        rundata[copyto] += *val; // normalize rundata
                    }
                }
            }
        }
    }
    return rundata;
}
std::vector<G4double> RunAction::normalize_vect(std::vector<G4double> vec, unsigned long int neventsthisrun) {
    auto normvec = vec; // copy cstr
    for (auto& x : normvec) {
        x /= neventsthisrun;
    }
    return normvec;
}


struct SparseData {
  std::vector<unsigned long> index;
  std::vector<double> value;
};

SparseData sparsify(const std::vector<G4double>& dense, double threshold=0.0) {
  // find max value
  double max_val = *std::max_element(dense.begin(), dense.end());
  double thresh = threshold*max_val;

  // Convert to sparse
  SparseData sparse;
  unsigned long nnz = 0;
  unsigned long ii = 0;
  for (; ii < dense.size(); ++ii) {
    if (dense[ii] >= thresh && dense[ii] != 0.0) {
      sparse.index.push_back(ii);
      sparse.value.push_back(dense[ii]);
      ++nnz;
    }
  }
  G4cout << "retained "<<nnz<<" nonzero elements of "<<ii<<" ("<<float(nnz)/ii*100.0<<"%)"<<G4endl;
  return sparse;
}
#define MAGIC_DENSE  "\x0\x0\xae\xfe"
#define MAGIC_SPARSE "\x0\x0\xae\xfd"
void write_sparse_data(const char* fname, const SparseData& sparse, iThreeVector volsize) {
  auto outfile = std::ofstream(fname, std::ios::out | std::ios::binary);

  char magic[] = MAGIC_SPARSE;
  outfile.write((const char*)&magic, sizeof(char)*4); // magic
  unsigned int usize[3] = {volsize.x, volsize.y, volsize.z};
  outfile.write((const char*)usize, 3*sizeof(unsigned int));
  unsigned long nnz = sparse.index.size();

  // assert (sizeof(unsigned long == 8));

  outfile.write((const char*)&nnz, sizeof(unsigned long)); // nnz
  // write data
  outfile.write((char*)sparse.index.data(), nnz*sizeof(unsigned long));
  outfile.write((char*)sparse.value.data(), nnz*sizeof(double));

  outfile.close();
}
void write_dense_data(const char* fname, const std::vector<double>& dense, iThreeVector volsize) {
  auto outfile = std::ofstream(fname, std::ios::out | std::ios::binary);
  outfile.write((char*)dense.data(), dense.size() * sizeof(G4double));
  outfile.close();
}

void RunAction::UpdateOutput(const G4MultiFunctionalDetector* mfd, const std::map<G4String, G4THitsMap<G4double>*>& hitsmaps, G4String fsuffix) {
    const G4Run& run = *G4RunManager::GetRunManager()->GetCurrentRun();
    long int nEventsThisRun = run.GetNumberOfEventToBeProcessed();

    const std::string& resultsdirname = g_outputdir;
    create_directory(resultsdirname, true);

    // create run-specific folder
    std::string rundirname;
    if (g_snapshot_mode || g_individual_mode) {
        std::ostringstream ss_rundirname;
        ss_rundirname << resultsdirname << "/" << "run" << std::setw(3)<<std::setfill('0')<<run.GetRunID();
        rundirname = ss_rundirname.str();
        create_directory(rundirname, true);
    }

    // store dose
    std::vector<G4double> store_dose3d;

    // Process hitsmaps into vectors
    for (G4int ii=0; ii < mfd->GetNumberOfPrimitives(); ++ii) {
        const G4VPrimitiveScorer& scorer = *mfd->GetPrimitive(ii);
        std::string detname = scorer.GetName();
        G4cout << detname << G4endl;

        std::vector<G4double> cumdata(det_size.size(), 0.0);
        std::vector<G4double> rawdata = process_hitsmap(hitsmaps.at(mfd_name + "/" + detname));
        if (detname.find("dose") != std::string::npos) {
          for (auto& v : rawdata) {
            v /= (joule/kg); // convert to units of Gy
          }
        }
        std::vector<G4double> rundata = normalize_vect(rawdata, nEventsThisRun);

        G4cout << "Processed hits map for scorer " << "\"mfd/" + detname << "\" with size: (" << det_size.x << ", " << det_size.y << ", " << det_size.z << ")" << G4endl;


        if (g_snapshot_mode || g_cumulative_mode) {
            G4String cum_fname;
            {
                std::ostringstream ss;
                ss << resultsdirname << "/" << detname << fsuffix << ".bin";
                cum_fname = ss.str();
            }

            // load existing cumulative data and update with rundata
            if (file_exists(cum_fname)) {
                std::ifstream infile(cum_fname.c_str(), std::ios::in | std::ios::binary);
                if (infile.fail()) {
                    G4cerr << "Error opening dose input file \""<<cum_fname<<"\"" << G4endl;
                } else {
                    infile.read((char*)cumdata.data(), cumdata.size()*sizeof(G4double));
                    infile.close();
                }
            }
            update_cumulative_data(cumdata, rundata, g_eventsProcessed, nEventsThisRun);

            // save to final
            if (g_cumulative_mode){
                auto outfile = std::ofstream(cum_fname.c_str(), std::ios::out | std::ios::binary);
                if (outfile.fail()) {
                    G4cerr << "Error opening dose output file \""<<cum_fname<<"\"" << G4endl;
                } else {
                    outfile.write((char*)cumdata.data(), cumdata.size() * sizeof(G4double));
                    outfile.close();
                }

                // // save variance
                // std::stringstream varoutname;
                // varoutname << resultsdirname << "/" << detname << fsuffix << "_cumulative_variance.bin";
                // auto varoutfile = std::ofstream(varoutname.str().c_str(), std::ios::out | std::ios::binary);
                // if (varoutfile.fail()) {
                //     G4cerr << "Error opening variance output file \""<<varoutname.str()<<"\"" << G4endl;
                // } else {
                //     varoutfile.write((char*)var.data(), var.size() * sizeof(G4double));
                //     varoutfile.close();
                // }
            }

            // save to snapshot
            if (g_snapshot_mode){
                std::ostringstream ss;
                // ss << rundirname << detname << fsuffix <<"_snapshot"<<std::setw(3)<<std::setfill('0')<<run.GetRunID()<< ".bin";
                ss << rundirname << "/" << detname << fsuffix <<"_snapshot.bin";
                G4String snap_fname = ss.str();

                auto outfile = std::ofstream(snap_fname.c_str(), std::ios::out | std::ios::binary);
                if (outfile.fail()) {
                    G4cerr << "Error opening dose output file \""<<snap_fname<<"\"" << G4endl;
                } else {
                    outfile.write((char*)cumdata.data(), cumdata.size() * sizeof(G4double));
                    outfile.close();
                }
            }
        }

        //===========================================

        if (g_individual_mode) {
          std::ostringstream ss;
          // ss << rundirname << "/" << detname << fsuffix << "_" << std::setw(3) << std::setfill('0') << run.GetRunID() << ".bin";
          ss << rundirname << "/" << detname << fsuffix << ".bin";
          auto run_fname = G4String(ss.str());

          // save separate file per run (reset measurement for each run)
          auto outfile = std::ofstream(run_fname.c_str(), std::ios::out | std::ios::binary);
          if (outfile.fail()) {
            G4cerr << "Error opening dose output file \""<<run_fname<<"\"" << G4endl;
          } else {
            if (g_sparse) {
              SparseData sparse;
              if (detname.compare("dose3d") == 0) {
                sparse = sparsify(rundata, g_sparse_threshold);
              } else {
                // only keep non-zero
                sparse = sparsify(rundata);
              }
              write_sparse_data(run_fname.c_str(), sparse, det_size);
            } else {
              outfile.write((char*)rundata.data(), rundata.size() * sizeof(G4double));
            }
            outfile.close();
          }

          if (detname.compare("dose3d") == 0) {
            store_dose3d = rawdata;
          }

/*
          // Output detectedEvents
          std::ostringstream ss_outname2;
          ss_outname2 << rundirname << "/DetectedEvents.raw";
          std::string outname2 = ss_outname2.str();
          auto outfile2 = std::ofstream(outname2, std::ios::out | std::ios::binary);
          if (outfile2.fail()) {
            G4cerr << "Error opening output file \""<<outname2<<"\""<<std::endl;
            return;
          }
          for (auto const& single_thread_events : all_thread_events) {
            outfile2.write((char*)single_thread_events.data(), single_thread_events.size()*sizeof(DetectionEvent));
          }
          outfile2.close();
*/

          // Output detectedEvents
          std::ostringstream ss_pb_outname;
          ss_pb_outname << rundirname << "/DetectedEvents.pb";

          std::string pb_outname = ss_pb_outname.str();
          auto outfile3 = std::ofstream(pb_outname, std::ios::out | std::ios::binary);
          if (outfile3.fail()) {
            G4cerr << "Error opening output file \""<<pb_outname<<"\""<<std::endl;
            return;
          }

          // I suggest google's protocol buffers (https://developers.google.com/protocol-buffers) becuase they
          // provide a .cc/.hh file generator for you, so you don't need to write the code yourself.
          // You just specify a protobuf file which describes your data format, then
          // generate the corresponding .cc and .hh file that provide the write() function to use.
          // see the c++ tutorial: https://developers.google.com/protocol-buffers/docs/cpptutorial
          // Instantiate your "Message" class which was compiled from your protobuf file
          detection::pbDetectedEvents pbDetectedEvents;
          // then add your event data to the provided classes
          for (auto const& single_thread_events : all_thread_events) {
            for (auto const& event : single_thread_events) {
              auto *pbEvent = pbDetectedEvents.add_detectionevent();
              pbEvent->set_eventid(event.eventID);
              pbEvent->set_detectorid(event.detectorID);
              pbEvent->set_globaltime(event.globalTime);
              pbEvent->set_energy(event.energy);
            }
          }
          // finally write to file (note: this is *instead* of the other outfile.write() call above for the simple POD struct approach). You can reuse the outfile declaration from above though
          if (!pbDetectedEvents.SerializeToOstream(&outfile3)) {
            std::cerr << "Failed to write protobuf file \"" << pb_outname<< "\"." << std::endl;
          }
          outfile3.close();

        }
    }
}

Run* RunAction::GenerateRun()
{
    /*
    From GEant4 Manual:
    This method is invoked at the beginning of the BeamOn() method but after confirmation of the conditions
    of the Geant4 kernel. This method should be used to instantiate a user-specific run class object.

    A.k.a. The "run" class is a user-created class for holding data, and performing functions related to information
    we want to collect during the run.  Consider it a data container for what we tally (with functions).
    */
    return new Run();
}
