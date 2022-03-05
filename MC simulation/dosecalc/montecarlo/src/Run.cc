
#include "Run.hh"

#include "G4SDManager.hh"
#include "G4THitsMap.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4Event.hh"
#include "G4String.hh"

#include <fstream>
#include <sstream>
#include <string>

Run::Run() {
	G4SDManager *sdm = G4SDManager::GetSDMpointer();
	G4MultiFunctionalDetector *mfd = static_cast<G4MultiFunctionalDetector*>(sdm->FindSensitiveDetector(mfd_name));

	if (!mfd) { return; }
    for (G4int icol=0; icol < mfd->GetNumberOfPrimitives(); ++icol) {
        G4VPrimitiveScorer* scorer = mfd->GetPrimitive(icol);
        G4String full_name = mfd_name + "/" + scorer->GetName();
        G4int collectionID = sdm->GetCollectionID(full_name);
        if (collectionID < 0) {
            G4cerr << "Collection not found." << std::endl;
            continue;
        }
        //referencing a map by [] is an overloaded operator for implicitly inserting values if that element doesnt exist -.- sneaky bastards
        hitsmaps_by_name[full_name] = new t_hitsmap(mfd_name, scorer->GetName());
    }
}

Run::~Run() {
    // cleanup dynamic allocations - full beam
    for (string_map_iter it=hitsmaps_by_name.begin(); it!=hitsmaps_by_name.end(); ++it) {
        delete it->second;
    }
}

void Run::RecordEvent(const G4Event* event)
{
  /*From GEant4 Manual:
    Method to be overwritten by the user for recording events in this (thread-local) run. At the end of the implementation,
    G4Run base-class method for must be invoked for recording data members in the base class.
    */
  // mandatory
  G4Run::RecordEvent(event);


  G4HCofThisEvent* pHCE = event->GetHCofThisEvent();
  if (!pHCE) {
    return;
  }

  auto* sdm = G4SDManager::GetSDMpointer();

  // full volume detection
  for (string_map_iter it = hitsmaps_by_name.begin(); it != hitsmaps_by_name.end(); ++it) {
    int icol = sdm->GetCollectionID(it->first);
    t_hitsmap* event_map = static_cast<t_hitsmap*>(pHCE->GetHC(icol));

    if (event_map) {
      // Add this event to thread_local hitsmap
      *hitsmaps_by_name[it->first] += *event_map;
    }
  }
}


void Run::Merge(const G4Run* thread_local_run) {
  /* Called from Global Run object in MultiThreaded mode with each thread_local run object as input
   * The user is responsible for taking results from each thread_local_run object and accumulating
   * them into "*this" which belongs to the global Run object
   */

  // this is necessary because we are in an implementation of a virtual method that was inherited by our Run()
  // class from its base G4Run() class. In c++, we wouldn't need this, if we only accessed members that were
  // originally declared in the base G4Run() class, but since we need to access members of our derived Run() class,
  // this type of static_cast is necessary before we can access anything specific to Run().
  // This is known as "downcasting" in "Polymorphism" - a topic in object orientated programming.
  // see: https://www.tutorialcup.com/cplusplus/upcasting-downcasting.htm
  const Run *local_run = static_cast<const Run*>(thread_local_run);

  // full beam
  for (string_map_iter it=local_run->hitsmaps_by_name.begin(); it != local_run->hitsmaps_by_name.end(); ++it) {
    G4cout << "Merging HitsMap from Run (full beam): " << it->first << G4endl;
    *hitsmaps_by_name[it->first] +=  *it->second;
  }

  // mandatory. This finally calls the original base G4Run() class's implementation of Merge, which safely
  // informs the master thread that it can allow other threads to merge now (preventing race conditions).
  G4Run::Merge(thread_local_run);

}
