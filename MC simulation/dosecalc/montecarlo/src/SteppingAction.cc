#include "SteppingAction.hh"
#include "Run.hh"

#include "G4Step.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4StepPoint.hh"
#include "G4VProcess.hh"
#include "G4Threading.hh"

extern std::vector<std::vector<DetectionEvent>> all_thread_events;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::~SteppingAction()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// User specified actions to perform at every step (everytime an event is updated)
void SteppingAction::UserSteppingAction(const G4Step* step)
{
    // get volume of the current step
    auto postStepPoint = step->GetPostStepPoint();
    auto volume = postStepPoint->GetTouchableHandle()->GetVolume();
    // G4cout << "Time: " << postStepPoint ->GetGlobalTime() << " ns" << G4endl;
    if(!volume){
        return;
    }

    auto preStepPoint = step->GetPreStepPoint();
    auto volume0 = preStepPoint->GetTouchableHandle()->GetVolume();

    // G4cout << "flagggg1" << G4endl;
    // G4cout << volume->GetName()  << G4endl;
    // G4cout << volume0->GetName()  << G4endl;
    // G4cout << step->GetTrack()->GetDefinition()->GetParticleName() << G4endl;

    if ((volume0->GetName() == "ring")
    && (volume->GetName() == "crystalVox")
    && (step->GetTrack()->GetDefinition()->GetParticleName() == "gamma") 
    && postStepPoint->GetTotalEnergy() > 0.4088
    && postStepPoint->GetTotalEnergy() < 0.6132) {
        /*
        G4cout << "flagggg3" << G4endl;
        G4cout << "detector ID: " << postStepPoint->GetTouchableHandle()->GetVolume() ->GetCopyNo() << G4endl;
        G4cout << "Time: " << postStepPoint ->GetGlobalTime() << " ns" << G4endl;
     //   G4cout << "Parent ID: " << step->GetTrack()->GetParentID() << G4endl;
     //   G4cout << "Track ID: " << step->GetTrack()->GetTrackID() << G4endl;
        G4cout << "Energy: " << postStepPoint->GetTotalEnergy() << G4endl;
        G4cout << "Event ID: " << G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() << G4endl;
        G4cout << "Thread ID: " << G4Threading::G4GetThreadId() << G4endl;

        std::ofstream myFile("ids.csv", std::ofstream::out | std::ofstream::app);
        // Send data to the stream
        myFile << postStepPoint->GetTouchableHandle()->GetVolume() ->GetCopyNo() << ",";
        myFile << postStepPoint ->GetGlobalTime() << ",";
        myFile << postStepPoint->GetTotalEnergy() << ",";
        myFile << G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() << ",";
        myFile << G4Threading::G4GetThreadId() << "\n";
        // Close the file
        myFile.close();
        */

        auto eventData = DetectionEvent{};
        eventData.eventID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
        eventData.detectorID = postStepPoint->GetTouchableHandle()->GetVolume() ->GetCopyNo() ;
        eventData.globalTime = postStepPoint->GetGlobalTime();
        eventData.energy = postStepPoint->GetTotalEnergy();

        //auto *_run = static_cast<Run*>(G4RunManager::GetRunManager()->GetCurrentRun());
        //_run->detectedEvents.push_back(eventData);

        //std::vector<DetectionEvent> currentDetectedEvent = all_thread_events[currentRun->runId];
        all_thread_events[G4Threading::G4GetThreadId()].push_back(eventData);

        //G4Run* currentRun = G4RunManager::GetRunManager()->GetCurrentRun();
        //currentRun->detectedEvents.push_back(eventData);
    }


    /*
    if ( step->GetTrack()->GetDefinition()->GetParticleName() == "gamma" ) {
        G4cout << "flagggg1" << G4endl;
        G4cout << volume->GetName()  << G4endl;
        //G4cout << volume2->GetName()  << G4endl;
        G4cout << fDetConstruction->GetRingPV()->GetName() << G4endl;
        G4cout << volume  << G4endl;
        G4cout << fDetConstruction->GetRingPV() << G4endl;


        G4cout << preStepPoint->GetTouchableHandle()->GetVolume()->GetInstanceID() << G4endl;
        G4cout << preStepPoint->GetTouchableHandle()->GetVolume(1) ->GetCopyNo() << G4endl;
        G4cout << preStepPoint->GetTouchableHandle()->GetVolume(1)->GetInstanceID() << G4endl;
        G4cout << preStepPoint->GetTouchableHandle()->GetVolume(2) ->GetCopyNo() << G4endl;
        G4cout << preStepPoint->GetTouchableHandle()->GetVolume(2)->GetInstanceID() << G4endl;
    }

    */

    /*
    const G4String processName="annihil";
    if ( step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() != processName) {

    } else {
        // G4cout << "flagggg" << G4endl;
    }

    G4StepPoint* point2 = step->GetPostStepPoint();
    G4cout << point2->GetProcessDefinedStep()->GetProcessName() << G4endl;
    G4cout << step->GetTrack()->GetParentID() << G4endl;
    G4cout << step->GetTotalEnergyDeposit() << G4endl;
    G4cout << step->GetStepLength() << G4endl;
    G4cout << step->GetTrack()->GetVolume()->GetName() << G4endl;
    G4cout << step->GetTrack()->GetVertexKineticEnergy() << G4endl;
    */
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

