// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// $Id: PositronAnnihilation.cc 81087 2014-05-20 15:44:27Z gcosmo $
// GEANT4 tag $Name: geant4-09-04 $
//
// PositronAnnihilation
#include "PositronAnnihilation.hh"
#include "G4VSolid.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VPVParameterisation.hh"
#include "G4UnitsTable.hh"

////////////////////////////////////////////////////////////////////////////////
// (Description)
//   This is a primitive scorer class for scoring only squared energy deposit.
//
//
// Created: 2005-11-14  Tsukasa ASO, Akinori Kimura.
// 2010-07-22   Introduce Unit specification.
//
///////////////////////////////////////////////////////////////////////////////

PositronAnnihilation::PositronAnnihilation(G4String name, G4int depth)
  :G4VPrimitiveScorer(name,depth),HCID(-1),EvtMap(0)
{
}

PositronAnnihilation::PositronAnnihilation(G4String name, const G4String& unit,
				 G4int depth)
  :G4VPrimitiveScorer(name,depth),HCID(-1),EvtMap(0)
{
}

PositronAnnihilation::~PositronAnnihilation()
{;}

G4bool PositronAnnihilation::ProcessHits(G4Step* aStep,G4TouchableHistory*)
{
  // const G4String flagggg="e+ annihilation...";
  if (aStep->GetTrack()->GetKineticEnergy()== 0 && aStep->GetTrack()->GetDynamicParticle()->GetDefinition()->GetParticleName() == particleName) {
    // G4cout << flagggg << G4endl;
    // G4cout << aStep->GetPreStepPoint()->GetWeight() << G4endl;

    G4double weight = 1.0;
    weight *= aStep->GetPreStepPoint()->GetWeight();
    G4int  index = GetIndex(aStep);
    EvtMap->add(index,weight);
    // PrintAll();
    return TRUE;

  } else {
  return FALSE;
  }
}

void PositronAnnihilation::Initialize(G4HCofThisEvent* HCE)
{
  EvtMap = new G4THitsMap<G4double>(GetMultiFunctionalDetector()->GetName(),
				    GetName());
  if(HCID < 0) {HCID = GetCollectionID(0);}
  HCE->AddHitsCollection(HCID, (G4VHitsCollection*)EvtMap);
}

void PositronAnnihilation::EndOfEvent(G4HCofThisEvent*)
{;}

void PositronAnnihilation::clear()
{
  EvtMap->clear();
}

void PositronAnnihilation::DrawAll()
{;}

void PositronAnnihilation::PrintAll()
{
  G4cout << " MultiFunctionalDet  " << detector->GetName() << G4endl;
  G4cout << " PrimitiveScorer " << GetName() << G4endl;
  G4cout << " Number of entries " << EvtMap->entries() << G4endl;
  std::map<G4int,G4double*>::iterator itr = EvtMap->GetMap()->begin();
  for(; itr != EvtMap->GetMap()->end(); itr++) {
    G4cout << "  copy no.: " << itr->first
	   << " Positron Annihilation: "
	   << *(itr->second)/GetUnitValue()
	   << " ["<<GetUnit() <<"]"
	   << G4endl;
  }
}

