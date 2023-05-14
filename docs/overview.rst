====================
Overview
====================

The SIMOC Ecosystem
===================
The SIMOC Ecosystem is a collection of software tools that can be used to 
simulate the operation of a Mars habitat.  The tools are designed to be used 
together, but can also be used independently.  The tools are:

* SIMOC-Web - A web-based interface for interacting with the simulation. A live
  version of this interface is hosted by National Geographic at 
  http://ngs.simoc.space.
* SIMOC-SAM - A package for taking live readings from sensors and sending them 
  to the SIMOC-Web interface. This package is used by the Space Analog for the
  Moon and Mars (SAM), an experimental research facility located at the
  University of Arizona's Biosphere 2 outside Oracle, Arizona.
* SIMOC-ABM - The core simulation engine (this package).

Basic Use: Preset Configurations
================================
SIMOC-ABM can be used to simulate arbitrary agent-based models, but it's
development has been guided by two main scenarios:

1. A Mars habitat. This was the original use case for the software.  The
   simulation is designed to model the operation of a habitat on Mars, with
   humans living inside.  The simulation can be used to test the viability of
   different habitat designs, and to test different operational strategies.

2. The Biosphere 2 facility. The Biosphere 2 facility is a large, sealed
   structure located in Oracle, Arizona.  It was originally built as a
   self-sustaining habitat, but is now used for research.  The facility is
   divided into several biomes, each with different environmental conditions.
   The facility is also used to test different operational strategies for
   managing the facility. SIMOC can be used to simulate the original 2 missions
   in 1991 and 1994, as well as test different configurations for the facility.
