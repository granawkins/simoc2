.. SIMOC-ABM documentation master file, created by
   sphinx-quickstart on Sun May 14 07:50:59 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SIMOC-ABM: Agent-Based Modeling for Ecosystems
==============================================

SIMOC-ABM is a Agent-Based Model (ABM) built in Python and based on a 
currency-exchange paradigm. It was developed for high-fidelity ecosystem
simulations and includes a library of pre-built agents and configurations, 
but can also be used to simulate arbitrary agent-based models. 

SIMOC-ABM is the 'backend' package of the Scalable Interactive Model of an 
Off-world Community (SIMOC), and is designed to work seamlessly with SIMOC-Web,
a graphical user interface, and SIMOC-Live, a package for taking live readings
from physical sensors.

SIMOC has been developed for specific use-cases: first to model a Mars habitat
using data from NASA, and second to model the Biosphere 2 Missions using data
from published literature and direct input from Biosphere 2 scientists.

Features
========

* Built on a **currency-exchange paradigm**, with built-in functions for 
  specifying in-flows and out-flows, storage and capacities, thresholds and 
  environmental responses, and more.
* Includes an **agent library** built on scientific literature including 
  humans, plants, environmental control and life support systems, and more, as 
  well as several **preset configurations** for functional Mars Habitats and
  the historic Biosphere 2 missions.
* Functions as a stand-alone ABM or within the **SIMOC ecosystem**, which 
  includes a graphical user interface (SIMOC-Web) and pacakge for taking live
  readings from physical sensors (SIMOC-Live).

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   SIMOC-ABM Overview <overview>
   Basic Tutorial <tutorials/basic_tutorial>
   Advanced Tutorial <tutorials/advanced_tutorial>
   API Documentation <api>

