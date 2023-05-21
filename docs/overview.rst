====================
Overview
====================
This section aims to build up an intuitive understanding of SIMOC-ABM works. 
For a more technical description, see the API documentation, and for hands-on 
examples see the tutorials.

Model
=====
SIMOC-ABM simulations are comprised of an AgentModel that is populated with 
agents and currencies, and then stepped forward in time. There are two method 
of initializing an AgentModel:

1. **The Configuration Method** -- Initializes model, agents and currencies
   from a single set of JSON-serializable keyword-arguments. Automatically 
   merges agents with data from the agent library and selects the correct
   agent class. This is the recommended method for exploring and building on
   preset configurations, or loading a saved simulation.

.. code-block:: python

    from simoc_abm.agent_model import AgentModel
    from simoc_abm.util load_preset_config

    # Load Configuration file
    config = load_preset_config('1h')
    # Configure Model
    model = AgentModel.from_config(**config)
    # Run Model until termination conditions are met
    model.run()

2. **The Direct Method** -- Requires currencies and agents to be instantiated 
   and added to the model manually, and provides a greater degree of 
   flexibility and customization. 

.. code-block:: python

    from simoc_abm.agent_model import AgentModel
    from simoc_abm.agents import BaseAgent

    # Initialize Agents
    habitat = BaseAgent(model, 'habitat', capacity={'o2': 1000}, storage={'o2': 1000})
    human = BaseAgent(model, 'human', flows={'in': {'o2': {'value': 1, 'connections': ['habitat']}}})
    # Configure Model
    model = AgentModel()
    model.add_currency('o2')
    model.add_agent('habitat', habitat)
    model.add_agent('human', human)
    # Run Model one step at a time
    for i in range(100):
        model.step()

After running a simulation, records can be extracted from the model using the
``get_records`` method, or the simulation state can be saved to a 
JSON-serializable dict using the ``save`` method. SIMOC-ABM includes a set of
helper functions for parsing and visualizing records.

.. code-block:: python

    from simoc_abm.viz import plot_agent

    # Extract Records
    records = model.get_records()
    # Plot flows for human agent
    plot_agent(records, 'human', 'flows')

Agents
======
Agents are the main building-blocks of SIMOC-ABM. They are instantiated using
``BaseAgent`` or one of its subclasses from a JSON-serializable dict of 
keyword-arguments. These fields are copied directly into the agent's state,
determine its behavior, provide the scaffolding for auto-generated records, and
are exported directly to the save file. 

Each of the fields described below is designated either *static* (unchanging) 
or *dynamic* (changing). Dynamic fields are automatically recorded at the end 
of each step, and can be extracted using the ``get_records`` method. Static 
fields are not required to be recorded each step, but can be added to the
extracted records by adding the ``static=True`` keyword-argument.

Amount and Active
-----------------
The baseline amount is stored in the ``amount`` field (static), while the
amount currently alive is stored in the ``active`` field (dynamic).

Capacity and Storage
--------------------
Agents can store currencies for which they have capacity, up to the specified 
amount (typically in kg). The ``capacity`` field is a dict of currency names 
and *maximum* amounts, and the ``storage`` field is a dict of currency names
and *current* amounts. Capacity records are static, while storage is dynamic.

Properties and Attributes
-------------------------
Variabes related to an agent's state, such as age or height, are stored in one
two fields: ``properties`` for static variables and ``attributes`` for dynamic 
variables. Attributes are updated each step by the class object - some 
universal, like ``age`` and some class-specific, like ``par_factor``.

.. tip::
  When creating custom agents, it's recommended to use these fields as much as
  possible for efficient record-keeping and extraction - i.e. add new variables 
  to the  ``properties`` or ``attributes`` fields, rather than adding them to the 
  agent class directly.

Flows
-----
Flow are currency exchanges with other agents. Flows definitions (static) are 
stored in the ``flows`` field in the agent class, while actual flows at each 
step (dynamic) are recorded as ``flows`` in the records object. Flows require 
at a minumum a  direction, currency, value and at least one connection. The 
basic structure is shown below:

.. code-block:: python

    flows: {
        'in': {
            'co2': {
                'value': 1,
                'connections': ['agent_id_1'],
                ...
            },
        },
        'out': {...},
    }

Several optional parameters are also available which define how the flow 
changes over time or responds to environment. These are all defined inside
the inner-most dict, parellel to value and connections:

* **value** can be an integer or float. In some cases, flows are defined with
  a value of 0, e.g. to add a connection to be used by the agent class.
* **connections** is a list of agent_id's. When the flow is executed in the 
  step function, it will try to transfer the specified amount of currency to
  or from the first connection; if that agent does not have enough currency,
  it will try for the remainder from the next connection, and so on. 
* **criteria** are conditions that must be met for the flow to occur. These
  are defined as a dict of {path: criterion} pairs, where ``path`` is a string
  representing the path to the variable, and criterion is a dict including (at
  a minimum) ``limit``, a string representing the operator (e.g. ``<``), and
  ``value``, the value to compare against. Criterion can optionally include a
  ``buffer``, which is a time-delay before activating.
* **deprive** is a list of currencies: value pairs which define the amount of 
  steps without available currency after which the agent will die.
* **weighted** is a list of variables (storage levels, properties, attributes
  or integers) by which the flow value is multiplied. This is used for, among
  other things, scaling the flows of a plant to its current stored biomass and 
  to the time of day.
* **requires** scales a flow value to the proportion of some available input
  currency. For example, if humans receive only half of their desired ``potable``
  water input, their ``urine`` output will be scaled to half of its normal value.
* **growth** variables apply standard mathematical curves, such as ``normal``
  and ``sigmoid``, to the flow value over the ``daily`` or ``lifetime`` cycle in
  the form of a multiplier (mean=1).
  
.. tip::
  The recommended approach to flows is to define its *mean* value in the 
  ``value`` field, and then use the other parameters to scale it up and down. 
  For example, for plant biomass accumulation, the ``value`` field is set to 
  the mean lifetime value, and it's then scaled for lifetime growth, daily 
  growth, and up or down in response to light and CO2 levels.