====================
Overview
====================

There are two main ways to initialize with the SIMOC-ABM API:

1. **From Configuration** -- This method uses fully JSON-serializable inputs
   to initialize the AgentModel. Agents and currencies are instantiated 
   automatically, and base agent descriptions are automatically loaded from
   the agent library. This is the recommended method for preset configurations
   and basic users.

2. **From AgentModel** -- This method requires currencies and agents to be
   instantiated and added to the model manually, and provides a greater degree
   of flexibility and customization. 

Best Practices
==============