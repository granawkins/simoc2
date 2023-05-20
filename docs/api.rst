=====================
Classes API Reference
=====================

Agent Model
===========
.. autoclass:: simoc_abm.agent_model.AgentModel
    :members: __init__, add_currency, add_agent, from_config, step, run, get_records, save

Agent Classes
=============
.. autoclass:: simoc_abm.agents.BaseAgent
    :members: __init__, register, register_flow, view, get_records, save, increment, get_flow_value, process_flow, step, kill

.. autoclass:: simoc_abm.agents.AtmosphereEqualizerAgent
    :members: register, step

.. autoclass:: simoc_abm.agents.ConcreteAgent
    :members: __init__, calc_max_carbonation, step

.. autoclass:: simoc_abm.agents.LampAgent
    :members: _update_lamp_attributes, register, step

.. autoclass:: simoc_abm.agents.PlantAgent
    :members: register, get_flow_value, _calculate_co2_response, step, kill

.. autoclass:: simoc_abm.agents.SunAgent
    :members: __init__, step

Utilities
=========
.. autofunction:: simoc_abm.util.load_preset_configuration

.. autofunction:: simoc_abm.util.recursively_check_required_kwargs

.. autofunction:: simoc_abm.util.evaluate_reference

.. autofunction:: simoc_abm.util.evaluate_growth

.. autofunction:: simoc_abm.util.parse_data

Visualization
=============
.. autofunction:: simoc_abm.viz.plot_agent