"""The escalation ladder both cited-source fetchers run: outbound guard, context, classification.

Nothing is re-exported here on purpose. Every consumer imports the module it needs
(``guard``, ``context``, ``classify``) and reads the name off it at call time, so a test
patching a name intercepts the implementation the call path actually uses.
"""
