"""Fault injection package.

Import concrete fault functions from their modules, for example
``src.fault_injection.gaussian_noise``. Keeping package initialization light
prevents optional plotting dependencies in future fault modules from affecting
the Gaussian experiment path.
"""
