"""A merchant checkout that integrates against the routing service over HTTP.

Deliberately a separate process on a separate port. The router is a service;
the only way to show that is to have something else actually consume it.
"""
