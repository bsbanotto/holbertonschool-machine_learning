#!/usr/bin/env python3
"""
Find schools where a topic is taught
"""


def schools_by_topic(mongo_collection, topic):
    """
    Args:
        mongo_collection: pymongo collection object
        topic (string): topic to find 
    """
    return mongo_collection.find({"topics": topic})
