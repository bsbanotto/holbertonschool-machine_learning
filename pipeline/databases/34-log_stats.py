#!/usr/bin/env python3
"""
Script to get stats about Nginx logs stored in MongoDB
"""
from pymongo import MongoClient

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client['logs']
    nginx_collection = db['nginx']

    # Print the total number of logs the way the checker wants it
    nginx_logs = nginx_collection.count_documents({})
    print(str(nginx_logs) + ' logs')
    print("Methods:")

    # Print the number of documents for each method
    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    method_stats_dict = {}

    for method in method:
        method_count = nginx_collection.count_documents({"method": method})
        method_stats_dict[method] = method_count

    for item, value in method_stats_dict.items():
        print('\tmethod ' + item + ': ' + str(value))

    # Print the number of documents with method=GET, path=/status
    status_count = nginx_collection.count_documents({'method': 'GET',
                                                     'path': "/status"})
    print(str(status_count) + " status check")
