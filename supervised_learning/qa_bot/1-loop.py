#!/usr/bin/env python3
"""
Script that repeats a Q/A loop until user exits
Prompts user with 'Q: '
No answers are returned
Exit commands: exit, quit, goodbye, bye, case insensitive
"""
exit_commands = ['exit', 'quit', 'goodbye', 'bye']
while(True):
    d = input('Q: ')
    if d.lower() in exit_commands:
        print('A: Goodbye')
        break
    print("A: ")
