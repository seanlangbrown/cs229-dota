'''
Citation: Anthropic. (2024). Claude 3.5 Sonnet [Large Language Model]. Retrieved from https://www.anthropic.com
'''

import json

class JSONStreamWriter:
    def __init__(self, filename, list=False):
        self.filename = filename
        self.first_item = True
    
    def __enter__(self):
        self.file = open(self.filename, 'w')
        if list is True:
            print("writing a list")
            self.file.write('[')
        else:
            print("writing an object")
            self.file.write('{')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if list is True:
            self.file.write(']')
        else:
            self.file.write('}')
        self.file.close()
    
    def write(self, key, value):
        if list is True:
            raise Exception('cannot write key,value to a list')
        if not self.first_item:
            self.file.write(',')
        
        json.dump(key, self.file)
        self.file.write(':')
        json.dump(value, self.file)
        
        self.first_item = False
    
    def writeValue(self, value):
        if not list:
            raise Exception('cannot write value to an object')
        if not self.first_item:
            self.file.write(',')

        json.dump(value, self.file)
        
        self.first_item = False