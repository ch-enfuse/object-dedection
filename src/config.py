import json

class Config:
    def __init__(self, config_file='src/config.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
    def get(self, key):
        return self.config.get(key)