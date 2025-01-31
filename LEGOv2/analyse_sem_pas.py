import re
from collections import defaultdict

class SemParse:
    def __init__(self):
        self.nested_counts = defaultdict(lambda: defaultdict(int))
    
    def _parse_recursive(self, tokens, index, current_dict):
        """Recursively parses tokens and constructs nested dictionaries with counts."""
        stack = []
        current_level = current_dict
        current_key = None

        while index < len(tokens):
            token = tokens[index]
            if token == "(":
                new_dict = defaultdict(lambda: defaultdict(int))
                if current_key is not None:
                    if not isinstance(current_level[current_key], defaultdict):
                        current_level[current_key] = defaultdict(lambda: defaultdict(int))
                    current_level[current_key] = new_dict
                stack.append((current_level, current_key))
                current_level = new_dict
                current_key = None
            elif token == ")":
                if stack:
                    current_level, current_key = stack.pop()
            elif "[" in token and "]" in token:
                key = re.match(r"(\w+)\[(.*?)\]", token)
                if key:
                    current_key, current_value = key.groups()
                    if current_key not in current_level:
                        current_level[current_key] = defaultdict(int)
                    if current_value not in current_level[current_key]:
                        current_level[current_key][current_value] = 0
                    current_level[current_key][current_value] += 1
            else:
                if current_key is not None:
                    if not isinstance(current_level[current_key], defaultdict):
                        current_level[current_key] = defaultdict(int)
                    if token not in current_level[current_key]:
                        current_level[current_key][token] = 0
                    current_level[current_key][token] += 1
                else:
                    if token not in current_level:
                        current_level[token] = 0
                    current_level[token] += 1
                    current_key = token
            index += 1
        return index, current_dict
    
    def count_entities(self, data_string):
        """Parses a formatted string and constructs nested dictionaries with counts."""
        if data_string.strip().lower() == "semantic no match":
            return {}
        tokens = re.findall(r"\(|\)|\w+\[.*?\]|\w+", data_string)
        _, parsed_dict = self._parse_recursive(tokens, 0, defaultdict(lambda: defaultdict(int)))
        return parsed_dict



 
