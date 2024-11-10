import yaml

def write_to_yaml(fname, data):
    with open(fname, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def load_from_yaml(fname):
    with open(fname, 'r') as f:
        data = yaml.safe_load(fname)
    return data