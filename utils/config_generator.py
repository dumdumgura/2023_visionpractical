import yaml
import sys
from collections import OrderedDict

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

def save_config(config, filename):
    with open(filename, 'w') as file:
        ordered_dump(config, file, Dumper=yaml.SafeDumper)

def input_value(prompt, value_type):
    while True:
        user_input = input(prompt)
        if not user_input:
            return None

        if value_type == 'string':
            return user_input
        elif value_type == 'integer':
            try:
                return int(user_input)
            except ValueError:
                print("Invalid input, please enter an integer.")
        elif value_type == 'list':
            return [item.strip() for item in user_input.split(',')]
        else:
            return user_input

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <template_file_path> <output_file_path>")
        sys.exit(1)

    template_file = sys.argv[1]
    output_file = sys.argv[2]

    config = ordered_load(open(template_file, 'r'))

    keys_to_edit = input('Enter comma-separated keys to edit (e.g., dataset.supervision, arch.rank): ')
    keys_to_edit = [key.strip() for key in keys_to_edit.split(',')]

    def prompt_for_values(d, parent_key=''):
        for key, value_info in d.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value_info, dict) and 'value' in value_info and 'type' in value_info:
                if full_key in keys_to_edit:
                    prompt = f'Enter value for {full_key} (current: {value_info["value"]}): '
                    d[key]['value'] = input_value(prompt, value_info['type'])
            elif isinstance(value_info, dict):
                prompt_for_values(value_info, full_key)

    prompt_for_values(config)

    def convert_to_values(d):
        for key, value in list(d.items()):
            if isinstance(value, dict) and 'value' in value:
                d[key] = value['value']
            elif isinstance(value, dict):
                convert_to_values(value)

    convert_to_values(config)

    save_config(config, output_file)
    print(f"Configuration file saved as {output_file}")

if __name__ == "__main__":
    main()
