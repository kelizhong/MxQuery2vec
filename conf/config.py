import clg
import yaml
import yamlordereddictloader
from os import path

CMD_FILE = path.abspath(path.join(path.dirname(__file__), 'kvm.yml'))

# Add custom command-line types.
# Add custom command-line types.
from deploy9 import InterfaceType, DiskType, FormatType
clg.TYPES.update({'Interface': InterfaceType, 'Disk': DiskType, 'Format': FormatType})
def main():
    cmd = clg.CommandLine(yaml.load(open('test.yaml'),
                                    Loader=yamlordereddictloader.Loader))
    cmd.parse()

if __name__ == '__main__':
    main()