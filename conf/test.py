import clg
import yaml
import yamlordereddictloader
from os import path



# Add custom command-line types.
# Add custom command-line types.
from deploy9 import InterfaceType, DiskType, FormatType
clg.TYPES.update({'Interface': InterfaceType, 'Disk': DiskType, 'Format': FormatType})
def main():
    cmd = clg.CommandLine(yaml.load(open('test.yaml'),
                                    Loader=yamlordereddictloader.Loader))
    args = cmd.parse()

    print("Namespace object: %s" % args)
    print("Namespace attributes: %s" % vars(args))
    print("Iter arguments:")
    for arg, value in args:
        print("  %s: %s" % (arg, value))
if __name__ == '__main__':
    main()