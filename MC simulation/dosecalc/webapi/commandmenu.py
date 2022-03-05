import sys
import argparse

import parse

def menucommand(func):
    """Decorator to use with every 'command' belonging to a CommandMenuBase instance"""
    def wrapper(self):
        parser = parse.ArgumentParser(description=func.__doc__)
        return func(self, parser)
    wrapper.__dict__['ismenucommand'] = True
    wrapper.__doc__ = func.__doc__
    return wrapper

class CommandMenuBase():
    description = ''
    def __init__(self, description=None, parse_remaining=False, **kwargs):
        """
        Args:
            parse_remaining (bool): if True, don't treat as a submenu, but rather as a leaf, where all
                arguments will be processed as usual. The difference is that when True, the command menu
                will not leave any args for the parser in @menucommands
        """
        if not description:
            description = self.description # pull from class static variable
        self.parser = parse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description = str(description) + '\n\n' + \
                  'Command Menu Options:\n' +
                  '---------------------\n' +
                  ''.join(sorted(['{!s:20s} {!s}\n'.format(methodname, getattr(self, methodname).__doc__)
                      for methodname in dir(self) if 'ismenucommand' in dir(getattr(self, methodname))]))
        )
        self.parser.add_argument('command', help='Command to run')
        self.register_addl_args(self.parser)

        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        command_options = [x for x in dir(self) if 'ismenucommand' in dir(getattr(self, x))]
        parseend = len(sys.argv)
        if not parse_remaining:
            parseend = 1
            while parseend<len(sys.argv):
                parseend +=1
                if sys.argv[parseend-1] in command_options:
                    break

        try:
            self.args = self.parser.parse_args(sys.argv[1:parseend])
        except Exception as e:
            self.parser.print_help()
        del sys.argv[1:parseend]

        if not hasattr(self, self.args.command):
            print('Unrecognized command', file=sys.stderr)
            self.parser.print_help()
            exit(1)

        # call subclass defined instructions
        self.run_after_parse()

        # use dispatch pattern to invoke method with same name
        getattr(self, self.args.command)()

    def run_after_parse(self):
        """Define any actions to take after parsing arguments, and before calling the subcommand
        """
        pass

    def register_addl_args(self, parser):
        """Here you should add any additional arguments to parser and they
        will be automatically registered to this instance of the CommandMenu.
        These should be registered to the parser passed in as function parameter
        """
        pass
