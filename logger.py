import os
from colorama import init, Fore, Back, Style

FORES = [ Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE ]
BACKS = [ Back.BLACK, Back.RED, Back.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN, Back.WHITE ]
STYLES = [ Style.DIM, Style.NORMAL, Style.BRIGHT ]

class Logger():
    """ Simple, cool logger that adds color to info and warnings"""
    def __init__(self, outputPath, overwrite=False):
        init()
        self.txtFile = os.path.join(outputPath, "dc_log.txt")
        self.info("Logging to {}".format(os.path.abspath(self.txtFile)))
        if overwrite and os.path.exists(self.txtFile):
            os.remove(self.txtFile)

    def write(self, txt):
        if os.path.exists(self.txtFile):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        with open(self.txtFile, append_write) as f:
            f.write(txt + "\n")
        return

    def print(self, txt=''):
        print(txt)
        self.write(txt)
        return

    def info(self, txt):
        print(Style.BRIGHT + Fore.BLACK + Back.CYAN + "INFO:" + Style.RESET_ALL)
        print("\t" + txt)
        self.write("INFO:\n\t{}".format(txt))
        return

    def warn(self, txt):
        print(Style.BRIGHT + Back.RED + "WARNING:" + Style.RESET_ALL)
        print("\t" + txt)
        self.write("WARNING:\n\t{}".format(txt))
        return