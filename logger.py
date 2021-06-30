import os
from colorama import init, Fore, Back, Style

FORES = [ Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE ]
BACKS = [ Back.BLACK, Back.RED, Back.GREEN, Back.YELLOW, Back.BLUE, Back.MAGENTA, Back.CYAN, Back.WHITE ]
STYLES = [ Style.DIM, Style.NORMAL, Style.BRIGHT ]

class Logger():
    def __init__(self, outputPath):
        init()
        self.txtFile = os.path.join(outputPath, "dc_log.txt")
        self.info("Logging to {}".format(os.path.abspath(self.txtFile)))
        if os.path.exists(self.txtFile):
            os.remove(self.txtFile)

    def write(self, txt):
        with open(self.txtFile, "a") as f:
            f.write(txt + "\n")
        return

    def print(self, txt=''):
        print(txt)
        self.write(txt)
        return

    def info(self, txt):
        print(Style.BRIGHT + Fore.WHITE + Back.BLUE + "INFO:", end="\r")
        print(Style.RESET_ALL + "\t" + txt)
        self.write("INFO:\n\t{}".format(txt))
        return

    def warn(self, txt):
        print(Style.BRIGHT + Back.RED + "WARNING:", end="\r")
        print(Style.RESET_ALL + "\t" + txt)
        self.write("WARNING:\n\t{}".format(txt))
        return