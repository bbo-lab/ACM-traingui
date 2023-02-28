# FIx for pyinstaller, see https://github.com/pyinstaller/pyinstaller/issues/7309
import os
import sys
if getattr(sys, 'frozen', False):
    dll_dir = os.path.join(sys._MEIPASS, 'av.libs')
    os.environ['PATH'] = dll_dir + os.pathsep + os.environ['PATH'] 
import av

from ACMtraingui.__main__ import main
import tqdm 

if __name__ == '__main__':
    main()