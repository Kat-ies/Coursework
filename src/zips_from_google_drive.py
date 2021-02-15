""""
functions:
drive_connect()
unpacking_zip()
"""

import zipfile
from google.colab import drive

directories = ['/content/drive/My Drive/КУ Курсачи/'
               'Курсовой проект 2020/WIDER_FACE (zip)/WIDER_train.zip']


def drive_connect():
    """function drive_connect() connects to google-drive"""
    drive.mount('/content/drive')


def unpacking_zip():
    """
    function unpacking_zip() unpacks zip files from the list of directories
    now we need only one archive, when we need more, we'll add new paths
    to the list and easily unpack all archives at once"""
    for zips in directories:
        (zipfile.ZipFile(zips, 'r')).extractall()


if __name__ == '__main__':
    drive_connect()
    unpacking_zip()
