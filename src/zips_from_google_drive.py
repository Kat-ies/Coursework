import zipfile
from google.colab import drive

directories = ['/content/drive/My Drive/КУ Курсачи/Курсовой проект 2020/WIDER_FACE (zip)/WIDER_train.zip']


# function drive_connect connects to google-drive

def drive_connect():
    drive.mount('/content/drive')


# function unpacking_zip unpacks zip files from the list of directories
# now we need only one archive, when we need more, we'll add new paths
# to the list and easily unpack all archives at once

def unpacking_zip():
    # распакуем архивы из гугл диска
    z = []
    for zips in directories:
        z.append(zipfile.ZipFile(zips, 'r'))

    for archives in z:
        archives.extractall()


if __name__ == '__main__':
    drive_connect()
    unpacking_zip()
