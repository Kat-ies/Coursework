""""
functions:
drive_connect()
"""

from google.colab import drive


def drive_connect():
    """
    function drive_connect() connects to a google-drive
    """
    drive.mount('/content/drive')


if __name__ == '__main__':
    drive_connect()
