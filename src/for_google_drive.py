""""
functions:
drive_connect()
"""

from google.colab import drive

# забавный факт: в .ipynb файле с фичами команда %run for_google_drive.py
# позволяет скипнуть авторизацию, а в других файлах нет
# WTF ????


def drive_connect():
    """
    function drive_connect() connects to a google-drive
    """
    drive.mount('/content/drive')


if __name__ == '__main__':
    drive_connect()
