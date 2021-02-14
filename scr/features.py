from lxml import etree
from collections import namedtuple
import os

MAXSIZE = (25, 25)


class HaarFeature:

    def __init__(self, rect_list):
        self.rect_list = rect_list

    def cacl_feature(self, cumsum_marix):
        f_x = 0
        for rects in self.rect_list:
            if (rects.w, rects.h) >= MAXSIZE:
                continue
            else:
                f_x += (cumsum_marix[rects.y - 1 + rects.h][rects.x - 1 + rects.w]
                        - cumsum_marix[rects.y - 1][rects.x - 1 + rects.w]
                        - cumsum_marix[rects.y - 1 + rects.h][rects.x - 1]
                        + cumsum_marix[rects.y - 1][rects.x - 1]) * rects.weight
        return f_x


PATH = '/content/drive/My Drive/КУ Курсачи/Курсовой проект 2020/files/'
HaarRect = namedtuple('HaarRect', 'x y w h weight')


def parse_file():
    h_features = []
    # загрузим xml и распарсим его
    with open(os.path.join(PATH, 'my_features.xml')) as fobj:
        xml = fobj.read()
        root = etree.fromstring(xml)

        for elems in root.getchildren():
            for rects in elems.getchildren():
                rectangles = []
                for haar_rect in rects.getchildren():
                    numbers = haar_rect.text[11:].split(" ")
                    rectangles.append(HaarRect(int(numbers[0]), int(numbers[1]), int(
                        numbers[2]), int(numbers[3]), float(numbers[4])))

                h_features.append(HaarFeature(rectangles))
    return h_features


if __name__ == '__main__':
    haars_features = parse_file()
    print(haars_features[0].rect_list)
