import pandas as pd
import pytesseract
from PIL import Image
from scipy.stats import mode
from matplotlib import pylab as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from math import fabs

"""
Работа алгоритма
1. Чтение файла и препроцессинг
2. MSER определение средних размеров букв
3. Определение границ символов cv2.findCounters
4. Фильтрация боксов символов по размеру: удаляем совсем мелкие и вложенные в другие
5. Определение слов в тексте Определение границ слов
6. Сопоставление границ слов с границами Tesseract
7. Сравнение длин CV2 и Tesseract
8. Найти соотвествия букв, т.е. примеры по классам букв
9. Сделать классификатор содержания бокса (ищем букву)
10. Найти и решить проблемные участки: там где тессеракт или опенси сиви ошибся
11. Найти расстояние мужду буквами в слове (после  пишем п. 5)
12. Визуализация полученного
"""

"""
1. Чтение файла и препроцессинг
"""
def preprocess_function(image_path):
    """
    :param image_path: путь к изображению
    :return: изображение, чб изображение, изображение по порогу
    """
    im = cv2.imread(image_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imwrite("gray.jpg", gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
    	ksize=-1)
    gradX = np.absolute(gradX)
    thresh1 = cv2.adaptiveThreshold(tophat,255,1 ,1,9,2)
    edged = cv2.Canny(gray, 10, 250)
    _, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im, gray, edged, bw

"""
2. MSER определение средних размеров букв
"""
def box_cteate(points_list):
    """
    :param points_list: список точек, описывающих контур
    :return: таблица вида: крайняя левая координата по x, верний угол контура по y, ширина, высота
    """
    box_list = []
    for hull in points_list:
        hull_np = np.array(hull).reshape((len(hull),2))
        min_x = np.min(hull_np[:, 0])
        min_y = np.min(hull_np[:, 1])
        max_x = np.max(hull_np[:, 0])
        max_y = np.max(hull_np[:, 1])
        width = max_x - min_x
        height = max_y - min_y
        box_list += [[min_x, min_y, width, height, max_y]]
    box_df = pd.DataFrame(box_list, columns=['left', 'top', 'width', 'height', 'l'])
    return box_df

def image_to_mser_data(edged):
    """
    :param thresh: изображение
    :return: таблица рамок
    """
    (h, w) = im.shape[:2]
    image_size = h*w
    mser = cv2.MSER_create()
    mser.setMaxArea(int(image_size/2))
    mser.setMinArea(10)
    regions, rects = mser.detectRegions(edged)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    box_df_mser = box_cteate(hulls)
    return box_df_mser

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def mean_box(box_df):
    """
    :param box_df: таблица рамок
    :return: средний размер рамки: ширина и высота
    """
    width_list = []
    height_list = []
    for i in range(len(box_df)):
        width_i = box_df['width'].iloc[i]
        height_i = box_df['height'].iloc[i]
        width_list += [width_i]
        height_list += [height_i]
    width_mean = int(mode(np.array(width_list))[0])
    height_mean = int(mode(np.array(height_list))[0])
    # print('количество модальных значений', mode(np.array(width_list))[1])
    # print('количество модальных значений', mode(np.array(height_list))[1])
    return width_mean, height_mean

"""
3. Определение границ символов cv2.findCounters
"""
def counters_function(edged):
    """
    :param thresh: патч
    :return: таблицу с границами каки-то символов
    """
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    box_df_counters = box_cteate(contours)
    print('количество боксов до фильтрации', len(box_df_counters))
    return box_df_counters

"""
4. Фильтрация боксов символов по размеру: удаляем совсем мелкие и вложенные в другие
"""

def filter_boxes(box_df_counters, width_mean, height_mean,k=0.8):
    # фильтры размера бокса
    box_df_counters = box_df_counters[box_df_counters["width"] > width_mean * k]
    box_df_counters = box_df_counters[box_df_counters["height"] > height_mean * k]
    box_df_counters = box_df_counters.sort_values(by=['l'])
    box_number = len(box_df_counters)
    box_df_counters.index = pd.RangeIndex(0, box_number)
    # добавление информации о строках и сортировка по порядку всех боксов
    line = 0
    box_df_counters['line'] = line
    for box in range(1, box_number):
        dist_top = box_df_counters['l'].iloc[box] - box_df_counters['l'].iloc[box - 1]
        print('расстояние между символами: ', dist_top, 'высота средняя: ', height_mean)
        print('высота символа', box_df_counters['height'].iloc[box])
        if dist_top > height_mean * k:
            line += 1
            print('новая строка')
        else:
            pass
        box_df_counters['line'].iloc[box] = line
    print(f'В документе {line + 1} строк')
    print(box_df_counters)
    box_df_counters = box_df_counters.sort_values(by=['line', 'left'])
    # удаление внутренних областей в буквах
    box_df_counters.index = pd.RangeIndex(0, box_number)
    dell_list =[]
    for box in range(1, box_number):
        dist_w = fabs(box_df_counters["left"].iloc[box] - box_df_counters["left"].iloc[box - 1])
        condition1 = (dist_w < width_mean * k)
        condition2 = (box_df_counters["left"].iloc[box] + box_df_counters["width"].iloc[box]
                    - box_df_counters["left"].iloc[box - 1] - box_df_counters["width"].iloc[box - 1]) < 0
        condition3 = (box_df_counters["line"].iloc[box] == box_df_counters["line"].iloc[box-1])
        condition = (condition1 | condition2) & condition3
        if condition:
            print('Внутренний бокс: ', dist_w, ' < ',  width_mean * k/2)
            dell_list += [box]
    print('list', dell_list)
    print('len', len(dell_list))
    box_df_counters.drop(dell_list, inplace=True)
    box_number = len(box_df_counters)
    box_df_counters.index = pd.RangeIndex(0, box_number)
    print('боксов после фильтрации', len(box_df_counters))
    box_df_counters.to_csv('result.csv')
    return box_df_counters

"""
5. Определение слов в тексте Определение границ слов
"""
def words_detection(box_df_counters, width_mean, k = 1.6):
    box_number = len(box_df_counters)
    box_df_counters.index = pd.RangeIndex(0, box_number)
    box_words = []
    word = -1
    box_df_counters['word'] = word
    word = 0
    box_df_counters['word'].iloc[0] = word
    for box in range(1, box_number):
        dist_x = fabs(box_df_counters['left'].iloc[box] - box_df_counters['left'].iloc[box - 1])
        dist_l_w = (box_df_counters['left'].iloc[box] - box_df_counters['left'].iloc[box - 1]
                    - box_df_counters['width'].iloc[box - 1])
        print('расстояние между символами: ', dist_x, 'ширина средняя*k: ', width_mean*k)
        print('ширина символа', box_df_counters['height'].iloc[box])
        condition1 = (dist_x > width_mean * k)
        condition2 = (box_df_counters["line"].iloc[box] == box_df_counters["line"].iloc[box-1])
        condition3 = (box_df_counters["line"].iloc[box] != box_df_counters["line"].iloc[box - 1])
        condition4 = dist_l_w > width_mean * k/4
        condition5 = condition1 & condition2 & condition4
        condition = condition5 | condition3
        print(f'условия: ({condition1} & {condition2} & {condition4}) | {condition3} = {condition}')
        if condition:
            print('новое слово')

            # находим координаты слова
            box_df_counters_copy = box_df_counters.copy()
            box_df_w = box_df_counters_copy[box_df_counters_copy['word'] == word]
            print('выборка по слову')
            print(box_df_w)
            if len(box_df_w) == 1:
                left = box_df_counters['left'].iloc[box - 1]
                top = box_df_counters['top'].iloc[box - 1]
                width = box_df_counters['width'].iloc[box - 1]
                line = box_df_counters['line'].iloc[box - 1]
            else:
                left = np.min(np.array(box_df_w['left']))
                print('left',left)
                top = box_df_counters['top'].iloc[box - 1]
                print('top', top)
                width = np.max(np.array(box_df_w['left'])) + box_df_counters['width'].iloc[box - 1] - left
                print('width', width)
                height = box_df_counters['height'].iloc[box - 1]
                print('height', height)
                line = box_df_counters['line'].iloc[box - 1]
                print('line', line)
            box_words += [[left, top, width, height, line, word]]

            word += 1

        elif condition4 == False: # скорее всего буква "ы"
            pass
        else:
            pass

        box_df_counters['word'].iloc[box] = word

    box_df_words = pd.DataFrame(box_words, columns=['left', 'top', 'width', 'height', 'line', 'word'])
    box_df_counters.to_csv('result.csv')
    box_df_words.to_csv('words_result.csv')
    return box_df_counters, box_df_words

"""
бн. Подача боксов в Tesseract: классификация содержимого бокса
"""

def char_box(bw, box_df_counters, height_mean, k=2):
    boxes_final = []
    h_bwp = w_bwp = int(height_mean * k)
    chares = ['а', 'А', 'б', 'Б','в', 'В', 'г', 'Г', 'д', 'Д', 'е', 'Е', 'ё', 'Ё', 'ж'
              , 'Ж', 'з', 'З', 'и', 'И', 'й', 'Й', 'к', 'К', 'л', 'Л', 'м', 'М', 'н', 'Н'
              , 'о', 'О', 'п', 'П', 'р', 'Р', 'с', 'С', 'т', 'Т', 'у', 'У', 'ф', 'Ф', 'х'
              , 'Х', 'ц', 'Ц', 'ч', 'Ч', 'ш', 'Ш', 'щ', 'Щ', 'ъ', 'ы', 'ь', 'э', 'Э', 'ю'
              , 'Ю', 'я', 'Я']
    frames = len(box_df_counters)
    for f in range(frames):
        box_with_padding = np.zeros((h_bwp, w_bwp))
        box_f = box_df_counters.iloc[f]
        f_y1 = box_df_counters['top'].iloc[f]
        f_y2 = box_df_counters['top'].iloc[f]+ box_df_counters['height'].iloc[f]
        f_x1 = box_df_counters['left'].iloc[f]
        f_x2 = box_df_counters['left'].iloc[f] + box_df_counters['width'].iloc[f]
        padding = 2
        char_box = thresh[f_y1-padding: f_y2+padding, f_x1-padding: f_x2+padding]
        h_cb = char_box.shape[0]
        w_cb = char_box.shape[1]
        px_bwp = int((w_bwp - w_cb) / 2)
        py_bwp = int((h_bwp - h_cb)/2)
        try:
            box_with_padding[py_bwp:(py_bwp+h_cb),px_bwp:(px_bwp+w_cb)] = char_box
        except ValueError:
            print(f'Ошибка записи в бокс{f}')
        # box_with_padding = cv2.resize(box_with_padding,(64,64))
        # print('размер бокса для сохранения', box_with_padding.shape)
        # box_with_padding = cv2.bitwise_not(box_with_padding)
        # cv2.imwrite(f'chars/{f}.png',box_with_padding)
    # https://stackoverflow.com/questions/20831612/getting-the-bounding-box-of-the-recognized-words-using-python-tesseract
        try:
            char_tes = pytesseract.image_to_string(Image.fromarray(box_with_padding), lang='rus')
            print('нашли букву... ',char_tes, type(char_tes))
        except SystemError:
            char_tes = None
        if char_tes in chares:
            box_f = list(box_f.append(pd.Series(char_tes)))
            boxes_final += [box_f]
    box_df_final = pd.DataFrame(boxes_final, columns=['left', 'top', 'width', 'height', 'l', 'line', 'word','char'])   
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    print('box_df_final: ', box_df_final.shape)
    return  box_df_final

"""
12. Визуализация полученного
"""
def plt_boxes(box_df, img, figsize=(15, 15), line_color = 'r', linewidth = 1):
    fig,ax = plt.subplots(1,figsize=figsize)
    vis = img.copy()
    ax.imshow(vis)
    for i in range(len(box_df)):
        left = box_df['left'].iloc[i]
        top = box_df['top'].iloc[i]
        width = box_df['width'].iloc[i]
        height = box_df['height'].iloc[i]
        rect1 = patches.Rectangle((left,top),width,height,linewidth=linewidth,edgecolor= line_color,facecolor='none')
        ax.add_patch(rect1)
    plt.savefig(f'{line_color}_result.png', format='png', dpi=300)
    print('Изображение сохранено локально')

if __name__ == '__main__':
    image_path= '12.jpg'
    im, gray, edged, bw = preprocess_function(image_path)
    box_df_mser = image_to_mser_data(edged)
    cv2.imwrite("big_boxes.jpg",edged)
    width_mean, height_mean = mean_box(box_df_mser)
    box_df_counters = counters_function(bw)
    box_df_counters = filter_boxes(box_df_counters, width_mean, height_mean,k=0.6)
    box_df_counters, box_df_words = words_detection(box_df_counters, width_mean)
    # box_df_final = char_box(thresh, box_df_counters, height_mean)
    plt_boxes(box_df_counters, im, figsize=(15, 15), line_color = 'r', linewidth = 1)
    plt_boxes(box_df_words, im, figsize=(15, 15), line_color='g', linewidth=1)

