import pandas as pd
import pytesseract
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
from scipy.stats import mode
from matplotlib import pylab as plt
from math import fabs

"""
Работа алгоритма
1. Чтение файла и препроцессинг
2. MSER определение средних размеров букв
3. Определение границ символов cv2.findCounters
4. Фильтрация боксов символов по размеру: удаляем совсем мелкие и вложенные в другие
5. Определение слов в тексте. Определение границ слов. Найти расстояние мужду буквами в слове
6. Сопоставление границ слов с границами Tesseract
7. Сравнение длин CV2 и Tesseract
8. Найти соотвествия букв, т.е. примеры по классам букв
9. Сделать классификатор содержания бокса (ищем букву)
10. Найти и решить проблемные участки: там где тессеракт или опенси сиви ошибся
11. Визуализация полученного
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
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return im, gray, thresh

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

def image_to_mser_data(thresh):
    """
    :param thresh: изображение
    :return: таблица рамок
    """
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(thresh)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    box_df_mser = box_cteate(hulls)
    return box_df_mser

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
def counters_function(thresh):
    """
    :param thresh: патч
    :return: таблицу с границами каки-то символов
    """
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    box_df_counters = box_cteate(contours)
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
        else:
            pass
        box_df_counters['line'].iloc[box] = line
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
    box_df_counters.drop(dell_list, inplace=True)
    box_number = len(box_df_counters)
    box_df_counters.index = pd.RangeIndex(0, box_number)
    box_df_counters.to_csv('result.csv')
    return box_df_counters

"""
5. Определение слов в тексте Определение границ слов
"""
def words_detection(box_df_counters, width_mean, k1 = 0.4, k2 = 0.6):
    box_number = len(box_df_counters)
    box_df_counters.index = pd.RangeIndex(0, box_number)
    box_words = []
    word = -1
    box_df_counters['word'] = word
    word = 0
    box_df_counters['word'].iloc[0] = word
    dist_l = dist_w = None
    box_df_counters['dist_l'] = dist_l
    box_df_counters['dist_w'] = dist_w

    box_s = []
    for box in range(1, box_number):
        dist_x = fabs(box_df_counters['left'].iloc[box] - box_df_counters['left'].iloc[box - 1])
        dist_l = (box_df_counters['left'].iloc[box] - box_df_counters['left'].iloc[box - 1]
                    - box_df_counters['width'].iloc[box - 1])
        condition1 = (box_df_counters["line"].iloc[box] == box_df_counters["line"].iloc[box-1])

        condition2 = dist_l > width_mean * k1
        condition3 = box_df_counters["word"].iloc[box - 2] == box_df_counters["word"].iloc[box - 1]

        if condition2&condition1&condition3 == True: # скорее всего буква "ы"
            # рамка от границ предыдущего символа к слежующему
            padding_h = int(box_df_counters['height'].iloc[box - 1]*0.2)
            f_y1 = box_df_counters['top'].iloc[box - 1] - padding_h
            f_y2 = box_df_counters['top'].iloc[box - 1] + box_df_counters['height'].iloc[box - 1] + padding_h
            f_x1 = box_df_counters['left'].iloc[box - 1] + box_df_counters['width'].iloc[box - 1]
            f_x2 = box_df_counters['left'].iloc[box]
            box_i = thresh[f_y1: f_y2, f_x1: f_x2]
            box_s += [[f_x1, f_y1, (f_x2-f_x1), (f_y2-f_y1), f_y2]]
            box_df_l = counters_function(box_i)
            if len(box_df_l) != 0:
                box_df_l = box_df_l[box_df_l["height"] < box_df_counters['height'].iloc[box - 1]*1.1]
                box_df_l = box_df_l[box_df_l["height"] > box_df_counters['height'].iloc[box - 1]*0.9]
                if len(box_df_l) == 1:
                    # буква ы внутри слова - увеличивываем размер бокса
                    box_df_counters['width'].iloc[box - 1] = (box_df_counters['width'].iloc[box - 1]
                                                              + box_df_l['left'].iloc[0] + box_df_l['width'].iloc[0])
                    dist_x = fabs(box_df_counters['left'].iloc[box] - box_df_counters['left'].iloc[box - 1])
                    dist_l = (box_df_counters['left'].iloc[box] - box_df_counters['left'].iloc[box - 1]
                              - box_df_counters['width'].iloc[box - 1])
            else:
                pass

        condition4 = dist_l > width_mean * k2
        condition5 = (box_df_counters["line"].iloc[box] != box_df_counters["line"].iloc[box - 1])
        condition = (condition1 & condition4) | condition5
        if condition:
            print('новое слово')
            # находим координаты слова
            box_df_w = box_df_counters[box_df_counters['word'] == word]
            if len(box_df_w) == 1:
                left = box_df_counters['left'].iloc[box-1]
                top = box_df_counters['top'].iloc[box-1]
                width = box_df_counters['width'].iloc[box-1]
                line = box_df_counters['line'].iloc[box-1]
            else:
                left = np.min(np.array(box_df_w['left']))
                top = np.min(np.array(box_df_w['top']))
                width = np.max(np.array(box_df_w['left'])) + box_df_counters['width'].iloc[box - 1] - left
                height = np.max(np.array(box_df_w['l'])) - np.min(np.array(box_df_w['top']))
                line = box_df_counters['line'].iloc[box - 1]
                print('строка', line)

            box_words += [[left, top, width, height, line, word, dist_l]]
            box_df_counters['dist_w'].iloc[box - 1] = dist_l

            word += 1

        else:
            box_df_counters['dist_l'].iloc[box - 1] = dist_l

        box_df_counters['word'].iloc[box] = word

    box_df_s = pd.DataFrame(box_s, columns=['left', 'top', 'width', 'height', 'l'])
    box_df_words = pd.DataFrame(box_words, columns=['left', 'top', 'width', 'height', 'line', 'word', 'dist_w'])
    print('сохранены таблицы с границами букв и слов. расстояниеми между букв в словах и между слов.')
    box_df_counters.to_csv('result.csv')
    box_df_words.to_csv('words_result.csv')
    return box_df_counters, box_df_words

"""
бн. Подача боксов в Tesseract: классификация содержимого бокса
"""

def char_box(thresh, box_df_counters, height_mean, k=2):
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
        try:
            char_tes = pytesseract.image_to_string(Image.fromarray(box_with_padding), lang='rus')
            print('нашли букву... ',char_tes, type(char_tes))
        except SystemError:
            char_tes = None
        if char_tes in chares:
            box_f = list(box_f.append(pd.Series(char_tes)))
            boxes_final += [box_f]
    box_df_final = pd.DataFrame(boxes_final, columns=['left', 'top', 'width', 'height', 'l',
                                                      'line', 'word', 'dist_l', 'dist_w', 'char'])
    print('box_df_final: ', box_df_final.shape)
    return  box_df_final

"""
7. Выделение слов при помощи Tesseract, расчёт размеров боксов

"""
def read_tesser(im):
    import pytesseract
    from pytesseract import Output
    
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    text = pytesseract.image_to_string(Image.fromarray(gray), lang='rus')
    bb = pytesseract.image_to_boxes(Image.fromarray(gray), lang='rus')
    data = pytesseract.image_to_data(Image.fromarray(gray), lang='rus')
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (xt, yt, wt, ht) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(orig, (xt, yt), (xt + wt, yt + ht), (0, 255, 0), 2)
        substitutions = [
        ('^ *', ''),
        (' *$', ''),  
        (r' *\| *', '|'), 
    ]
    if all(line.lstrip().startswith('|') and line.rstrip().endswith('|') for line in str_input.strip().split('\n')):
        substitutions.extend([
            (r'^\|', ''),  
            (r'\|$', ''),
        ])
    for pattern, replacement in substitutions:
        str_input = re.sub(pattern, replacement, str_input, flags=re.MULTILINE)
    return pd.read_csv(pd.compat.StringIO(str_input), **kwargs)
    bb_df = read_pipe_separated_str(bb, sep = ' ', header = None)
    bb_df.columns = ['char','left','top','width','height','n']
    # Классификатор
    indx_w = 0
    list_s = ['.',',','!','?',';',':','/','|','(',')','[',']','—','{','}']
    list_p = ['и','в','на','a','но','из','без','за','как', 'c', 'от','то','до', 'во']
    index = 0
    c = pd.DataFrame(columns=['index','part','world','w_len','ab_mean_w','ab_mean_c', 'ed_mean_c', 'mean_len_char'])
    # грубое усреднение рас.
    data_df3['mean_len_char'] = data_df2['width']/len(data_df2['text'])

    for i in range(len(data_df3)):  
        part = data_df3['text'].iloc[i]
        print('part:', part)
        w_len = len(part)
        # проверка слова на включение в список предлогов и частиц
        if part.lower() in list_p:
            print('part')
            pass
        else:
            word = ''
            ab_list = []
            ed_list = []
            for char in range(indx_w,indx_w+w_len):
                # проверка символа на включение в список знаков препинания
                if bb_df['char'][char] in list_s:
                    print('прежняя длина строки',data_df3['width'].iloc[i])
                    c_w = bb_df['x2'][char] - bb_df['x1'][char]
                    print('c_w',c_w)
                    print(bb_df['x2'][char], bb_df['x1'][char])
                    # вычитание длины символов пунктуации
                    data_df3['width'].iloc[i] = int(data_df2['width'].iloc[i] - c_w)
                    print('новая длина строки',data_df3['width'].iloc[i])
                else:
                    word += bb_df['char'][char]
                    if char != indx_w+w_len-1:
                        # определяем евклидово рас. между крайней точкой символа и первой точкой последующего  
                        ed = distance.euclidean([bb_df['x2'][char],bb_df['y2'][char]],[bb_df['x1'][char+1],bb_df['y1'][char+1]])
                        # определяем рас. между крайней точкой имвола и первой точкой последующего
                        ab = bb_df['x1'][char+1] - bb_df['x2'][char]
                        ab_list += [ab]
                        ed_list += [ed]
                        print('char', word, 'a-b',ab, 'ed', ed)
                        
            print('world:', word)
            print('w_len:', w_len)
            # ср.арифметичесское рас. на основе общей длины слова за вычитом длины символов пунктуации
            ab_mean_w = data_df3['width'].iloc[i]/len(word)
            height_w = data_df3['height'].iloc[i]
            print('ab_mean_w:', ab_mean_w)
            mean_len_char = data_df3['mean_len_char'].iloc[i]
            if len(word) > 3:
                # забирапем медианное значение рас. для слова в таблицу
                ed_mean_c = np.max(mode(ed_list))
                ab_mean_c = np.max(mode(ab_list))
                print('ab_mean_c:', ab_mean_c)
                print('mean_len_char',mean_len_char)
                # запись данных в таблицу
                c_dict = {'index':[index],'part': part,'world': word,'w_len': w_len,
                          'ab_mean_w': ab_mean_w,'ab_mean_c': ab_mean_c, 'ed_mean_c':ed_mean_c, 
                          'mean_len_char':mean_len_char, 'height':height_w}
                c_list = pd.DataFrame(data=c_dict, index=['index'])
                c = c.append(c_list, sort=False)
                index += 1
        indx_w += w_len 
    
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
        rect1 = patches.Rectangle((left,top),width,height,
                                  linewidth=linewidth,edgecolor= line_color,facecolor='none')
        ax.add_patch(rect1)
    plt.savefig(f'{line_color}_result.png', format='png', dpi=300)
    print('Изображение сохранено локально')

if __name__ == '__main__':
    image_path= '12.jpg'
    im, gray, thresh = preprocess_function(image_path)
    box_df_mser = image_to_mser_data(thresh)
    width_mean, height_mean = mean_box(box_df_mser)
    box_df_counters = counters_function(thresh)
    box_df_counters = filter_boxes(box_df_counters, width_mean, height_mean,k=0.6)
    box_df_counters, box_df_words = words_detection(box_df_counters, width_mean)
    # box_df_final = char_box(thresh, box_df_counters, height_mean)
    plt_boxes(box_df_counters, im, figsize=(15, 15), line_color = 'r', linewidth = 1)
    plt_boxes(box_df_words, im, figsize=(15, 15), line_color='g', linewidth=1)
