import cv2
import mediapipe as mp
import numpy as np

#создаем детектор
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
goal = [[2, 249, 226, 138], [3, 55, 37, 156], [4, 255, 255, 255], [5, 34, 175, 18], [6, 152, 190, 171], [9, 2, 119, 11], [11, 172, 2, 23], [12, 184, 129, 1], [14, 139, 46, 67], [15, 2, 242, 242], [16, 2, 119, 11], [17, 48, 68, 254], [19, 73, 170, 241], [20, 236, 236, 236], [21, 64, 208, 234], [22, 253, 213, 74], [24, 252, 75, 75], [25, 172, 2, 23], [26, 8, 171, 193], [27, 74, 138, 203], [28, 100, 100, 100], [29, 93, 116, 153], [30, 100, 100, 100], [31, 139, 102, 214], [32, 119, 219, 219], [33, 93, 116, 153], [34, 186, 77, 75], [35, 175, 135, 50], [36, 247, 132, 36], [37, 143, 79, 149], [38, 71, 71, 111], [39, 24, 13, 125], [42, 93, 116, 153], [44, 93, 56, 201], [45, 90, 159, 80], [47, 93, 116, 153], [49, 93, 56, 201], [51, 93, 116, 153]]
countries = {'Greece': [249, 226, 138, 0], 'Albania': [55, 37, 156, 1], 'Bulgaria': [34, 175, 18, 3], 'Turkey': [152, 190, 171, 4], 'Italy': [2, 119, 11, 5, 10], 'France': [172, 2, 23, 6, 17], 'Portugal': [184, 129, 1, 7], 'Jugoslavia': [139, 46, 67, 8], 'Spain': [2, 242, 242, 9], 'Switzerland': [48, 68, 254, 11], 'Hungary': [73, 170, 241, 12], 'Austria': [236, 236, 236, 13], 'Romania': [64, 208, 234, 14], 'Luxembourg': [253, 213, 74, 15], 'Czechoslovakia': [252, 75, 75, 16], 'Belgium': [8, 171, 193, 18], 'Hollandia': [74, 138, 203, 19], 'Germany': [100, 100, 100, 20, 22], 'Denmark': [93, 116, 153, 21, 25, 32, 35, 37], 'Poland': [214, 102, 139, 23], 'Litva': [119, 219, 219, 24], 'Latvia': [186, 77, 75, 26], 'Estonia': [175, 135, 50, 27], 'Sweden': [247, 132, 36, 28], 'Finland': [143, 79, 149, 29], 'Norway': [71, 71, 111, 30], 'Soviet': [24, 13, 125, 31], 'Britain': [93, 56, 201, 33, 36], 'Ireland': [90, 159, 80, 34]}
while(cap.isOpened()):
    ret, frame = cap.read()
    a = cv2.imread('europa.png')
    s = a.copy()
    s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(s, 1, 100, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(s, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in goal:
        cv2.fillPoly(a, pts=[contours[i[0]]], color=(i[1], i[2], i[3]))
    # переводим в BGR и показываем результа
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    # переводим его в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    # Распознаем
    results = handsDetector.process(flippedRGB)
    # Рисуем распознанное, если распозналось
    if results.multi_hand_landmarks is not None:
        # нас интересует только подушечка указательного пальца (индекс 8)
        # нужно умножить координаты а размеры картинки
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                flippedRGB.shape[0])
        cv2.circle(a,(x_tip, y_tip), 10, (255, 0, 0), -1)
    cv2.drawContours(a, contours, -1, (0, 0, 0),   1)
    cv2.imshow('image', a)
# освобождаем ресурсы
handsDetector.close()