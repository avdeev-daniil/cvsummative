import cv2
import mediapipe as mp
import numpy as np
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Checking if a point is inside a polygon
def point_in_polygon(point, polygon):
    num_vertices = len(polygon)
    x, y = point.x, point.y
    inside = False

    # Store the first point in the polygon and initialize the second point
    p1 = polygon[0]

    # Loop through each edge in the polygon
    for i in range(1, num_vertices + 1):
        # Get the next point in the polygon
        p2 = polygon[i % num_vertices]

        # Check if the point is above the minimum y coordinate of the edge
        if y > min(p1.y, p2.y):
            # Check if the point is below the maximum y coordinate of the edge
            if y <= max(p1.y, p2.y):
                # Check if the point is to the left of the maximum x coordinate of the edge
                if x <= max(p1.x, p2.x):
                    # Calculate the x-intersection of the line connecting the point to the edge
                    x_intersection = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x

                    # Check if the point is on the same line as the edge or to the left of the x-intersection
                    if p1.x == p2.x or x <= x_intersection:
                        # Flip the inside flag
                        inside = not inside

        # Store the current point as the first point for the next iteration
        p1 = p2

    # Return the value of the inside flag
    return inside
#создаем детектор
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
goal = [[2, 249, 226, 138], [3, 55, 37, 156], [4, 255, 255, 255], [5, 34, 175, 18], [6, 152, 190, 171], [9, 2, 119, 11], [11, 172, 2, 23], [12, 184, 129, 1], [14, 139, 46, 67], [15, 2, 242, 242], [16, 2, 119, 11], [17, 48, 68, 254], [19, 73, 170, 241], [20, 236, 236, 236], [21, 64, 208, 234], [22, 253, 213, 74], [24, 252, 75, 75], [25, 172, 2, 23], [26, 8, 171, 193], [27, 74, 138, 203], [28, 100, 100, 100], [29, 93, 116, 153], [30, 100, 100, 100], [31, 139, 102, 214], [32, 119, 219, 219], [33, 93, 116, 153], [34, 186, 77, 75], [35, 175, 135, 50], [36, 247, 132, 36], [37, 143, 79, 149], [38, 71, 71, 111], [39, 24, 13, 125], [42, 93, 116, 153], [44, 93, 56, 201], [45, 90, 159, 80], [47, 93, 116, 153], [49, 93, 56, 201], [51, 93, 116, 153]]
countries = {'greece': [249, 226, 138, 0], 'albania': [55, 37, 156, 1], 'bulgaria': [34, 175, 18, 3], 'turkey': [152, 190, 171, 4], 'italy': [2, 119, 11, 5, 10], 'france': [172, 2, 23, 6, 17], 'portugal': [184, 129, 1, 7], 'jugoslavia': [139, 46, 67, 8], 'spain': [2, 242, 242, 9], 'switzerland': [48, 68, 254, 11], 'hungary': [73, 170, 241, 12], 'austria': [236, 236, 236, 13], 'romania': [64, 208, 234, 14], 'luxembourg': [253, 213, 74, 15], 'czechoslovakia': [252, 75, 75, 16], 'belgium': [8, 171, 193, 18], 'hollandia': [74, 138, 203, 19], 'germany': [100, 100, 100, 20, 22], 'denmark': [93, 116, 153, 21, 25, 32, 35, 37], 'poland': [214, 102, 139, 23], 'litva': [119, 219, 219, 24], 'latvia': [186, 77, 75, 26], 'estonia': [175, 135, 50, 27], 'sweden': [247, 132, 36, 28], 'finland': [143, 79, 149, 29], 'norway': [71, 71, 111, 30], 'soviet': [24, 13, 125, 31], 'britain': [93, 56, 201, 33, 36], 'ireland': [90, 159, 80, 34]}
polygons = [([Point(123, 233), Point(183, 219), Point(218, 247), Point(200, 273), Point(200, 305), Point(164, 312), Point(127, 294), Point(142, 264)], 'france'),
            ([Point(215, 315), Point(220, 322), Point(208, 323), Point(216, 315)], 'france'),
            ([Point(62, 293), Point(81, 299), Point(52, 350), Point(38, 345), Point(39, 326)], 'portugal'),
            ([Point(70, 273), Point(66, 291), Point(81, 299), Point(52, 350), Point(64, 362), Point(115, 362), Point(127, 352), Point(131, 336), Point(167, 317)], 'spain'),
            ([Point(208, 232), Point(203, 231), Point(201, 235), Point(208, 237)], 'luxembourg'),
            ([Point(124, 212), Point(174, 215), Point(183, 196), Point(168, 156), Point(176, 137), Point(170, 119), Point(153, 123), Point(146, 147), Point(152, 180), Point(136, 192), Point(141, 202)], 'britain'),
            ([Point(129, 148), Point(107, 158), Point(95, 180), Point(112, 188), Point(125, 186), Point(135, 167), Point(127, 162), Point(133, 151)], 'ireland'),
            ([Point(135, 152), Point(143, 159), Point(135, 168), Point(128, 161)], 'britain'),
            ([Point(214, 333), Point(221, 335), Point(220, 360), Point(208, 357), Point(208, 338)], 'italy'),
            ([Point(207, 304), Point(222, 299), Point(278, 362), Point(289, 362), Point(301, 355), Point(253, 298), Point(267, 297), Point(268, 282), Point(253, 278), Point(238, 272), Point(230, 283), Point(205, 282)], 'italy'),
            ([Point(201, 273), Point(204, 279), Point(211, 277), Point(219, 282), Point(224, 279), Point(228, 281), Point(237, 273), Point(229, 263), Point(223, 262)], 'switzerland'),
            ([Point(651, 318), Point(628, 313), Point(633, 292), Point(612, 296), Point(600, 288), Point(609, 278), Point(600, 274), Point(570, 261), Point(596, 245), Point(573, 230), Point(540, 262), Point(582, 297), Point(578, 317), Point(567, 307), Point(555, 323), Point(519, 303), Point(508, 304), Point(450, 280), Point(430, 292), Point(402, 272), Point(388, 253), Point(370, 247), Point(376, 219), Point(375, 166), Point(367, 147), Point(366, 130), Point(391, 99), Point(361, 1), Point(651, 1)], 'soviet'),
            ([Point(204, 232), Point(200, 234), Point(190, 225), Point(186, 218), Point(199, 212), Point(210, 226), Point(208, 231)], 'belgium'),
            ([Point(209, 223), Point(199, 213), Point(204, 201), Point(225, 196), Point(209, 223)], 'hollandia'),
            ([Point(227, 198), Point(211, 224), Point(209, 241), Point(220, 248), Point(213, 263), Point(221, 260), Point(213, 263), Point(247, 263), Point(256, 267), Point(264, 250), Point(254, 240), Point(213, 263), Point(257, 234), Point(278, 228), Point(305, 240), Point(304, 231), Point(287, 214), Point(213, 263), Point(301, 186), Point(274, 197), Point(244, 184), Point(242, 177), Point(237, 179), Point(232, 197)], 'germany'),
            ([Point(236, 272), Point(248, 273), Point(254, 279), Point(276, 279), Point(295, 260), Point(287, 254), Point(265, 255), Point(257, 262), Point(247, 264), Point(233, 267)], 'austria'),
            ([Point(239, 177), Point(253, 166), Point(247, 156), Point(234, 164), Point(234, 171)], 'denmark'),
            ([Point(96, 1), Point(93, 44), Point(156, 44), Point(156, 1)], 'denmark'),
            ([Point(249, 175), Point(253, 179), Point(249, 182), Point(245, 179)], 'denmark'),
            ([Point(259, 169), Point(265, 175), Point(260, 181), Point(254, 176)], 'denmark'),
            ([Point(250, 154), Point(254, 150), Point(250, 145), Point(246, 150)], 'denmark'),
            ([Point(322, 178), Point(346, 183), Point(349, 191), Point(340, 206), Point(332, 205), Point(325, 203), Point(317, 207), Point(313, 201), Point(304, 201), Point(308, 201)], 'germany'),
            ([Point(255, 243), Point(266, 254), Point(288, 255), Point(295, 260), Point(312, 258), Point(318, 256), Point(341, 256), Point(353, 255), Point(344, 250), Point(327, 247), Point(306, 247), Point(304, 239), Point(297, 237), Point(277, 229), Point(264, 234), Point(257, 234)], 'czechoslovakia'),
            ([Point(281, 278), Point(296, 260), Point(311, 266), Point(317, 256), Point(342, 256), Point(321, 283), Point(314, 281), Point(302, 290)], 'hungary'),
            ([Point(268, 282), Point(276, 308), Point(285, 317), Point(304, 324), Point(312, 332), Point(318, 327), Point(321, 330), Point(323, 344), Point(326, 346), Point(343, 339), Point(340, 324), Point(343, 311), Point(336, 298), Point(323, 344), Point(330, 298), Point(322, 284), Point(314, 284), Point(323, 344), Point(301, 289), Point(280, 280)], 'jugoslavia'),
            ([Point(312, 331), Point(312, 358), Point(318, 361), Point(327, 347), Point(321, 341), Point(321, 330), Point(318, 328)], 'albania'),
            ([Point(317, 363), Point(344, 363), Point(341, 352), Point(350, 351), Point(356, 345), Point(370, 343), Point(382, 329), Point(375, 330), Point(368, 335), Point(352, 335), Point(339, 340), Point(326, 346)], 'greece'),
            ([Point(381, 362), Point(376, 354), Point(401, 341), Point(395, 336), Point(378, 346), Point(370, 344), Point(381, 334), Point(385, 326), Point(408, 334), Point(417, 331), Point(432, 318), Point(457, 318), Point(480, 322), Point(486, 317), Point(497, 316), Point(507, 304), Point(518, 303), Point(529, 313), Point(534, 314), Point(538, 331), Point(557, 363)], 'turkey'),
            ([Point(342, 310), Point(343, 322), Point(340, 329), Point(346, 337), Point(356, 333), Point(368, 334), Point(385, 325), Point(387, 315), Point(394, 307), Point(379, 303), Point(366, 309)], 'bulgaria'),
            ([Point(343, 308), Point(367, 309), Point(380, 302), Point(391, 304), Point(391, 296), Point(397, 291), Point(402, 272), Point(393, 256), Point(380, 249), Point(370, 248), Point(367, 245), Point(359, 245), Point(351, 257), Point(341, 257), Point(321, 284), Point(331, 297), Point(338, 300)], 'romania'),
            ([Point(307, 247), Point(328, 247), Point(352, 253), Point(360, 246), Point(368, 235), Point(376, 218), Point(369, 208), Point(376, 170), Point(366, 167), Point(355, 191), Point(347, 192), Point(339, 207), Point(315, 207), Point(304, 201), Point(308, 191), Point(301, 186), Point(296, 193), Point(296, 203), Point(285, 212), Point(299, 222), Point(307, 232)], 'poland'),
            ([Point(355, 191), Point(366, 167), Point(361, 163), Point(350, 162), Point(332, 164), Point(320, 173), Point(321, 179), Point(346, 183), Point(348, 190)], 'litva')]
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
        for polygon in polygons:
            if point_in_polygon(Point(x_tip, y_tip), polygon[0]):
                print(polygon[1])
    cv2.drawContours(a, contours, -1, (0, 0, 0),   1)
    cv2.imshow('image', a)
# освобождаем ресурсы
handsDetector.close()