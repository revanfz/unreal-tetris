PIXEL = 28
ROW, COL = 20, 10
MATRIX_WIDTH, MATRIX_HEIGHT = COL * PIXEL, ROW * PIXEL
FPS = 30

SIDEBAR_WIDTH = 200
PREVIEW_HEIGHT = 0.7
SCOREBAR_HEIGHT = 1 - PREVIEW_HEIGHT

WINDOW_WIDTH = MATRIX_WIDTH + SIDEBAR_WIDTH + PIXEL * 3
WINDOW_HEIGHT = MATRIX_HEIGHT + PIXEL * 2

UPDATE_START_SPEED = 210

IMAGE_PATH = "assets/img/"

TETROMINOS = {
    'T': {'shape': [(0, 0), (-1, 0), (0, -1), (1, 0)], 'color': '#A020F0', 'image': 'T.png'},
    'Z': {'shape': [(0, 0), (1, 0), (1, 1), (1, 2)], 'color': '#D30000', 'image': 'Z.png'},
    'S': {'shape': [(0, 0), (1, 0), (-1, 1), (0, 1)], 'color': '#1BFC06', 'image': 'S.png'},
    'O': {'shape': [(0, 0), (1, 0), (0, 1), (1, 1)], 'color': '#FBF719', 'image': 'O.png'},
    'L': {'shape': [(0, 0), (1, 0), (2, 0), (2, -1)], 'color': '#FF793B', 'image': 'L.png'},
    'J': {'shape': [(0, 0), (1, 0), (2, 0), (0, -1)], 'color': '#192586', 'image': 'J.png'},
    'I': {'shape': [(0, 0), (1, 0), (2, 0), (3, 0)], 'color': '#01FFFF', 'image': 'I.png'}
}