PIXEL = 21
ROW, COL = 20, 10
MATRIX_WIDTH, MATRIX_HEIGHT = COL * PIXEL, ROW * PIXEL
FPS = 30

SIDEBAR_WIDTH = MATRIX_WIDTH / 2
PREVIEW_HEIGHT = 0.6
PREVIEW_TRAINING_HEIGHT = 1
SCOREBAR_HEIGHT = 1 - PREVIEW_HEIGHT

WINDOW_WIDTH = MATRIX_WIDTH + SIDEBAR_WIDTH + PIXEL * 3
WINDOW_HEIGHT = MATRIX_HEIGHT + PIXEL * 2

FALL_SPEED = 120
ROTATE_WAIT_TIME = 50
MOVE_WAIT_TIME = 30

IMG_DIR = "custom_env/game/assets/img"
BLOCK_IMG_DIR = IMG_DIR + "/block"
TETROMINOS_IMG_DIR = IMG_DIR + "/tetrominos"

TETROMINOS = {
    "Z": {
        "id": 1,
        "shape": [(0, 0), (0, -1), (1, 0), (-1, -1)],
        "color": "#FF6347",
        "image": "Z.png",
        "type": [1, 0, 0, 0, 0, 0, 0],
    },
    "S": {
        "id": 2,
        "shape": [(0, 0), (0, -1), (1, -1), (-1, 0)],
        "color": "#00FF7F",
        "image": "S.png",
        "type": [0, 1, 0, 0, 0, 0, 0],
    },
    "O": {
        "id": 3,
        "shape": [(1, -1), (1, 0), (0, -1), (0, 0)],
        "color": "#FFFF00",
        "image": "O.png",
        "type": [0, 0, 1, 0, 0, 0, 0],
    },
    "L": {
        "id": 4,
        "shape": [(0, 0), (-1, 0), (1, 0), (1, -1)],
        "color": "#FFA500",
        "image": "L.png",
        "type": [0, 0, 0, 1, 0, 0, 0],
    },
    "J": {
        "id": 5,
        "shape": [(0, 0), (-1, 0), (1, 0), (-1, -1)],
        "color": "#aabbff",
        "image": "J.png",
        "type": [0, 0, 0, 0, 1, 0, 0],
    },
    "I": {
        "id": 6,
        "shape": [(1, -1), (-1, -1), (0, -1), (2, -1)],
        "color": "#00FFFF",
        "image": "I.png",
        "type": [0, 0, 0, 0, 0, 1, 0],
    },
    "T": {
        "id": 7,
        "shape": [(0, 0), (-1, 0), (0, -1), (1, 0)],
        "color": "#ffbbaa",
        "image": "T.png",
        "type": [0, 0, 0, 0, 0, 0, 1],
    },
}

CLEAR_REWARDS = {1: 100, 2: 300, 3: 500, 4: 800}
