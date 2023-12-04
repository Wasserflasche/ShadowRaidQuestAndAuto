import cv2
import ctypes
import easyocr
import os
import pyautogui
import time
import sys
import numpy as np
from ctypes import windll
from ctypes import wintypes
from multiprocessing import Process, freeze_support
from pynput import keyboard
from typing import Tuple, List

currentPath = ""
if getattr(sys, 'frozen', False):
    currentPath = os.path.dirname(sys.executable)
else:
    currentPath = os.path.dirname(os.path.realpath(__file__))

#################################################### Help Functions ####################################################### 

def GetColorFromCoords(position: Tuple [int, int]) -> Tuple[int, int, int]:
    return Screenshot().getpixel(position)

def GetCoordsFromDetection(image: np.ndarray, certainity = 0.4) -> Tuple [int, int]:
    image = cv2.resize(image, (0,0), fx=scaleFactor[0], fy=scaleFactor[1])
    result = cv2.matchTemplate(ScreenshotNp(), image, cv2.TM_CCOEFF_NORMED)
    sorted_result_indices = np.argsort(result, axis=None)[::-1]
    max_val_index = sorted_result_indices[0]
    max_loc = np.unravel_index(max_val_index, result.shape)
    result = result[max_loc]
    if result > certainity:
        return (max_loc[1], max_loc[0])
    else:
        return (0, 0)
    
def GetCoordsFromDetectionTwo(image: np.ndarray, certainity = 0.4) -> Tuple [int, int]:
    image = cv2.resize(image, (0,0), fx=scaleFactor[0], fy=scaleFactor[1])
    result = cv2.matchTemplate(ScreenshotNp(), image, cv2.TM_CCOEFF_NORMED)
    sorted_result_indices = np.argsort(result, axis=None)[::-1]
    max_val_index = sorted_result_indices[0]
    second_max_val_index = sorted_result_indices[1]
    max_loc = np.unravel_index(max_val_index, result.shape)
    second_max_loc = np.unravel_index(second_max_val_index, result.shape)
    if result[max_loc] > certainity and result[second_max_loc] > certainity:
        return (max_loc[1], max_loc[0]), (second_max_loc[1], second_max_loc[0])
    else:
        return (0, 0), (0, 0)
    
def GetCoordsFromDetectionFour(image: np.ndarray, certainity = 0.4) -> Tuple [int, int]:
    image = cv2.resize(image, (0,0), fx=scaleFactor[0], fy=scaleFactor[1])
    result = cv2.matchTemplate(ScreenshotNp(), image, cv2.TM_CCOEFF_NORMED)
    sorted_result_indices = np.argsort(result, axis=None)[::-1]
    max_val_index = sorted_result_indices[0]
    second_max_val_index = sorted_result_indices[1]
    third_max_val_index = sorted_result_indices[2]
    fourth_max_val_index = sorted_result_indices[3]
    max_loc = np.unravel_index(max_val_index, result.shape)
    second_max_loc = np.unravel_index(second_max_val_index, result.shape)
    third_max_loc = np.unravel_index(third_max_val_index, result.shape)
    fourth_max_loc = np.unravel_index(fourth_max_val_index, result.shape)
    if result[max_loc] > certainity and result[second_max_loc] > certainity and result[third_max_loc] > certainity and result[fourth_max_loc] > certainity:
        return [(max_loc[1], max_loc[0]), (second_max_loc[1], second_max_loc[0]), (third_max_loc[1], third_max_loc[0]), (fourth_max_loc[1], fourth_max_loc[0])]
    else:
        return [(0, 0), (0, 0), (0, 0), (0, 0)]

def GetCoordsFromLeftHalfOfDetection(image: np.ndarray, certainity = 0.4) -> Tuple [int, int]:
    image = cv2.resize(image, (0,0), fx=scaleFactor[0], fy=scaleFactor[1])
    screenshot = ScreenshotNp()
    screenshot = screenshot[:, :screenshot.shape[1] // 2]
    result = cv2.matchTemplate(screenshot, image, cv2.TM_CCOEFF_NORMED)
    sorted_result_indices = np.argsort(result, axis=None)[::-1]
    coords = (0,0)
    for val_index in sorted_result_indices:
        loc = np.unravel_index(val_index, result.shape)
        if result[loc] > certainity:
            coords = (loc[1], loc[0])
        if coords != (0,0):
            break
    return coords

def GetCoordsFromLeftThirdOfDetectionThree(image: np.ndarray, certainity = 0.4, min_distance = 20) -> List[Tuple[int, int]]:
    image = cv2.resize(image, (0,0), fx=scaleFactor[0], fy=scaleFactor[1])
    screenshot = ScreenshotNp()
    screenshot = screenshot[:, :screenshot.shape[1] // 3]
    result = cv2.matchTemplate(screenshot, image, cv2.TM_CCOEFF_NORMED)
    sorted_result_indices = np.argsort(result, axis=None)[::-1]
    coords = []
    for val_index in sorted_result_indices:
        loc = np.unravel_index(val_index, result.shape)
        if result[loc] > certainity and all(np.linalg.norm(np.array(loc[::-1]) - np.array(c)) > min_distance for c in coords):
            if NoColorInArea(loc, [(0,209,79)], screenshot):
                coords.append((loc[1], loc[0]))
        if len(coords) == 3:
            break
    return coords

def NoColorInArea(coord: Tuple[int, int], colors: List[Tuple[int, int, int]], screenshot: np.ndarray) -> bool:
    screenshot = screenshot[max(0, coord[0] - 10): min(screenshot.shape[1], coord[0] + 10), 
                            max(0, coord[1] - 10): min(screenshot.shape[0], coord[1] + 10)]
    cv2.imwrite(currentPath + "\\Images\\IncreaseChampionLevelThreeTimes\\screenshot.png", screenshot)
    count = 0
    for color in colors:
        lower_bound = (color[2] - 10, color[1] - 10, color[0] - 10)
        if lower_bound[0] < 0:
            lower_bound = (0, lower_bound[1], lower_bound[2])
        if lower_bound[1] < 0:
            lower_bound = (lower_bound[0], 0, lower_bound[2])
        if lower_bound[2] < 0:
            lower_bound = (lower_bound[0], lower_bound[1], 0)
        upper_bound = (color[2] + 10, color[1] + 10, color[0] + 10)
        if upper_bound[0] > 255:
            upper_bound = (255, upper_bound[1], upper_bound[2])
        if upper_bound[1] > 255:
            upper_bound = (upper_bound[0], 255, upper_bound[2])
        if upper_bound[2] > 255:
            upper_bound = (upper_bound[0], upper_bound[1], 255)
        range = cv2.inRange(screenshot, lower_bound, upper_bound)
        count += np.count_nonzero(range)
    if count == 0:
        return True
    else:
        return False

def ReadFromFile(fileName: str) -> str:
    with open(fileName) as file:
        return file.read()

def ScreenshotNp() -> np.ndarray:
    return cv2.cvtColor(np.array(pyautogui.screenshot().convert('RGB')), cv2.COLOR_RGB2BGR)

def Screenshot():
    return pyautogui.screenshot().convert('RGB')

def WriteToFile(fileName: str, input: str):
    with open(fileName, "w") as file:
        file.write(str(input))   
   
#################################################### Config ####################################################### 

screenResolution =  (0,0)
gameResolution = (0,0)
rounds = 0
campaignLevel = 0
teamStrength = 0
with open(os.path.join(currentPath, currentPath + "\\config.txt")) as file:
    for line in file:
        key, value = line.strip().split('=')
        if key == 'screenResolution':
            screenResolution = tuple(map(int, value.split(',')))
        elif key == 'gameResolution':
            gameResolution = tuple(map(int, value.split(',')))
        elif key == 'rounds':
            rounds = int(value)
        elif key == 'campaignLevel':
            kampagneLevel = int(value)

scaleFactor = (gameResolution[0] / screenResolution[0], gameResolution[1] / screenResolution[1])

############################################################################################### Mouse and Keyboard ##############################################################################################################    
      
MOUSEEVENTF_RIGHTDOWN = 0x0008 
MOUSEEVENTF_RIGHTUP = 0x0010 
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
one = 0x31
two = 0x32
three = 0x33
four = 0x34
five = 0x35
six = 0x36
seven = 0x37
e = 0x45
k = 0x4B
q = 0x51
s = 0x53
esc = 0x1B
skills = [three, four, five, six, seven]
user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_UNICODE     = 0x0004
KEYEVENTF_SCANCODE    = 0x0008

MAPVK_VK_TO_VSC = 0

wintypes.ULONG_PTR = wintypes.WPARAM

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))

LPINPUT = ctypes.POINTER(INPUT)

def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args

user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT, # nInputs
                             LPINPUT,       # pInputs
                             ctypes.c_int)  # cbSize  
  
def MouseClick():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    InputSleep()
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    InputSleep()

def MouseDrag():
    startPosition = (screenResolution[0] // 2, screenResolution[1] // 2)
    endPosition = (startPosition[0] - 600, startPosition[1])
    MoveMouseTo(startPosition)
    InputSleep()
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    InputSleep()
    MoveMouseTo(endPosition)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    InputSleep()  

def MoveMouseTo(position):
    x_normalized = int(65535 * (position[0] / screenResolution[0]))
    y_normalized = int(65535 * (position[1] / screenResolution[1]))
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, x_normalized, y_normalized, 0, 0)   
    InputSleep()
   
def MouseRightDown():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    InputSleep()
    
def MouseRightUp():   
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    InputSleep()
   
def MouseMove():
    windll.user32.mouse_event(1, 100, 0, 0, 0)
    InputSleep()
    windll.user32.mouse_event(1, 100, 0, 0, 0)
    InputSleep()
    windll.user32.mouse_event(1, 100, 0, 0, 0)
   
def MouseScroll(amount):
    MOUSEEVENTF_WHEEL = 0x0800
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, amount, 0)

def KeyDown(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
    time.sleep(0.035)
   
def PressKey(hexKeyCode):
    KeyDown(hexKeyCode)
    ReleaseKey(hexKeyCode)

def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
    time.sleep(0.035)
    
def On_press(key):
    if any([key in z for z in [{keyboard.KeyCode(char='+')}]]):
        if ReadFromFile(currentPath + "\\stop.txt") == "0":
            WriteToFile(currentPath + "\\stop.txt", "1")
            print("Bot stopped")
        else:
            WriteToFile(currentPath + "\\stop.txt", "0")    
            print("Bot started")  
    if any([key in z for z in [{keyboard.KeyCode(char='#')}]]):
        if ReadFromFile(currentPath + "\\stop.txt") == "2":
            WriteToFile(currentPath + "\\stop.txt", "1")
            print("Daily quests stopped")
        else:
            WriteToFile(currentPath + "\\stop.txt", "2")    
            print("Daily quests started") 
 
def InputSleep():
    time.sleep(0.02) 
     
############################################################################################### Main Functions #####################################################################################################

def BuyItemFromMarket():
    CheckEscape()
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\BuyItemFromMarket\\marketFhd.png"))
    coords = (coords[0] + 30, coords[1] + 50)
    MoveMouseTo(coords)
    MouseClick()
    time.sleep(2)
    coords = FindCheapestItem()
    if coords != (0,0):
        MoveMouseTo(coords)
        MouseClick()
        time.sleep(0.5)
        ClickImage("\\Images\\BuyItemFromMarket\\buyItemFhd.png")
        PressKey(esc)

def CompleteDailyQuests():
    CheckEscape()
    ClickImage("\\Images\\CompleteDailyQuests\\questsFhd.png")
    while True:
        coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\CompleteDailyQuests\\claimRewardFhd.png"), 0.6)
        if coords != (0,0):
            MoveMouseTo(coords)
            MouseClick()
            time.sleep(1)
        else:
            levelUpCoords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\levelUpFhd.png"), 0.6)
            if levelUpCoords != (0,0):
                PressKey(esc)
            else:
                escapeCoords2 = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\escape2Fhd.png"), 0.7)
                if escapeCoords2 != (0,0):
                    PressKey(esc)
                else:
                    break
    PressKey(esc)

def CheckEscape():
    while True:
        escapeCoords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\escapeFhd.png"), 0.6)
        if escapeCoords != (0,0):
            PressKey(esc)
            time.sleep(1)
            coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\attentionOfferFhd.png"), 0.6)
            if coords != (0,0):
                MoveMouseTo(coords)
                MouseClick()
                time.sleep(1)
        else:
            escapeCoords2 = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\escape2Fhd.png"), 0.6)
            if escapeCoords2 != (0,0):
                PressKey(esc)
                time.sleep(1)
            else:
                time.sleep(1)
                break

def ClickImage(path:str):
    coords = GetCoordsFromDetection(cv2.imread(currentPath + path))
    MoveMouseTo(coords)
    MouseClick()
    time.sleep(0.5)

def DailyQuests():
    DefeatTenBosses()
    UpgradeArtifactFourTimes()
    SummonThreeChampions()
    BuyItemFromMarket()
    IncreaseChampionLevelThreeTimes()
    FightFiveTimesInArena()
    CompleteDailyQuests()
    print("Bot finished")
    WriteToFile(currentPath + "\\stop.txt", "1")

def DefeatTenBosses():
    CheckEscape()
    ClickImage("\\Images\\DefeatTenBosses\\battleFhd.png")
    time.sleep(0.5)
    ClickImage("\\Images\\DefeatTenBosses\\campaignFhd.png")
    time.sleep(0.5)
    image = cv2.imread(currentPath + "\\Images\\DefeatTenBosses\\campaignOneFhd.png")
    if campaignLevel == 2:
        image = cv2.imread(currentPath + "\\Images\\DefeatTenBosses\\kampagneTwoFhd.png")
    elif campaignLevel == 3:
        image = cv2.imread(currentPath + "\\Images\\DefeatTenBosses\\kampagneThreeFhd.png")
    elif campaignLevel == 4:
        image = cv2.imread(currentPath + "\\Images\\DefeatTenBosses\\kampagneFourFhd.png")
        MouseDrag()
    time.sleep(0.5)
    coords = GetCoordsFromDetection(image)
    MoveMouseTo(coords)
    MouseClick()
    time.sleep(0.5)
    ClickImage("\\Images\\DefeatTenBosses\\bossBattleFhd.png")
    ClickImage("\\Images\\DefeatTenBosses\\startBossBattleFhd.png")
    counter = 1
    print("Boss round " + str(counter) + " of 10 rounds")
    while counter < 10:
        if DetectReplayButton():
            time.sleep(2)
            coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\DefeatTenBosses\\infoFhd.png"), 0.6)
            if coords != (0,0):
                counter += 1
                print("Boss round " + str(counter) + " of 10 rounds")
    coords = (0,0)
    while coords == (0,0):
        time.sleep(2)
        coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\replayKampagneFhd.png"), 0.6)
    PressKey(esc)
    PressKey(esc)
    PressKey(esc)
    PressKey(esc)

def DetectReplayButton() -> bool:
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\replayKampagneFhd.png"), 0.6)
    if coords != (0,0):
        MoveMouseTo(coords)
        MouseClick()
        time.sleep(1)
        return True
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\replayDungeonFhd.png"), 0.6)
    if coords != (0,0):
        MoveMouseTo(coords)
        MouseClick()
        time.sleep(1)
        return True
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\replayFractionFhd.png"), 0.6)
    if coords != (0,0):
        MoveMouseTo(coords)
        MouseClick()
        time.sleep(1)
        return True
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\attentionOfferFhd.png"), 0.7)
    if coords != (0,0):
        MoveMouseTo(coords)
        MouseClick()
    escapeCoords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\escapeFhd.png"), 0.7)
    if escapeCoords != (0,0):
        PressKey(esc)
    else:
        levelUpCoords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\levelUpFhd.png"), 0.6)
        if levelUpCoords != (0,0):
            PressKey(esc)
        else:
            escapeCoords2 = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\escape2Fhd.png"), 0.7)
            if escapeCoords2 != (0,0):
                PressKey(esc)
    return False

def FightFiveTimesInArena():
    CheckEscape()
    ClickImage("\\Images\\FightFiveTimesInArena\\battleFhd.png")
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\FightFiveTimesInArena\\arenaFhd.png"))
    if coords != (0,0):
        MoveMouseTo(coords)
        MouseClick()
        time.sleep(1)
        ClickImage("\\Images\\FightFiveTimesInArena\\arenaEnterFhd.png")
        for i in range(5):
            coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\FightFiveTimesInArena\\oponentPositionFhd.png"))
            MoveMouseTo(coords)
            coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\FightFiveTimesInArena\\selectOponentFhd.png"), 0.6)
            while coords == (0,0):
                for i in range(5):
                    MouseScroll(-10)
                    time.sleep(0.1)
                coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\FightFiveTimesInArena\\selectOponentFhd.png"), 0.6)
            ClickImage("\\Images\\FightFiveTimesInArena\\selectOponentFhd.png")
            ClickImage("\\Images\\FightFiveTimesInArena\\startBattleFhd.png")
            time.sleep(3)
            ClickImage("\\Images\\FightFiveTimesInArena\\autoFhd.png")
            coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\FightFiveTimesInArena\\battleFinishedFhd.png"), 0.6)
            while coords == (0,0):
                coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\FightFiveTimesInArena\\battleFinishedFhd.png"), 0.6)
                time.sleep(0.5)
            PressKey(esc)
        PressKey(esc)
        time.sleep(0.5)
        PressKey(esc)
        time.sleep(0.5)
        PressKey(esc)

def FindArmorCoords() -> Tuple[int, int]:
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\UpgradeArtifactFourTimes\\bootsFhd.png"), 0.6)
    if coords != (0,0):
        return coords
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\UpgradeArtifactFourTimes\\chestPlateFhd.png"), 0.6)
    if coords != (0,0):
        return coords
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\UpgradeArtifactFourTimes\\gauntletsFhd.png"), 0.6)
    if coords != (0,0):
        return coords
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\UpgradeArtifactFourTimes\\helmetFhd.png"), 0.6)
    if coords != (0,0):
        return coords
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\UpgradeArtifactFourTimes\\shieldFhd.png"), 0.6)
    if coords != (0,0):
        return coords
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\UpgradeArtifactFourTimes\\swordFhd.png"), 0.6)
    if coords != (0,0):
        return coords
    return (0,0)

def FindCheapestItem() -> Tuple[int, int]:
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\BuyItemFromMarket\\energyFhd.png"))
    screenshot = ScreenshotNp()
    screenshot = screenshot[coords[1] + 100:, :]
    cv2.imwrite(currentPath + "\\Images\\BuyItemFromMarket\\screenshot.png", screenshot)
    return RecognizeLowestNumberCoords(coords[1] + 100)

def FindChampionClosestToEmptySlot() -> Tuple[int, int]:
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\IncreaseChampionLevelThreeTimes\\scrollDownFhd.png"))
    MoveMouseTo((coords[0], coords[1] + 100))
    ScrollToEmptySlot()
    coords = GetCoordsFromLeftHalfOfDetection(cv2.imread(currentPath + "\\Images\\IncreaseChampionLevelThreeTimes\\starFhd.png"), 0.7)
    count = 0
    while coords == (0,0):
        count += 1
        coords = GetCoordsFromLeftHalfOfDetection(cv2.imread(currentPath + "\\Images\\IncreaseChampionLevelThreeTimes\\starFhd.png"), 0.7 - count * 0.01)
    return (coords[0] + 30, coords[1] + 30)

def FindThreeGrayChampions() -> Tuple[int, int]:
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\IncreaseChampionLevelThreeTimes\\scrollDown2Fhd.png"))
    MoveMouseTo((coords[0], coords[1] + 100))
    ScrollToEmptySlot()
    coords = GetCoordsFromLeftThirdOfDetectionThree(cv2.imread(currentPath + "\\Images\\IncreaseChampionLevelThreeTimes\\starFhd.png"), 0.6)
    return coords

def IncreaseChampionLevelThreeTimes():
    CheckEscape()
    ClickImage("\\Images\\IncreaseChampionLevelThreeTimes\\selectChampionsFhd.png")
    time.sleep(1)
    coords = FindChampionClosestToEmptySlot()
    MoveMouseTo(coords)
    MouseClick()
    time.sleep(0.5)
    ClickImage("\\Images\\IncreaseChampionLevelThreeTimes\\tavernFhd.png")
    time.sleep(2)
    coords = FindThreeGrayChampions()
    for coord in coords:
        MoveMouseTo(coord)
        MouseClick()
        time.sleep(0.1)
    ClickImage("\\Images\\IncreaseChampionLevelThreeTimes\\upgradeFhd.png")
    time.sleep(0.5)
    PressKey(esc)

def RecognizeLowestNumberCoords(offset:int) -> Tuple[int, int]:
    reader = easyocr.Reader(['en'])
    results = reader.readtext(currentPath + "\\Images\\BuyItemFromMarket\\screenshot.png")
    numbers = []
    for result in results:
        text = result[1].replace(',', '')
        if text.isdigit() and int(text) > 1000:
            numbers.append((int(text), result[0]))
    smallestNumberCoords = None
    if numbers:
        smallestNumberCoords = min(numbers, key=lambda x: x[0])[1]
        return (smallestNumberCoords[0][0], smallestNumberCoords[0][1] + offset)
    else:
        return (0,0)

def ScrollToEmptySlot():
    emptySlotCoords = (0,0)
    while emptySlotCoords == (0,0):
        for i in range(5):
            MouseScroll(-10)
            time.sleep(0.1)
        emptySlotCoords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\IncreaseChampionLevelThreeTimes\\emptySlotFhd.png"), 0.7)

def ShadowRaidLegendsAuto():
    print("Bot loaded")
    WriteToFile(currentPath + "\\stop.txt", "1")
    counter:int = 0
    while True:
        if ReadFromFile(currentPath + "\\stop.txt") == "0":
            if DetectReplayButton():
                counter += 1
                print("Bot started round " + str(counter) + " of " + str(rounds) + " rounds")
            if counter == rounds:
                print("Bot finished")
                WriteToFile(currentPath + "\\stop.txt", "1")
                counter = 0
            time.sleep(0.1)
        else:
            if ReadFromFile(currentPath + "\\stop.txt") == "2":
                DailyQuests()
            else:
                InputSleep()    

def SortOutCoordWithHighestX(coords: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    highestX = coords[0]
    for coord in coords:
        if coord[0] > highestX[0]:
            highestX = coord
    coords.remove(highestX)
    return coords

def SummonThreeChampions():
    CheckEscape()
    time.sleep(10)
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\SummonThreeChampions\\portalFhd.png"))
    MoveMouseTo((coords[0] + 30, coords[1] + 30))
    MouseClick()
    ClickImage("\\Images\\SummonThreeChampions\\portalFhd.png")
    ClickImage("\\Images\\SummonThreeChampions\\summonChampFhd.png")
    time.sleep(10)
    for i in range(4):
        ClickImage("\\Images\\SummonThreeChampions\\summonChampAggainFhd.png")
        time.sleep(6.5)  
    PressKey(esc)

def UpgradeArtifactFourTimes():
    CheckEscape()
    ClickImage("\\Images\\UpgradeArtifactFourTimes\\selectChampionsFhd.png") 
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\UpgradeArtifactFourTimes\\skinFhd.png"), 0.6)
    coords = (coords[0] + int(281 * scaleFactor[0]), coords[1] - int(210 * scaleFactor[1]))
    MoveMouseTo(coords)
    MouseClick()
    time.sleep(0.5)
    coords = FindArmorCoords()
    MoveMouseTo(coords)
    MouseClick()
    time.sleep(0.5)
    coord = (0,0)
    coords = [*GetCoordsFromDetectionFour(cv2.imread(currentPath + "\\Images\\UpgradeArtifactFourTimes\\upgradeFhd.png"))]
    coord = min(coords, key=lambda c: (c[0], -c[1]))
    MoveMouseTo(coord)
    MouseClick()
    time.sleep(0.5)
    coords = GetCoordsFromDetection(cv2.imread(currentPath + "\\Images\\UpgradeArtifactFourTimes\\upgradeStartFhd.png"))
    MoveMouseTo(coords)
    for i in range(4):
        MouseClick()
        time.sleep(3)
    PressKey(esc)

############################################################################################### Start ##############################################################################################################      
       
if __name__ == '__main__':    
    freeze_support()
    processBot = Process(target = ShadowRaidLegendsAuto)
    processListener = keyboard.Listener(on_press=On_press)
    processBot.start()
    processListener.start()
    processBot.join()
    processListener.join()