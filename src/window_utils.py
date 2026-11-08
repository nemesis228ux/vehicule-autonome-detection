"""Utilitaires de dimensionnement et de fermeture des fenêtres OpenCV."""

from __future__ import annotations

import ctypes
import sys

import cv2

FALLBACK_SCREEN_WIDTH = 1280
FALLBACK_SCREEN_HEIGHT = 720
SCREEN_MARGIN_WIDTH = 40
SCREEN_MARGIN_HEIGHT = 90
MAX_UPSCALE = 1.0


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def get_usable_screen_size() -> tuple[int, int]:
    """Renvoie la zone écran utilisable, marges de fenêtre comprises."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        user32.SystemParametersInfoW.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        user32.SystemParametersInfoW.restype = ctypes.c_bool
        rect = _Rect()
        if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            width = rect.right - rect.left - SCREEN_MARGIN_WIDTH
            height = rect.bottom - rect.top - SCREEN_MARGIN_HEIGHT
            if width > 0 and height > 0:
                return width, height

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        width = root.winfo_screenwidth() - SCREEN_MARGIN_WIDTH
        height = root.winfo_screenheight() - SCREEN_MARGIN_HEIGHT
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    return FALLBACK_SCREEN_WIDTH, FALLBACK_SCREEN_HEIGHT


def compute_window_size(
    width: int,
    height: int,
    max_width: int = FALLBACK_SCREEN_WIDTH,
    max_height: int = FALLBACK_SCREEN_HEIGHT,
    max_upscale: float = MAX_UPSCALE,
) -> tuple[int, int]:
    """Calcule une taille de fenêtre qui conserve le ratio de l'image.

    L'image n'est jamais agrandie au-delà de sa résolution d'origine, et elle
    n'est réduite que si elle dépasse la zone écran disponible.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Les dimensions de l'image doivent être positives.")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("Les dimensions maximales doivent être positives.")
    if max_upscale <= 0:
        raise ValueError("max_upscale doit être strictement positif.")

    scale = min(max_width / width, max_height / height, max_upscale)
    return max(1, round(width * scale)), max(1, round(height * scale))


def fit_window(window: str, width: int, height: int) -> None:
    """Dimensionne la fenêtre à la vidéo sans dépasser l'écran."""
    max_width, max_height = get_usable_screen_size()
    display_width, display_height = compute_window_size(
        width, height, max_width, max_height
    )
    cv2.resizeWindow(window, display_width, display_height)


def is_window_open(window: str) -> bool:
    """Indique si la fenêtre existe encore (fermeture par la croix détectée)."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
        user32.IsWindowVisible.restype = ctypes.c_bool
        handle = user32.FindWindowW(None, window)
        return bool(handle) and bool(user32.IsWindowVisible(handle))

    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False
