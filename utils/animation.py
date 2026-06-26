import os
import random
import time


RESET = "\033[0m"


def rgb_fg(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def clear_screen():
    os.system("cls")


def animated_print(text, delay=0.02):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def get_logo_lines():
    font = {
        "T": ["█████", "  █  ", "  █  ", "  █  ", "  █  ", "  █  ", "  █  "],
        "P": ["████ ", "█   █", "█   █", "████ ", "█    ", "█    ", "█    "],
        "M": ["█     █", "██   ██", "█ █ █ █", "█  █  █", "█     █", "█     █", "█     █"],
        "L": ["█    ", "█    ", "█    ", "█    ", "█    ", "█    ", "█████"],
        " ": ["   ", "   ", "   ", "   ", "   ", "   ", "   "],
    }
    palabra = "TP ML"
    return [" ".join(font[letra][fila] for letra in palabra) for fila in range(7)]


def show_welcome():
    if os.name == "nt":
        os.system("")

    clear_screen()

    logo = get_logo_lines()
    caracteres_ruido = "░▒▓█"
    niveles_relleno = ["░", "▒", "▓", "█"]

    color_top = (138, 99, 210)
    color_bottom = (217, 119, 87)

    def fila_color(indice_fila, total_filas):
        t = indice_fila / (total_filas - 1) if total_filas > 1 else 0
        return lerp_color(color_top, color_bottom, t)

    for _ in range(3):
        clear_screen()
        print()
        for linea in logo:
            gris = random.randint(60, 150)
            pieza = rgb_fg(gris, gris, gris)
            ruido = "".join(
                random.choice(caracteres_ruido) if c == "█" else " " for c in linea
            )
            print("    " + pieza + ruido + RESET)
        print("\n    Iniciando TP ML...")
        time.sleep(0.08)

    intensidades = [0.35, 0.55, 0.75, 1.0]
    for nivel, intensidad in zip(niveles_relleno, intensidades):
        clear_screen()
        print()
        for fila_index, linea in enumerate(logo):
            r, g, b = fila_color(fila_index, len(logo))
            r, g, b = int(r * intensidad), int(g * intensidad), int(b * intensidad)
            pieza = rgb_fg(r, g, b)
            rellena = "".join(nivel if c == "█" else " " for c in linea)
            print("    " + pieza + rellena + RESET)
        print("\n    Iniciando TP ML...")
        time.sleep(0.12)

    clear_screen()
    print()
    for fila_index, linea in enumerate(logo):
        r, g, b = fila_color(fila_index, len(logo))
        pieza = rgb_fg(r, g, b)
        print("    " + pieza + linea + RESET)

    animated_print("\n    Cargando menú...", 0.02)
    time.sleep(0.3)
    clear_screen()
