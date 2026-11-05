# main.py
import threading
import time

import pygame

from worldObjects import world


class SharedState:
    """Thread-safe mailbox between the sim thread and the UI thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = None
        self.stop = threading.Event()
        self.paused = threading.Event()   # set() = paused

    def publish(self, snap):
        with self._lock:
            self._snapshot = snap

    def latest(self):
        with self._lock:
            return self._snapshot


def sim_loop(w, shared: SharedState, publish_hz: float = 60.0):
    """Runs on the worker thread. Ticks as fast as it can."""
    publish_interval = 1.0 / publish_hz
    last_publish = 0.0

    shared.publish(w.snapshot())  # so the UI has something immediately

    while not shared.stop.is_set():
        if shared.paused.is_set():
            time.sleep(0.01)
            continue
        time.sleep(0.02)
        w.tick()

        # No point building a snapshot more often than the UI can draw one
        now = time.perf_counter()
        if now - last_publish >= publish_interval:
            shared.publish(w.snapshot())
            last_publish = now
    w.hasEnded()


KIND_COLOURS = {
    "default":   (200, 200, 200),
    "creatureA": (220, 80, 80),
    "creatureB": (80, 140, 230),
    "plant":     (80, 200, 100),
    "water":     (60, 110, 200),
    "corpse":    (140, 120, 100),
}
SIDEBAR_W = 640
DECISION_LABELS = ["R", "L", "Move", "Atk", "Plant", "Corps", "Drink", "Mate"]
STAT_LABELS = ["HP", "Hun", "Thi", "Kids"]
ACTION_START = 3

LABEL_COLOUR = (160, 160, 175)
LOW_COLOUR = (230, 90, 90)


def stat_colour(value, low=25):
    """Red when a 0-100 stat is dangerously low."""
    return LOW_COLOUR if value < low else (210, 210, 220)


def draw_sidebar(screen, snap, font, x0, height):
    pygame.draw.rect(screen, (28, 28, 36), (x0, 0, SIDEBAR_W, height))
    pygame.draw.line(screen, (70, 70, 85), (x0, 0), (x0, height))

    line_h = font.get_linesize() + 2
    id_w, stat_w, col_w = 52, 44, 48

    # x positions of each column
    id_x = x0 + 8
    stat_x = [id_x + id_w + i * stat_w for i in range(len(STAT_LABELS))]
    dec_start = id_x + id_w + len(STAT_LABELS) * stat_w + 12
    dec_x = [dec_start + i * col_w for i in range(len(DECISION_LABELS))]

    y = 8

    # header
    screen.blit(font.render("creature", True, LABEL_COLOUR), (id_x, y))
    for x, label in zip(stat_x, STAT_LABELS):
        screen.blit(font.render(label, True, LABEL_COLOUR), (x, y))
    for x, label in zip(dec_x, DECISION_LABELS):
        screen.blit(font.render(label, True, LABEL_COLOUR), (x, y))
    y += line_h + 4

    max_rows = max(0, (height - y - line_h) // line_h)
    for n, c in enumerate(snap.creatures):
        if n >= max_rows:
            more = len(snap.creatures) - max_rows
            screen.blit(font.render(f"... +{more} more", True, LABEL_COLOUR), (id_x, y))
            break

        # id tag
        tag = ("A" if c.kind == "creatureA" else "B") + str(c.id)
        screen.blit(font.render(tag, True, KIND_COLOURS[c.kind]), (id_x, y))

        # stats
        stats = [
            (f"{c.health:.0f}", stat_colour(c.health)),
            (f"{c.hunger:.0f}", stat_colour(c.hunger)),
            (f"{c.thirst:.0f}", stat_colour(c.thirst)),
            (str(c.numChildren), (210, 210, 220)),
        ]
        for x, (text, colour) in zip(stat_x, stats):
            screen.blit(font.render(text, True, colour), (x, y))

        # decision
        if not c.decision:
            screen.blit(font.render("--", True, (110, 110, 120)), (dec_x[0], y))
        else:
            chosen = ACTION_START + max(range(5), key=lambda k: c.decision[ACTION_START + k])
            for i, v in enumerate(c.decision):
                colour = (255, 220, 80) if i == chosen else (210, 210, 220)
                screen.blit(font.render(f"{v:+.2f}", True, colour), (dec_x[i], y))
        y += line_h

def ui_loop(shared: SharedState, window_size=(1300, 700)):
    """Runs on the main thread."""
    pygame.init()
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption("World")
    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 12)
    clock = pygame.time.Clock()

    while not shared.stop.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                shared.stop.set()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if shared.paused.is_set():
                    shared.paused.clear()
                else:
                    shared.paused.set()

        snap = shared.latest()
        screen.fill((20, 20, 25))

        if snap is not None:
            sw, sh = screen.get_size()
            world_w = max(1, sw - SIDEBAR_W)           # world gets the left part
            scale = min(world_w / snap.width, sh / snap.height)
            ox = (world_w - snap.width * scale) / 2
            oy = (sh - snap.height * scale) / 2

            pygame.draw.rect(screen, (60, 60, 70),
                             (ox, oy, snap.width * scale, snap.height * scale), 1)

            for obj in snap.objects:
                colour = KIND_COLOURS.get(obj.kind, KIND_COLOURS["default"])
                px = ox + obj.x * scale
                py = oy + obj.y * scale
                pygame.draw.circle(screen, colour, (px, py), max(2, obj.radius * scale))

            hud = f"tick {snap.tick}  objects {len(snap.objects)} creatures {len(snap.creatures)}  fps {clock.get_fps():.0f}"
            if shared.paused.is_set():
                hud += "  [PAUSED]"
            screen.blit(font.render(hud, True, (255, 255, 255)), (8, 8))

            draw_sidebar(screen, snap, small, sw - SIDEBAR_W, sh)
            if  len(snap.creatures)<=0:
                shared.paused.set()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    w = world.World(1)
    shared = SharedState()

    sim_thread = threading.Thread(target=sim_loop, args=(w, shared), daemon=True)
    sim_thread.start()

    try:
        ui_loop(shared)          # blocks on the main thread until the window closes
    finally:
        shared.stop.set()
        sim_thread.join(timeout=2)