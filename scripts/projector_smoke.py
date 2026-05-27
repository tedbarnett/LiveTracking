"""Dead-simple projector smoke test. Cycles big colored fullscreen frames
on each detected display for 3s each, labeled with the display index so you
can see which physical screen got which index."""
import os, time, sys
import pygame

pygame.init()
sizes = pygame.display.get_desktop_sizes()
print(f"detected {len(sizes)} display(s):")
for i, s in enumerate(sizes):
    print(f"  display {i}: {s[0]}x{s[1]}")

# Cycle each display for 3 seconds, fullscreen, with a label.
for i, size in enumerate(sizes):
    print(f"\n>>> opening display {i} ({size[0]}x{size[1]}) for 3 seconds")
    screen = pygame.display.set_mode(size, pygame.NOFRAME, display=i)
    font = pygame.font.SysFont(None, 200)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    t0 = time.time()
    frame = 0
    while time.time() - t0 < 3.0:
        c = colors[frame % 3]
        screen.fill(c)
        text = font.render(f"DISPLAY {i}  {size[0]}x{size[1]}", True, (255, 255, 255))
        screen.blit(text, (50, size[1] // 2 - 100))
        pygame.display.flip()
        for _ in pygame.event.get():
            pass
        time.sleep(0.4)
        frame += 1
    pygame.display.quit()
    pygame.display.init()

pygame.quit()
print("\nsmoke test done")
