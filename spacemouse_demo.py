import time

import pygame
from pygame.constants import QUIT, KEYDOWN
from pygame.locals import *
import sys
import hid  # Modern HID library for accessing SpaceMouse devices

from spinner import Spinner
from space_mouse_event_handler import SpaceMouseEventHandler

# import logging

# # Configure logging
# logging.basicConfig(
#     filename='spacemouse_demo.log',  # Log file name
#     level=logging.DEBUG,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#     format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
# )
#
# # Example log messages
# logging.info("Logging is initialized.")
# logging.debug("Debug-level log example.")


# Dictionary to store the min and max values for each key
key_stats = {}

def get_max_min(data):
    """
     Simulated function to process HID events and analyze the input data.
     :param data: A dictionary where each key corresponds to a numerical value (float/int)
     """
    global key_stats

    # Iterate through the key-value pairs in the data dictionary
    for key, value in data.items():
        # Only process int/float values
        if not isinstance(value, (int, float)):
            continue

        # Check if this key exists in `key_stats`
        if key not in key_stats:
            # Initialize with the current value
            key_stats[key] = {'min': value, 'max': value}
        else:
            # Update min and max values for the key
            key_stats[key]['min'] = min(key_stats[key]['min'], value)
            key_stats[key]['max'] = max(key_stats[key]['max'], value)

def summarize_data():
    """
    Compose a string displaying the max and min values for non-zero keys.
    :return: A formatted string summarizing non-zero keys
    """
    global key_stats
    # Build the summary string
    summary = []
    for key, stats in key_stats.items():
        min_val, max_val = stats['min'], stats['max']
        # Include in the summary only if min or max is non-zero
        if min_val != 0 or max_val != 0:
            summary.append(f"{key}: {min_val}-{max_val}")

    # Return the formatted summary
    return ", ".join(summary)


def main():
    # Detect SpaceMouse_Pro_Wireless_BT with (Vendor ID: 9583, Product ID: 50744)
    # Vendor ID: 256f, Product ID: c638
    # However, see below for a more general approach to finding SpaceMouse devices
    # target_vendor_id = 0x256f  # 3DConnexion Vendor ID for SpaceMouse
    # target_product_id = 0xc638  # Sample Product ID (adjust based
    sm_device = None
    device_info = None

    spinner = Spinner("{time} Looking for SpaceMouse devices... ")
    handler = SpaceMouseEventHandler()

    # Initialize pygame...
    passed, failed = pygame.init()
    if failed > 0:
        print("Error initializing pygame")
        sys.exit(1)

    # Open a window...
    pygame.display.set_caption(f"{handler.product_string()} Demo")
    srf = pygame.display.set_mode((640, 480))

    # Event loop...
    running = True
    spinner.spin("{time} Running SpaceMouse demo... ")
    while running:
        # Get a list of pygame events (handles close, etc.)
        events = pygame.event.get()
        for evt in events:
            # Close button?
            if evt.type == QUIT:
                running = False

            # Escape key?
            elif evt.type == KEYDOWN and evt.key == 27:  # Escape key
                running = False

        time.sleep(0.1)

        data = handler.get_next_event()
        if data:
            get_max_min(data)
            # spinner.spin(f"{time} SpaceMouse data: {data}")
            # logging.info(f"SpaceMouse data: {summarize_data()}")
            # Set the Pygame display dimensions
            screen_width, screen_height = 800, 600
            screen = pygame.display.set_mode((screen_width, screen_height))
            # pygame.display.set_caption("SpaceMouse Stats Display")

            # Set font and colors
            font = pygame.font.Font(None, 36)
            background_color = (30, 30, 30)  # Dark gray
            text_color = (255, 255, 255)  # White

            # Example global data
            global key_stats

            running = True
            clock = pygame.time.Clock()

            # Clear the screen
            screen.fill(background_color)

            y_offset = 10  # Starting Y position
            text_surface = font.render(f"{handler.product_string()} Data Summary", True, text_color)
            screen.blit(text_surface, (20, y_offset))
            pygame.display.flip()

            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                data = handler.get_next_event()

                if not data:
                    continue

                # Clear the screen
                screen.fill(background_color)

                # get_max_min(data)
                print(f"{handler.product_string()} data: {data}")

                y_offset = 10  # Starting Y position
                text_surface = font.render(f"{handler.product_string()} Data Summary", True, text_color)
                screen.blit(text_surface, (20, y_offset))
                y_offset += 40
                # Display each key, value, min, and max
                for idx, (key, value) in enumerate(data.items()):
                    if key in key_stats.keys():
                        min_val, max_val = key_stats[key]['min'], key_stats[key]['max']
                    else:
                        min_val, max_val = 0, 0
                    text_surface = font.render(
                        f"{key}: {value:.2f} (Min: {min_val}, Max: {max_val})", True, text_color
                    )
                    screen.blit(text_surface, (20, y_offset))
                    y_offset += 40  # Increase Y position for each line

                # Update the display
                pygame.display.flip()

                # Cap the frame rate
                clock.tick(60)

    handler.stop_polling()
    print("SpaceMouse connection closed.")


if __name__ == "__main__":
    main()
