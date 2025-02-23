import sys
import threading
import traceback
from collections import deque

import hid
import pygame
import pygame.locals

__version__ = "0.1.4"
__author__ = "Leland Green"
__license__ = "opensource"

"""
A simple SpaceMouse event handler that reads HID data from a SpaceMouse device. 
This script demonstrates how to read HID data from a SpaceMouse device and parse it into a structured dictionary.
The script uses the `hid` library to read HID data from the device and Pygame to handle events.
The `SpaceMouseEventHandler` class provides methods to handle HID events, parse HID data, and poll events.
The script includes a simple example of how to use the `SpaceMouseEventHandler` class to read and process HID data.
The script is intended to be used as a standalone module or integrated into a larger application.
Use it standalone to read HID data from a SpaceMouse device and analyze it for use in your application.
Use it as a reference to understand how to read HID data from other devices or integrate HID data into your application.
Or use it as-is in your application to read HID data from a SpaceMouse device and process it in real-time.

If you do not have a blocking thread, you can use the following code to poll events:
pygame.init()  # Initialize Pygame just for the events
sm_handler = SpaceMouseEventHandler()
sm_handler.start_polling(frequency)

"""
SMP_DATA_SIZE = 13  # Size of the HID data report (bytes)

SMP_RAW_DATA_SIZE 	=			0x07
SMP_CHANNEL_OFFSET	=			0x00

SMP_MOVE_CHANNEL	=			0x01
SMP_X_LSB_OFFSET	=			0x01
SMP_X_MSB_OFFSET	=			0x02
SMP_Y_LSB_OFFSET	=			0x03
SMP_Y_MSB_OFFSET	=			0x04
SMP_Z_LSB_OFFSET	=			0x05
SMP_Z_MSB_OFFSET	=			0x06

SMP_ROTATE_CHANNEL	=			0x02
SMP_PITCH_LSB_OFFSET=			0x01
SMP_PITCH_MSB_OFFSET=			0x02
SMP_ROLL_LSB_OFFSET	=			0x03
SMP_ROLL_MSB_OFFSET	=			0x04
SMP_YAW_LSB_OFFSET	=			0x05
SMP_YAW_MSB_OFFSET	=			0x06

SMP_BUTTON_CHANNEL	=			0x03
SMP_BUTTON_GROUP1	=			0x01
SMP_BUTTON_GROUP2	=			0x02
SMP_BUTTON_GROUP3	=			0x03
SMP_BUTTON_GROUP4	=			0x04

class SpaceMouseEventHandler:
    def __init__(self, frequency=0.01):
        """
        Initialize and set up the SpaceMouse event handler.
        """
        self.previous_data = dict()  # Store the previous data for comparison
        self.running = False
        self.debug = False  # Set to True to enable debug output
        self.product = ""  # Product name of the SpaceMouse device
        self._event_stack = deque()  # Private stack to store data_dict events in order
        self.sm_device = None  # Ensure sm_device is defined during initialization
        self._initialize_spacemouse()  # Run setup code
        if not self.sm_device:
            print("SpaceMouse initialization failed. No events have been or shall EVER be processed.\r\n"
                  "Or at least not EVER UNTIL you plug in a SpaceMouse. :-)")
            return

    def get_device(self):
        return self.sm_device

    def start_polling(self, frequency=0.1):
        self.frequency = frequency
        self.running = True  # Ensure `running` is a class attribute
        self._poll_loop()

    def stop_polling(self):
        self.running = False

    def _poll_loop(self):
        if self.running:  # Check if polling is still active
            self.poll_events()
            threading.Timer(self.frequency, self._poll_loop).start()

    def poll_events(self):
        # Handle HID SpaceMouse events
        self.handle_event()
        return

    def _initialize_spacemouse(self):
        """
        SpaceMouse initialization logic.
        """
        print("Initializing SpaceMouse...")
        try:
            devices = [d for d in hid.enumerate()]
            if not devices:
                print("No HID devices found, so no SpaceMouse.")
                return

            print(f"Looking for SpaceMouse. Found {len(devices)} compatible HID devices (total):")
            for device in devices:
                if self.debug:
                    print(f"Checking device: {device['product_string']} (Vendor ID: {device['vendor_id']}, "
                          f"Product ID: {device['product_id']})")

                product = device['product_string'].lower() if device['product_string'] else ""
                if "spacemouse" in product:  # Simplified product detection
                    device_info = device
                    self.sm_device = hid.device()
                    self.sm_device.open(device_info['vendor_id'], device_info['product_id'])
                    self.product = device_info['product_string']
                    print(f"Connected to SpaceMouse: {device_info['product_string']} Vendor ID: %04x, Product ID: %04x" %
                          (device_info['vendor_id'], device_info['product_id']))
                    self.sm_device.set_nonblocking(1)
                    return  # Successfully connected, stop further initialization attempts

            print("No compatible SpaceMouse device found.")
        except Exception as ex:
            print(f"Failed to initialize SpaceMouse: {ex}\n{traceback.format_exc()}")

    def process_input(self, data):
        """
        Updates data (dictionary) with the processed input from the SpaceMouse device. Adds keys:
            pan_y, pan_z, pan_x and rot_amount
        These are updated based on correct values from the SpaceMouse device.
        Processes the space mouse data to move/rotate the object in the viewport.
        Parameters:
            data: A dictionary containing parsed information from the HID device,
            including time, position, button states, and other data fields extracted
            from the device report. Keys in the dictionary correspond to specific
            data fields, such as `t` for type of data, `x` for X-axis position, `y` for Y-axis, z for Z-axis
            and so on. rot is rotation, cc = counter-clockwise.
            p is up/down movement (0-255) and ya is direction. 255 when the device is moved up, 0 = down.
            f = 255 = direction of x & y-axis.
                for x: 0 =  left, 255 = right
                for y: 0 =  up, 255 = down
            r = direction of z:
                for z: 0 = back (toward user), 255 = forward (away from user)
            t =
                1 = move
                3 = button press
        """
        # Lazy import to avoid circular dependency

        if data is None:
            data = self.get_next_event()

        if data["t"] == SMP_MOVE_CHANNEL:
            pan_x = data["x"]
            pan_y = data["y"]
            pan_z = data["z"]
            # Move the object in the 3D viewport. Example methods:
            # move_object(pan_x, pan_y, zoom_factor)

            counter_clockwise = data["cc"] == 255
            rot_amount = data["rot"]
            if counter_clockwise:
                rot_amount = -rot_amount
            forward = data["r"] == 255
            if not forward:
                pan_z = -pan_z
            flip = data["f"] == 255
            if flip:
                pan_x = -pan_x
                pan_y = -pan_y
            data["pan_x"] = pan_x
            data["pan_y"] = pan_y
            data["pan_z"] = pan_z
            data["rot_amount"] = rot_amount

            # rotate_object(rot_amount, counter_clockwise)
        elif data["t"] == SMP_BUTTON_CHANNEL:
            # TODO: Handle button press events
            pass
        elif data["t"] == SMP_ROTATE_CHANNEL:
            # TODO: Determine what to do.
            #  Potentially rotate the object in the 3D viewport?
            pass
        return data

    def product_string(self):
        return self.product

    def read_data(self):
        # Attempt to read data from the device (non-blocking read)
        # if data:
        data = self.sm_device.read(SMP_DATA_SIZE)
        if not data or data == self.previous_data:  # If no data is available
            return
        self.previous_data = data  # Store the previous data for comparison

        # Parse data into a meaningful structure here
        if len(data) < SMP_DATA_SIZE:  # Ensure enough data for parsing (avoid IndexError)
            print(f"Warning: Received incomplete data ({len(data)} bytes): {data}")
            return

        # Parse data (this assumes a specific structure for the HID data report)
        parsed_data = self.parse_hid_data(data)
        if self.debug:
            print("Parsed HID data:", parsed_data)

        self._event_stack.append(parsed_data)

    def handle_hid_event(self, hid_device):
        """Process events from the SpaceMouse device."""
        if not hid_device:  # Validate device before processing
            print("Error: HID device is not initialized. Please check the device connection.")
            return

        try:
            self.read_data()
        except OSError as e:
            if self.debug:
                print(f"Error reading from HID device: {e}. The device may be disconnected or unavailable.")
        except Exception as e:
            print(f"Unexpected error during HID event handling: {e}\n{traceback.format_exc()}")

    def parse_hid_data(self, data):
        """Parse raw HID data into a structured dictionary."""
        rot = data[11]          # Extract rotation data
        rot = rot - 127 # Below zero = clockwise, above zero = counter-clockwise
        (t, x, y, z, r, p, ya, buttons, buttons_changed, xyz_rpy_change_count, f, rot, cc) = data
        if cc == 255:
            cc = True
            rot = -(255 - rot)
        else:
            cc = False
        if r == 255:
            r = False
        else:
            r = True
            z = -z
        if f == 255:
            f = True
        else:
            x = -x
            y = -y
            f = False

        data_dict = {"t" : t, "x" : x,  "y" : y, "z" : z, "r" : r, "p" : p, "ya" : ya,
                     "buttons" : buttons, "buttons_changed" : buttons_changed,
                     "xyz_rpy_change_count" : xyz_rpy_change_count,
                     "f" : f, "rot" : rot, "cc" : cc}
        return data_dict

    def handle_event(self):
        """Wrapper function to handle events."""
        if self.sm_device:
            self.handle_hid_event(self.sm_device)
        else:
            print("SpaceMouse device is not connected. Handling event skipped.")

    def get_next_event(self):
        """
        Retrieve the next event (data_dict) in the order it was added,
        removing it from the stack.

        Returns:
            dict: The next `data_dict` from the stack, or None if the stack is empty.
        """
        if self._event_stack:
            return self._event_stack.popleft()  # Remove and return the earliest event
        return None  # Return None if no events are available

    def has_events(self):
        """
        Check if there are any events in the stack.

        Returns:
            bool: True if there are events in the stack, False otherwise.
        """
        return bool(self._event_stack)


# Example usage
if __name__ == "__main__":
    try:
        handler = SpaceMouseEventHandler()
        handler.start_polling(0.1)
        # Example data polling loop
        while True:
            data = handler.get_next_event()
            while data:
                print(f"Processed Event: {data}")
                data = handler.get_next_event()
    except KeyboardInterrupt:
        print("Stopping event handler...")
        handler.stop_polling()
