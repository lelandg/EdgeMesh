import threading
import time
from queue import Queue

import space_mouse_event_handler

class SpaceMouseController:
    SMP_RAW_DATA_SIZE = 0x07
    SMP_CHANNEL_OFFSET = 0x00

    SMP_MOVE_CHANNEL = 0x01
    SMP_X_LSB_OFFSET = 0x01
    SMP_X_MSB_OFFSET = 0x02
    SMP_Y_LSB_OFFSET = 0x03
    SMP_Y_MSB_OFFSET = 0x04
    SMP_Z_LSB_OFFSET = 0x05
    SMP_Z_MSB_OFFSET = 0x06

    SMP_ROTATE_CHANNEL = 0x02
    SMP_PITCH_LSB_OFFSET = 0x01
    SMP_PITCH_MSB_OFFSET = 0x02
    SMP_ROLL_LSB_OFFSET = 0x03
    SMP_ROLL_MSB_OFFSET = 0x04
    SMP_YAW_LSB_OFFSET = 0x05
    SMP_YAW_MSB_OFFSET = 0x06

    SMP_BUTTON_CHANNEL = 0x03
    SMP_BUTTON_GROUP1 = 0x01
    SMP_BUTTON_GROUP2 = 0x02
    SMP_BUTTON_GROUP3 = 0x03
    SMP_BUTTON_GROUP4 = 0x04

    def __init__(self):
        """
        Initializes the SpaceMouseController with a provided 3D viewport instance.
        """
        self.sm_data = None
        self.running = False  # Flag to control the thread
        self.event_queue = Queue()
        self.handler = space_mouse_event_handler.SpaceMouseEventHandler()
        self.sm_device = self.handler.sm_device

    def _poll_and_process(self):
        while self.running:
            data = self.poll_input()
            if data:
                self.event_queue.put(data)  # Add sm_data to queue
            time.sleep(0.05)

    def next_event(self):
        """
        Returns the next event from the event queue.
        :return: The next event from the queue.
        """
        return self.event_queue.get()

    def process_events(self) -> dict:
        # Process all pending events in a thread-safe manner
        while not self.event_queue.empty():
            data = self.event_queue.get()
            self.sm_data = self.process_input(data)
            self.event_queue.task_done()
        return self.sm_data

    def poll_input(self):
        """
        Polls the space mouse for input. This method will simulate obtaining
        movement and rotation sm_data from the space mouse.
        :return: Example sm_data dictionary containing x, y, z, and rot directions.
        """
        self.handler.handle_event()
        data = self.handler.get_next_event()
        if data:
            data = self.process_input(data)
        return data

    def read_data(self):
        """
        Reads the space mouse data from the HID device.
        :return: A dictionary containing the space mouse sm_data.
        """
        data = self.handler.read_data()
        return self.process_input(data)

    def process_input(self, data=None) -> dict:
        """
        Processes the space mouse sm_data to move/rotate the object in the viewport.
        Parameters:
            data: A dictionary containing parsed information from the HID device,
            including time, position, button states, and other sm_data fields extracted
            from the device report. Keys in the dictionary correspond to specific
            sm_data fields, such as `t` for type of sm_data, `x` for X-axis position, `y` for Y-axis, z for Z-axis
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
        if data is None:
            data = self.handler.get_next_event()
        if data is None:
            return None
        # Lazy import to avoid circular dependency

        if data["t"] == space_mouse_event_handler.SMP_MOVE_CHANNEL:
            pan_x = data["x"]
            pan_y = data["y"]
            zoom_factor = data["z"]
            data["pan_x"] = pan_x
            data["pan_y"] = pan_y
            data["zoom_factor"] = zoom_factor

            rot_amount = data["rot"]
            counter_clockwise = data["cc"] == 255
            data["rot_amount"] = rot_amount
            data["counter_clockwise"] = counter_clockwise
        elif data["t"] == space_mouse_event_handler.SMP_BUTTON_CHANNEL:
            # TODO: Handle button press events
            pass
        elif data["t"] == space_mouse_event_handler.SMP_ROTATE_CHANNEL:
            # TODO: Determine what to do.
            #  Potentially rotate the object in the 3D viewport?
            pass

        return data

    def run(self):
        """
        Starts the controller in a separate thread.
        """
        self.running = True
        self.thread = threading.Thread(target=self._poll_and_process)
        self.handler.start_polling()
        self.thread.start()

    def stop(self):
        """
        Stops the controller and joins the thread.
        """
        self.running = False
        self.handler.stop_polling()
        if self.thread.is_alive():
            self.thread.join()


# Example usage
if __name__ == "__main__":
    # Lazy import to avoid circular dependency at the top level


    controller = SpaceMouseController()
    try:
        controller.run()
        while True:  # Keep the main program running
            time.sleep(1)
        data = controller.next_event()
    except KeyboardInterrupt:
        controller.stop()
        print("Space mouse event handler stopped.")
