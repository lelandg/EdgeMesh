import time
from datetime import datetime


class Spinner:
    def __init__(self, message=" {time} {count} ", format_string="%Y.%m.%d %H:%M:%S"):
        self.last_update = time.time()
        self.spinner_states = ['|', '/', '—', '\\']  # Classic spinner characters
        self.current_state = 0
        if not message.endswith(" "):
            message += " " # Ensure we have a space at the end. For spinner appearance. ;-)
        self.message = message
        self.count = 0
        try:
            stime = datetime.now().strftime(format_string)  # Ensure the format string is valid
            print(f"Time format string is valid: {stime}")
        except ValueError:
            format_string = "%Y.%m.%d %H:%M:%S"
            print(f"Invalid time format string. Using default: {format_string}")
        self.format_string = format_string
        self.print_it()

    def spin(self, message=""):
        current_time = time.time()
        if current_time - self.last_update >= 0.1:  # Ensure it advances at most twice a second
            self.last_update = current_time
            self.print_it(message)

    def print_it(self, message=""):
        if message == "":
            message = self.message
        if message.find("{time}") >= 0:
            message = message.replace("{time}", datetime.now().strftime(self.format_string))
        if message.find("{count}") >= 0:
            self.count += 1
            message = message.replace("{count}", str(self.count))
        self.current_state = (self.current_state + 1) % len(self.spinner_states)  # Cycle through states
        print(f"{message}{self.spinner_states[self.current_state]}", end='\r')

import sys

# Example usage
if __name__ == "__main__":
    args = " ".join(sys.argv[1:-1])  # Get command line arguments
    counter = 200
    try:
        counter = int(sys.argv[-1]) if len(sys.argv) > 1 else 1000  # Get the last argument as the counter
        print(f"Counter max set to {counter}")
    except ValueError: # If the last argument is not an integer, just use it as a part of the message
        args = " ".join(sys.argv[1:])

    if not args:
        args = "Spin #{count} at {time}" # Cool, right? Haha...
    spinner = Spinner(args)  # Create the spinner
    for _ in range(200):  # Example loop to see the spinner in action
        spinner.spin()  # Spin it!
        time.sleep(0.1)  # Simulate other work
    print()  # Ensure the next print is on a new line
