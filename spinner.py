import time
from datetime import datetime


class Spinner:
    def __init__(self, message=" {time} {count} ", format_string="%Y.%m.%d %H:%M:%S", limit=0.1):
        """
        Initialize a spinner for displaying a spinning arrow next to a message with a timestamp and count,
        limited in updates by a specified interval in seconds.

        Attributes:
            last_update (float): Tracks the last update time in seconds since epoch.
            spinner_states (list[str]): List of characters representing the spinner states.
            current_state (int): Index of the current spinner state in `spinner_states`.
            message (str): The message displayed alongside the spinner.
            count (int): Tracks the number of updates or spins invoked.
            limit (float): Minimum time interval (in seconds) required between spinner updates.
            format_string (str): Format string for timestamp to be displayed alongside the message.

        Args:
            message (str): Message displayed alongside the spinner. Default is " {time} {count} ".
            format_string (str): Custom time format string for the timestamp. Default is "%Y.%m.%d %H:%M:%S".
            limit (float): Minimum time interval between spinner updates (in seconds). Default is 0.1, which
            limits to a maximum of 10 updates per second. 1.0 = 1 update per second, 0.5 = 2 updates per second, etc.
            Depending on what you're doing, you may need 60-120 updates per second, so adjust this accordingly. It does
            not slow your system, no matter how you set it. It just controls the frequency of updates.
            Note you'd only need 60-120 updates per second for very specific applications, like real-time data
            visualization or high-speed data processing.
        """
        self.max_len = 0
        self.last_update = time.time()
        self.spinner_states = ['|', '/', '—', '\\']  # Classic spinner characters
        self.current_state = 0
        if not message.endswith(" "):
            message += " " # Ensure we have a space at the end. For spinner appearance. ;-)
        self.message = message
        self.count = 0
        self.limit = limit
        try:
            datetime.now().strftime(format_string)  # Ensure the format string is valid
        except ValueError:
            format_string = "%Y.%m.%d %H:%M:%S"
            print(f"Invalid time format string. Using default: {format_string}")
        self.format_string = format_string
        self.spin() # Print the initial message

    def spin(self, message=""):
        """
        Spins the status or progress indicator by printing a message if the time since the
        last update exceeds the predefined limit.

        This method is utilized to regulate how frequently a spinning or updating
        action is performed. It ensures that updates happen only after a specific time
        interval has passed since the last operation. If the interval condition is
        satisfied, it performs the update by invoking the `print_it` method with the
        provided message.

        Parameters:
            message (str): Optional. A string to display during the spinning update.
        """
        current_time = time.time()
        if current_time - self.last_update >= self.limit:  # Ensure it advances at most N times a second
            self.last_update = current_time
            self.print_it(message)

    def print_it(self, message="", end='\r'):
        """
        Prints a formatted message with optional placeholders and a spinner, updating
        its state to give the appearance of progression. The method substitutes
        placeholders in the message for the current time and a count, cycles through
        predefined spinner states, and prints the result.

        Args:
            message (str): The message to print, with optional placeholders {time}
                for the current time and {count} for an incrementing counter.
                Defaults to an empty string.
            end (str): The string appended after the last value. Defaults to '\r'.
        """
        if message == "":
            message = self.message
        if message.find("{time}") >= 0:
            message = message.replace("{time}", datetime.now().strftime(self.format_string))
        if message.find("{count}") >= 0:
            self.count += 1
            message = message.replace("{count}", str(self.count))
        if not message.endswith(" "):
            message += " "  # Ensure we have a space at the end. For spinner appearance. ;-)
        self.current_state = (self.current_state + 1) % len(self.spinner_states)  # Cycle through states
        if len(message) < self.max_len:
            message += " " * (self.max_len - len(message))  # Clear any previous spinner
        self.max_len = len(message)
        print(f"{message}{self.spinner_states[self.current_state]}", end=end, flush=True)

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
    spinner = Spinner(args, limit=0.0)  # Create the spinner
    for _ in range(0, 200):  # Example loop to see the spinner in action
        spinner.spin()  # Spin it!
        time.sleep(0.1)  # Simulate other work
    spinner.print_it(end='\n')  # Print the final message
