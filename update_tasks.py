"""
Update tasks.md file to mark completed tasks.

This script reads the tasks.md file, identifies tasks marked as complete [x],
and updates the file accordingly.

Usage:
    python update_tasks.py

The script will automatically detect changes in the tasks.md file and update it.
"""

import os
import re
import datetime

def update_tasks_file():
    """Update the tasks.md file to mark completed tasks."""
    tasks_file_path = os.path.join('docs', 'tasks.md')

    # Check if the file exists
    if not os.path.exists(tasks_file_path):
        print(f"Error: {tasks_file_path} does not exist.")
        return False

    # Read the current content of the file
    with open(tasks_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Make a backup of the original file
    backup_path = f"{tasks_file_path}.bak"
    with open(backup_path, 'w', encoding='utf-8') as file:
        file.write(content)

    # Count completed tasks, excluding examples in the explanation section
    # First, split the content to exclude the explanation section
    explanation_end = content.find("## Architecture and Code Organization")
    explanation_section = content[:explanation_end]
    task_section = content[explanation_end:]

    # Count in the task section only
    completed_count = task_section.count('[x]')
    total_count = task_section.count('[ ]') + completed_count

    # Add completion status at the top of the file if not already present
    completion_status = f"**Current Progress:** {completed_count}/{total_count} tasks completed ({completed_count/total_count*100:.1f}%)"

    if "**Current Progress:**" not in content:
        # Add after the implementation recommendations line
        pattern = r"(For implementation recommendations for each task, see the \[implementation_recommendations\.md\]\(implementation_recommendations\.md\) file\.)"
        replacement = f"\\1\n\n{completion_status}\n\nLast updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        content = re.sub(pattern, replacement, content)
    else:
        # Update existing progress information
        pattern = r"\*\*Current Progress:\*\*.*\n\nLast updated:.*"
        replacement = f"{completion_status}\n\nLast updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        content = re.sub(pattern, replacement, content)

    # Write the updated content back to the file
    with open(tasks_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Tasks file updated. Current progress: {completed_count}/{total_count} tasks completed ({completed_count/total_count*100:.1f}%)")
    return True

if __name__ == "__main__":
    update_tasks_file()
