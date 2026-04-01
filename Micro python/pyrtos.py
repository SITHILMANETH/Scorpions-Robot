import time

_tasks = []

class timeout:
    def __init__(self, seconds):
        self.seconds = seconds


class Task:
    def __init__(self, func, priority=1):

        self.generator = func()
        self.priority = priority
        self.next_run = time.ticks_ms()

    def run(self):

        try:

            result = next(self.generator)

            if result:

                for event in result:

                    if isinstance(event, timeout):

                        delay = int(event.seconds * 1000)
                        self.next_run = time.ticks_add(time.ticks_ms(), delay)

        except StopIteration:
            return False

        return True


def add_task(func, priority=1):

    global _tasks

    task = Task(func, priority)

    _tasks.append(task)

    _tasks.sort(key=lambda t: t.priority)


def start():

    while True:

        now = time.ticks_ms()

        for task in _tasks:

            if time.ticks_diff(now, task.next_run) >= 0:

                alive = task.run()

                if not alive:
                    _tasks.remove(task)

        time.sleep_ms(1)
