class Stack:
    """A simple stack implementation using a list."""

    def __init__(self):
        self._items = []

    def push(self, item):
        """Pushes an item onto the stack."""
        self._items.append(item)

    def pop(self):
        """Removes and returns the top item from the stack."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()

    def peek(self):
        """Returns the top item without removing it."""
        if self.is_empty():
            return None
        return self._items[-1]

    def is_empty(self) -> bool:
        """Checks if the stack is empty."""
        return len(self._items) == 0

    def size(self) -> int:
        """Returns the number of items in the stack."""
        return len(self._items)
