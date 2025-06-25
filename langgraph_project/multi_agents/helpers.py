import functools
from langchain.tools import Tool, BaseTool


class ToolInvocationTracker:
    def __init__(self, tool: BaseTool, min_calls: int = 1, max_calls: int = None):
        self.original = tool
        self.name = tool.name
        self.min_calls = min_calls
        self.max_calls = max_calls
        self.call_count = 0

        # wrap the tool's function
        @functools.wraps(tool.func)
        def wrapped(*args, **kwargs):
            self.call_count += 1
            return tool.func(*args, **kwargs)

        # create a new Tool with the wrapped function
        self.wrapped_tool = Tool.from_function(
            func=wrapped,
            name=tool.name,
            description=tool.description,
        )

    def assert_counts(self):
        if self.call_count < self.min_calls:
            raise RuntimeError(
                f"Tool '{self.name}' was called {self.call_count} times; expected at least {self.min_calls}."
            )
        if self.max_calls is not None and self.call_count > self.max_calls:
            raise RuntimeError(
                f"Tool '{self.name}' was called {self.call_count} times; expected at most {self.max_calls}."
            )

