import enum


class MCPFunctionValueType(enum.IntEnum):
    NOT_NEED_VALUE = enum.auto()
    NEED_VALUE = enum.auto()
    SON_NEED_VALUE = enum.auto()


class MCPFunctionType(enum.IntEnum):
    CONDITION_CALL = enum.auto()
    FILE_WRITE = enum.auto()
    FILE_READ = enum.auto()
    FILE_HYBRID = enum.auto()
    OTHER = enum.auto()


class McpExcuteStageType(enum.IntEnum):
    TOOL_INITIALIZATION = 2
    TOOL_EXECUTION = 3
    CONDITION_CHECK = 6
    CONDITIONAL_EXECUTION = 7
    FILE_OPEN = 8
    FILE_WRITE = 10
    FILE_CLOSE = 11
    TEST_STYLE_CHECK = 12
    TEST_COMPILE_CHECK = 13
    TEST_FUNC_CHECK = 14
    TEST_REPORT = 15
    QUERY_ANALYSIS = 17
    NET_SEARCH = 18
    DATA_ANALYSIS = 22
    NET_CONNECT = 26
    NET_DISCONNECT = 27
    OTHER = 28
