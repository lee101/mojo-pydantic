from max.algorithm import parallelize
from std.ffi import external_call
from std.math import floor
from std.memory import stack_allocation
from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def is_space(c: UInt8) -> Bool:
    return c == UInt8(32) or c == UInt8(9) or c == UInt8(10) or c == UInt8(13)


def skip_space(data: BPtr, n: Int, i: Int) -> Int:
    var pos = i
    while pos < n and is_space(data[pos]):
        pos += 1
    return pos


def equal_ascii(data: BPtr, start: Int, end: Int, text: String) -> Bool:
    var p = text.unsafe_ptr()
    if end - start != text.byte_length():
        return False
    for i in range(end - start):
        var a = data[start + i]
        var b = p[i]
        if a >= UInt8(65) and a <= UInt8(90):
            a += UInt8(32)
        if b >= UInt8(65) and b <= UInt8(90):
            b += UInt8(32)
        if a != b:
            return False
    return True


def token_end(data: BPtr, n: Int, start: Int) -> Int:
    var i = start
    while i < n and data[i] != UInt8(44) and data[i] != UInt8(93):
        i += 1
    var end = i
    while end > start and is_space(data[end - 1]):
        end -= 1
    return end


def advance_commas(data: BPtr, n: Int, start: Int, count: Int) -> Int:
    comptime W = simd_width_of[DType.uint8]()
    var remaining = count
    var i = start
    while i + W <= n:
        var chars = data.load[width=W](i)
        var mask = chars.eq(UInt8(44))
        var found = Int(mask.cast[DType.uint8]().reduce_add())
        if found >= remaining:
            for lane in range(W):
                if mask[lane]:
                    remaining -= 1
                    if remaining == 0:
                        return i + lane + 1
        else:
            remaining -= found
        i += W
    while i < n:
        if data[i] == UInt8(44):
            remaining -= 1
            if remaining == 0:
                return i + 1
        i += 1
    return -1


def numeric_bounds(data: BPtr, n: Int, start: Int) -> Tuple[Int, Int, Int]:
    var i = skip_space(data, n, start)
    var quoted = 0
    if i < n and data[i] == UInt8(34):
        quoted = 1
        i += 1
        i = skip_space(data, n, i)
        var end = i
        while end < n and data[end] != UInt8(34):
            if data[end] == UInt8(92):
                return (-1, -1, -1)
            end += 1
        if end >= n:
            return (-1, -1, -1)
        var value_end = end
        while value_end > i and is_space(data[value_end - 1]):
            value_end -= 1
        return (i, value_end, end + 1)
    var end = token_end(data, n, i)
    return (i, end, end)


def valid_float(data: BPtr, start: Int, end: Int) -> Bool:
    if start >= end:
        return False
    var i = start
    if data[i] == UInt8(43) or data[i] == UInt8(45):
        i += 1
    var digits = 0
    while i < end and data[i] >= UInt8(48) and data[i] <= UInt8(57):
        digits += 1
        i += 1
    if i < end and data[i] == UInt8(46):
        i += 1
        while i < end and data[i] >= UInt8(48) and data[i] <= UInt8(57):
            digits += 1
            i += 1
    if digits == 0:
        return False
    if i < end and (data[i] == UInt8(101) or data[i] == UInt8(69)):
        i += 1
        if i < end and (data[i] == UInt8(43) or data[i] == UInt8(45)):
            i += 1
        var exponent_digits = 0
        while i < end and data[i] >= UInt8(48) and data[i] <= UInt8(57):
            exponent_digits += 1
            i += 1
        if exponent_digits == 0:
            return False
    return i == end


def valid_json_float(data: BPtr, start: Int, end: Int) -> Bool:
    if start >= end:
        return False
    var i = start
    if data[i] == UInt8(45):
        i += 1
    if i >= end or data[i] < UInt8(48) or data[i] > UInt8(57):
        return False
    if data[i] == UInt8(48):
        i += 1
        if i < end and data[i] >= UInt8(48) and data[i] <= UInt8(57):
            return False
    else:
        while i < end and data[i] >= UInt8(48) and data[i] <= UInt8(57):
            i += 1
    if i < end and data[i] == UInt8(46):
        i += 1
        var fraction_start = i
        while i < end and data[i] >= UInt8(48) and data[i] <= UInt8(57):
            i += 1
        if i == fraction_start:
            return False
    if i < end and (data[i] == UInt8(101) or data[i] == UInt8(69)):
        i += 1
        if i < end and (data[i] == UInt8(43) or data[i] == UInt8(45)):
            i += 1
        var exponent_start = i
        while i < end and data[i] >= UInt8(48) and data[i] <= UInt8(57):
            i += 1
        if i == exponent_start:
            return False
    return i == end


def valid_json_int(data: BPtr, start: Int, end: Int) -> Bool:
    if start >= end:
        return False
    var i = start
    if data[i] == UInt8(45):
        i += 1
    if i >= end:
        return False
    if data[i] == UInt8(48):
        return i + 1 == end
    if data[i] < UInt8(49) or data[i] > UInt8(57):
        return False
    i += 1
    while i < end and data[i] >= UInt8(48) and data[i] <= UInt8(57):
        i += 1
    return i == end


def parse_float(data: BPtr, start: Int, end: Int) -> Float64:
    return external_call["strtod", Float64](data + start, Int(0))


def parse_small_i64(data: BPtr, start: Int, end: Int) -> Tuple[Bool, Int64]:
    var i = start
    var negative = False
    if data[i] == UInt8(45):
        negative = True
        i += 1
    var value = Int64(0)
    comptime LIMIT = Int64(9007199254740991)
    while i < end:
        var digit = Int64(data[i] - UInt8(48))
        if value > (LIMIT - digit) // 10:
            return (False, Int64(0))
        value = value * 10 + digit
        i += 1
    return (True, -value if negative else value)


def finish_element(data: BPtr, n: Int, i: Int) -> Tuple[Int, Int]:
    var pos = skip_space(data, n, i)
    if pos >= n:
        return (-1, -1)
    if data[pos] == UInt8(44):
        return (pos + 1, 0)
    if data[pos] == UInt8(93):
        return (pos + 1, 1)
    return (-1, -1)


def parse_f64_partition(
    data_addr: Int,
    n: Int,
    dst_addr: Int,
    capacity: Int,
    strict: Bool,
    byte_start: Int,
    count_start: Int,
    count_end: Int,
) -> Bool:
    var data = BPtr(unsafe_from_address=data_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    var i = byte_start
    var count = count_start
    while count < count_end:
        var original_start = skip_space(data, n, i)
        if original_start >= n:
            return False
        var quoted = data[original_start] == UInt8(34)
        var bounds = numeric_bounds(data, n, i)
        var start = bounds[0]
        var end = bounds[1]
        var after = bounds[2]
        if start < 0 or (strict and quoted):
            return False
        var value = 0.0
        if (quoted and valid_float(data, start, end)) or (
            not quoted and valid_json_float(data, start, end)
        ):
            value = parse_float(data, start, end)
        elif not strict and not quoted and equal_ascii(
            data, start, end, String("true")
        ):
            value = 1.0
        elif not strict and not quoted and equal_ascii(
            data, start, end, String("false")
        ):
            value = 0.0
        else:
            return False
        dst[count] = value
        count += 1
        var finish = finish_element(data, n, after)
        i = finish[0]
        if i < 0:
            return False
        if finish[1] == 1:
            return (
                count == capacity
                and count_end == capacity
                and skip_space(data, n, i) == n
            )
    return count_end < capacity and data[i - 1] == UInt8(44)


def json_f64_array_parallel(
    data_addr: Int,
    n: Int,
    dst_addr: Int,
    capacity: Int,
    strict: Bool,
    first: Int,
) -> Int:
    comptime TASKS = 8
    comptime FAILURE_BITS = Int64(9221120237041090560)
    var positions = stack_allocation[TASKS, Int]()
    var data = BPtr(unsafe_from_address=data_addr)
    positions[0] = first
    for task in range(1, TASKS):
        var previous = capacity * (task - 1) // TASKS
        var target = capacity * task // TASKS
        positions[task] = advance_commas(
            data, n, positions[task - 1], target - previous
        )
        if positions[task] < 0:
            return -1

    def worker(task: Int) capturing:
        var count_start = capacity * task // TASKS
        var count_end = capacity * (task + 1) // TASKS
        var ok = parse_f64_partition(
            data_addr,
            n,
            dst_addr,
            capacity,
            strict,
            positions[task],
            count_start,
            count_end,
        )
        if not ok:
            IPtr(unsafe_from_address=dst_addr)[count_start] = FAILURE_BITS

    parallelize[worker](TASKS, TASKS)
    var result_bits = IPtr(unsafe_from_address=dst_addr)
    for task in range(TASKS):
        if result_bits[capacity * task // TASKS] == FAILURE_BITS:
            return -1
    return capacity


def json_f64_array(data: BPtr, n: Int, dst: FPtr, capacity: Int, strict: Bool) -> Int:
    var i = skip_space(data, n, 0)
    if i >= n or data[i] != UInt8(91):
        return -1
    i = skip_space(data, n, i + 1)
    if i < n and data[i] == UInt8(93):
        return 0 if skip_space(data, n, i + 1) == n else -1
    var count = 0
    while i < n and count < capacity:
        var original_start = skip_space(data, n, i)
        if original_start >= n:
            return -1
        var quoted = data[original_start] == UInt8(34)
        var bounds = numeric_bounds(data, n, i)
        var start = bounds[0]
        var end = bounds[1]
        var after = bounds[2]
        if start < 0 or (strict and quoted):
            return -1
        var value = 0.0
        if (quoted and valid_float(data, start, end)) or (
            not quoted and valid_json_float(data, start, end)
        ):
            value = parse_float(data, start, end)
        elif not strict and not quoted and equal_ascii(data, start, end, String("true")):
            value = 1.0
        elif not strict and not quoted and equal_ascii(data, start, end, String("false")):
            value = 0.0
        else:
            return -1
        dst[count] = value
        count += 1
        var finish = finish_element(data, n, after)
        i = finish[0]
        if i < 0:
            return -1
        if finish[1] == 1:
            return count if skip_space(data, n, i) == n else -1
    return -1


def json_i64_array(data: BPtr, n: Int, dst: IPtr, capacity: Int, strict: Bool) -> Int:
    var i = skip_space(data, n, 0)
    if i >= n or data[i] != UInt8(91):
        return -1
    i = skip_space(data, n, i + 1)
    if i < n and data[i] == UInt8(93):
        return 0 if skip_space(data, n, i + 1) == n else -1
    var count = 0
    while i < n and count < capacity:
        var original_start = skip_space(data, n, i)
        if original_start >= n:
            return -1
        var quoted = data[original_start] == UInt8(34)
        var bounds = numeric_bounds(data, n, i)
        var start = bounds[0]
        var end = bounds[1]
        var after = bounds[2]
        if start < 0 or (strict and quoted):
            return -1
        if not quoted and valid_json_int(data, start, end):
            var parsed = parse_small_i64(data, start, end)
            if not parsed[0]:
                return -1
            dst[count] = parsed[1]
        else:
            var value = 0.0
            if strict:
                return -1
            if (
                (quoted and valid_float(data, start, end))
                or (not quoted and valid_json_float(data, start, end))
            ):
                value = parse_float(data, start, end)
            elif not quoted and equal_ascii(data, start, end, String("true")):
                value = 1.0
            elif not quoted and equal_ascii(data, start, end, String("false")):
                value = 0.0
            else:
                return -1
            if value != floor(value) or value <= -9007199254740992.0 or value >= 9007199254740992.0:
                return -1
            dst[count] = Int64(value)
        count += 1
        var finish = finish_element(data, n, after)
        i = finish[0]
        if i < 0:
            return -1
        if finish[1] == 1:
            return count if skip_space(data, n, i) == n else -1
    return -1


def json_bool_array(data: BPtr, n: Int, dst: BPtr, capacity: Int, strict: Bool) -> Int:
    var i = skip_space(data, n, 0)
    if i >= n or data[i] != UInt8(91):
        return -1
    i = skip_space(data, n, i + 1)
    if i < n and data[i] == UInt8(93):
        return 0 if skip_space(data, n, i + 1) == n else -1
    var count = 0
    while i < n and count < capacity:
        var original_start = skip_space(data, n, i)
        if original_start >= n:
            return -1
        var quoted = data[original_start] == UInt8(34)
        var bounds = numeric_bounds(data, n, i)
        var start = bounds[0]
        var end = bounds[1]
        var after = bounds[2]
        if start < 0 or (strict and quoted):
            return -1
        var value = UInt8(0)
        if equal_ascii(data, start, end, String("true")):
            value = UInt8(1)
        elif equal_ascii(data, start, end, String("false")):
            value = UInt8(0)
        elif not strict and (
            equal_ascii(data, start, end, String("1"))
            or equal_ascii(data, start, end, String("1.0"))
        ):
            value = UInt8(1)
        elif not strict and (
            equal_ascii(data, start, end, String("0"))
            or equal_ascii(data, start, end, String("0.0"))
        ):
            value = UInt8(0)
        elif not strict and quoted and (
            equal_ascii(data, start, end, String("on"))
            or equal_ascii(data, start, end, String("t"))
            or equal_ascii(data, start, end, String("yes"))
            or equal_ascii(data, start, end, String("y"))
        ):
            value = UInt8(1)
        elif not strict and quoted and (
            equal_ascii(data, start, end, String("off"))
            or equal_ascii(data, start, end, String("f"))
            or equal_ascii(data, start, end, String("no"))
            or equal_ascii(data, start, end, String("n"))
        ):
            value = UInt8(0)
        else:
            return -1
        dst[count] = value
        count += 1
        var finish = finish_element(data, n, after)
        i = finish[0]
        if i < 0:
            return -1
        if finish[1] == 1:
            return count if skip_space(data, n, i) == n else -1
    return -1


@export("mp_json_i64_array")
def mp_json_i64_array(data_addr: Int, n: Int, dst_addr: Int, capacity: Int, strict: Int) abi("C") -> Int:
    if data_addr == 0 or dst_addr == 0 or n <= 0 or capacity < 0:
        return -1
    return json_i64_array(
        BPtr(unsafe_from_address=data_addr),
        n,
        IPtr(unsafe_from_address=dst_addr),
        capacity,
        strict != 0,
    )


@export("mp_json_f64_array")
def mp_json_f64_array(data_addr: Int, n: Int, dst_addr: Int, capacity: Int, strict: Int) abi("C") -> Int:
    if data_addr == 0 or dst_addr == 0 or n <= 0 or capacity < 0:
        return -1
    if capacity >= 100000 and n >= 524288:
        var data = BPtr(unsafe_from_address=data_addr)
        var i = skip_space(data, n, 0)
        if i >= n or data[i] != UInt8(91):
            return -1
        i = skip_space(data, n, i + 1)
        if i < n and data[i] == UInt8(93):
            return 0 if skip_space(data, n, i + 1) == n else -1
        return json_f64_array_parallel(
            data_addr, n, dst_addr, capacity, strict != 0, i
        )
    return json_f64_array(
        BPtr(unsafe_from_address=data_addr),
        n,
        FPtr(unsafe_from_address=dst_addr),
        capacity,
        strict != 0,
    )


@export("mp_json_bool_array")
def mp_json_bool_array(data_addr: Int, n: Int, dst_addr: Int, capacity: Int, strict: Int) abi("C") -> Int:
    if data_addr == 0 or dst_addr == 0 or n <= 0 or capacity < 0:
        return -1
    return json_bool_array(
        BPtr(unsafe_from_address=data_addr),
        n,
        BPtr(unsafe_from_address=dst_addr),
        capacity,
        strict != 0,
    )
