# OOM-safe, spawn/fork-safe drop-in replacement for testing_util.py

import ast
import json
import sys
import faulthandler
import platform
import multiprocessing
import queue
import signal
import time
import io
from io import StringIO
from unittest.mock import patch, mock_open
from types import ModuleType
from enum import Enum
from decimal import Decimal
from reprlib import Repr

# Optional import for parity with original
try:
    import numpy as np  # noqa: F401
except Exception:
    pass

from .utils import BASE_IMPORTS

# ======================================================================================
# Configuration
# ======================================================================================

OUTPUT_BYTE_CAP = 2_000_000        # 4 MB stdout cap
MAX_LINES_CAP = 1000            # Max output lines allowed
MAX_LINE_CHARS = 200_000           # Max characters per output line
MAX_TOKENS_PER_LINE = 10_000       # Max tokens per line for numeric compare
DEFAULT_MEMORY_BYTES = 2 * 1024**3 # 2 GiB child address-space cap (if supported)
CHILD_RECURSION_LIMIT = 2000       # Reasonable recursion depth

# ======================================================================================
# Utilities
# ======================================================================================

import_string = BASE_IMPORTS

def truncatefn(s, length=300):
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]

_R = Repr()
_R.maxstring = 300
_R.maxlist = 20
_R.maxdict = 10
_R.maxarray = 20
_R.maxlevel = 2
def safe_repr(x):
    try:
        return _R.repr(x)
    except Exception:
        return truncatefn(x, 300)

class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Time limit exceeded")

# ======================================================================================
# Capped stdout capture
# ======================================================================================

class OutputTooLarge(Exception):
    pass

class _CappedBuffer(io.TextIOBase):
    def __init__(self, cap_bytes: int):
        super().__init__()
        self.cap = int(cap_bytes)
        self._n = 0
        self._chunks = []

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        b = s.encode("utf-8", "replace")
        nb = len(b)
        if self._n + nb > self.cap:
            allowed = self.cap - self._n
            if allowed > 0:
                self._chunks.append(b[:allowed].decode("utf-8", "replace"))
                self._n = self.cap
            raise OutputTooLarge(f"stdout exceeded {self.cap} bytes")
        self._chunks.append(s)
        self._n += nb
        return len(s)

    def getvalue(self):
        return "".join(self._chunks)

    @property
    def nbytes(self):
        return self._n

class Capturing:
    def __init__(self, byte_cap=OUTPUT_BYTE_CAP):
        self.byte_cap = byte_cap
        self._stdout = None
        self._sink = None
        self.text = ""

    def __enter__(self):
        self._stdout = sys.stdout
        self._sink = _CappedBuffer(self.byte_cap)
        sys.stdout = self._sink
        return self

    def __exit__(self, *args):
        try:
            self.text = self._sink.getvalue()
        finally:
            sys.stdout = self._stdout
            self._sink = None
            self._stdout = None

# ======================================================================================
# Helpers
# ======================================================================================

def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
    except Exception:
        pass
    return code

def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception:
        return code

def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            return None
    return _inner_call_method(method)

def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None

def compile_code(code: str, timeout: int):
    # Note: signal.alarm not available on Windows; parent wall-time still applies
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, int(timeout)))
    except Exception:
        pass
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass
    return compiled_sol

def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line

def get_stripped_lines(val: str):
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]

# ======================================================================================
# Child hardening
# ======================================================================================

def reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        try:
            import resource  # Unix only
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
            if not platform.uname().system == "Darwin":
                resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
        except Exception:
            pass

    faulthandler.disable()

    import builtins
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

def _child_setup(time_limit_sec: int, memory_bytes: int | None):
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, int(time_limit_sec)))
    except Exception:
        pass
    try:
        # Preserve (or raise to) the higher of the current limit and 50k
        cur = sys.getrecursionlimit()
        target = max(cur, 50000)
        sys.setrecursionlimit(target)
    except Exception:
        pass
    reliability_guard(memory_bytes)

# ======================================================================================
# Worker functions (top-level: picklable)
# ======================================================================================

def _try_numeric_line_compare(pred_line: str, gt_line: str):
    if len(pred_line) > MAX_LINE_CHARS or len(gt_line) > MAX_LINE_CHARS:
        return False
    if not any(ch.isdigit() for ch in pred_line) or not any(ch.isdigit() for ch in gt_line):
        return False
    pred_tokens = pred_line.split()
    gt_tokens = gt_line.split()
    if len(pred_tokens) > MAX_TOKENS_PER_LINE or len(gt_tokens) > MAX_TOKENS_PER_LINE:
        return False
    ok1, dec_pred = convert_line_to_decimals(pred_line)
    ok2, dec_gt = convert_line_to_decimals(gt_line)
    if not (ok1 and ok2):
        return False
    return dec_pred == dec_gt


def _worker_call_based_suite_entry(q, code, fn_name, parsed_inputs, parsed_outputs, timeout):
    """
    Child-process entry: compile once, then run each test case with a per-case alarm.
    Sends (all_results, metadata) back through the queue and exits.
    all_results is a list of True/False/(-3)/(-4) mirroring original semantics:
      - True  : correct
      - False : wrong answer (and we return early with error_code -2)
      - -3    : time limit exceeded (return early with error_code -3)
      - -4    : runtime error (return early with error_code -4)
    """
    # Helper: bounded repr if available, otherwise fall back to truncatefn
    def _sr(x):
        try:
            return safe_repr(x)   # present in the hardened file; ignored if not
        except Exception:
            return truncatefn(x)

    # Set up child environment similarly to your original runner
    try:
        # Per-process hardening (no memory cap here; use your reliability_guard signature)
        reliability_guard()
    except Exception:
        pass

    # Try to have a generous recursion limit if your BASE_IMPORTS does not set it
    try:
        cur = sys.getrecursionlimit()
        if cur < 50000:
            sys.setrecursionlimit(50000)
    except Exception:
        pass

    # Compile once
    try:
        # Keep parity with original: prepend BASE_IMPORTS
        compiled_sol = compile_code(BASE_IMPORTS + "\n" + code, timeout)
        method = get_function(compiled_sol, fn_name)
        if method is None:
            q.put(([-4], {
                "error_code": -4,
                "error_message": "Method not found",
            }))
            return
    except TimeoutException as e:
        q.put(([-3], {
            "error": repr(e),
            "error_code": -3,
            "error_message": "Time Limit Exceeded during compile",
        }))
        return
    except Exception as e:
        q.put(([-4], {
            "error": repr(e),
            "error_code": -4,
            "error_message": "Runtime Error during compile",
        }))
        return

    all_results = []
    total_execution = 0.0

    # Run each test case with its own alarm
    for gt_inp, gt_out in zip(parsed_inputs, parsed_outputs):
        try:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(max(1, int(timeout)))
            except Exception:
                # SIGALRM not available on some platforms; parent will still enforce a suite wall clock.
                pass

            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start

            try:
                signal.alarm(0)
            except Exception:
                pass

            if isinstance(prediction, tuple):
                prediction = list(prediction)

            if prediction == gt_out:
                all_results.append(True)
                continue

            # Wrong answer: return early, mirroring original behavior
            all_results.append(False)
            q.put((all_results, {
                "output": _sr(prediction),
                "inputs": _sr(gt_inp),
                "expected": _sr(gt_out),
                "error_code": -2,
                "error_message": "Wrong Answer",
            }))
            return

        except TimeoutException as e:
            try:
                signal.alarm(0)
            except Exception:
                pass
            all_results.append(-3)
            q.put((all_results, {
                "error": repr(e),
                "error_code": -3,
                "error_message": "Time Limit Exceeded",
                "inputs": _sr(gt_inp),
                "expected": _sr(gt_out),
            }))
            return
        except Exception as e:
            try:
                signal.alarm(0)
            except Exception:
                pass
            all_results.append(-4)
            q.put((all_results, {
                "error": repr(e),
                "error_code": -4,
                "error_message": "Runtime Error",
                "inputs": _sr(gt_inp),
                "expected": _sr(gt_out),
            }))
            return

    # All passed
    q.put((all_results, {"execution time": total_execution}))



def _worker_stdio(code: str, gt_inp: str, gt_out: str, timeout: int):
    start = time.time()
    try:
        _child_setup(timeout, DEFAULT_MEMORY_BYTES)
        code = BASE_IMPORTS + "\n" + code
        compiled_mod = compile_code(code, timeout)
        method = get_function(compiled_mod, "solve")
        if method is None:
            return -4, {"error_code": -4, "error_message": "No solve() found"}, 0.0

        with Capturing(byte_cap=OUTPUT_BYTE_CAP) as cap:
            ret = call_method(method, gt_inp)

        exec_time = time.time() - start

        prediction = cap.text
        if not prediction.strip() and ret is not None:
            prediction = safe_repr(ret)

        # IMPORTANT: strip BOTH sides per line (fixes "all wrong" reward=0 issue)
        pred_lines = get_stripped_lines(prediction)
        gt_lines   = get_stripped_lines(gt_out)
       # print("pred_lines", pred_lines)
       # print("gt_lines", gt_lines)

        if len(pred_lines) > MAX_LINES_CAP or len(gt_lines) > MAX_LINES_CAP:
            return -2, {
                "output": safe_repr(prediction),
                "inputs": safe_repr(gt_inp),
                "expected": safe_repr(gt_out),
                "error_code": -2,
                "error_message": "Wrong answer: output too long",
            }, exec_time

        if len(pred_lines) != len(gt_lines):
            return -2, {
                "output": safe_repr(prediction),
                "inputs": safe_repr(gt_inp),
                "expected": safe_repr(gt_out),
                "error_code": -2,
                "error_message": "Wrong answer: mismatched output length",
            }, exec_time

        for idx, (pl, gl) in enumerate(zip(pred_lines, gt_lines)):
            if pl == gl:
                continue
            if _try_numeric_line_compare(pl, gl):
                continue
            return -2, {
                "output": truncatefn(pl),
                "inputs": truncatefn(gt_inp),
                "expected": truncatefn(gl),
                "error_code": -2,
                "error_message": f"Wrong answer at output_line_idx={idx}: {truncatefn(pl)} != {truncatefn(gl)}",
            }, exec_time

        return True, {}, exec_time

    except TimeoutException as e:
        return -3, {
            "error": repr(e),
            "error_code": -3,
            "error_message": "Time Limit Exceeded",
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
        }, 0.0
    except OutputTooLarge as e:
        return -2, {
            "error": repr(e),
            "error_code": -2,
            "error_message": "Wrong answer: output too long",
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
        }, 0.0
    except Exception as e:
        return -4, {
            "error": repr(e),
            "error_code": -4,
            "error_message": "Runtime Error",
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
        }, 0.0
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass

# ======================================================================================
# Spawn/Fork orchestration (top-level entry for pickling)
# ======================================================================================

def _entry(q, target, args):
    """Top-level entry point (picklable) for child process."""
    try:
        res = target(*args)
        q.put(res)
    except BaseException as e:
        q.put((
            -4,
            {"error_code": -4, "error_message": f"Worker crashed: {repr(e)}"},
            0.0
        ))

def _get_ctx():
    # Prefer fork when available (POSIX) for import/relative-import friendliness
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context("spawn")

def _spawn_and_run(target, args: tuple, timeout: int):
    ctx = _get_ctx()
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_entry, args=(q, target, args))
    p.start()
    p.join(timeout + 1)  # small buffer

    if p.is_alive():
        try:
            p.kill()
        except Exception:
            pass
        return -3, {"error_code": -3, "error_message": "Time Limit Exceeded"}, 0.0

    try:
        return q.get_nowait()
    except queue.Empty:
        return -4, {"error_code": -4, "error_message": "No result from worker"}, 0.0

# ======================================================================================
# Graders
# ======================================================================================

def grade_call_based(code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int):
    """
    Compile once and evaluate all call-based tests in a single child process.
    - Keeps cross-test state (important for some user solutions).
    - Returns early on first failure (same as original behavior).
    - Results list contains True for passes, False for first WA, or -3/-4 for TLE/RE.
    - Metadata matches original keys (error_code, error_message, inputs, expected) or
      {"execution time": total_seconds} if all pass.
    """
    # Parse inputs/outputs exactly like your original implementation
    parsed_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    parsed_outputs = [json.loads(output) for output in all_outputs]

    # Prefer fork on POSIX (keeps import context); fall back to spawn elsewhere
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        ctx = multiprocessing.get_context("spawn")

    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_worker_call_based_suite_entry,
        args=(q, code, fn_name, parsed_inputs, parsed_outputs, timeout),
    )

    # Wall-clock budget for the whole suite
    suite_timeout = max(1, int(timeout)) * max(1, len(parsed_inputs)) + 2

    p.start()
    p.join(suite_timeout)

    if p.is_alive():
        # Kill runaway child (e.g., platform without SIGALRM)
        try:
            p.kill()
        except Exception:
            pass
        return [-3], {
            "error_code": -3,
            "error_message": "Time Limit Exceeded (suite)"
        }

    try:
        results, meta = q.get_nowait()
        return results, meta
    except queue.Empty:
        return [-4], {
            "error_code": -4,
            "error_message": "No result from worker"
        }

def grade_stdio(code: str, all_inputs: list, all_outputs: list, timeout: int):
    all_results = []
    total_execution_time = 0.0

    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        flag, payload, exec_time = _spawn_and_run(
            _worker_stdio, (code, gt_inp, gt_out, timeout), timeout
        )
        if flag is True:
            all_results.append(True)
            total_execution_time += exec_time
            continue

        all_results.append(flag)
        return all_results, payload

    return all_results, {"execution time": total_execution_time}

# ======================================================================================
# Public entry point
# ======================================================================================

def run_test(sample, test=None, debug=False, timeout=6):
    if debug:
        print(f"start = {time.strftime('%H:%M:%S')}")

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError as e:
        raise e

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based
            method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output at {time.strftime('%H:%M:%S')}")

    if test is None:
        return in_outs, {"error": "no test code provided", "error_code": -4}

    if which_type == CODE_TYPE.call_based:
        try:
            results, metadata = grade_call_based(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                fn_name=method_name,
                timeout=timeout,
            )
            return results, metadata
        except Exception as e:
            print("exp1", e)

            return [-4], {"error_code": -4, "error_message": f"Error during testing: {e}"}

    elif which_type == CODE_TYPE.standard_input:
        try:
            results, metadata = grade_stdio(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                timeout=timeout,
            )
            return results, metadata
        except Exception as e:
            print("exp2", e)
            return [-4], {"error_code": -4, "error_message": f"Error during testing: {e}"}
