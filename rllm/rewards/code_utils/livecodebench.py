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

# Robust BASE_IMPORTS import (works even if relative import fails in spawned child)
BASE_IMPORTS = ""

# ======================================================================================
# Configuration
# ======================================================================================

OUTPUT_BYTE_CAP = 4_000_000        # 4 MB stdout cap
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
        sys.setrecursionlimit(CHILD_RECURSION_LIMIT)
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

def _worker_call_based(code: str, fn_name: str, gt_inp, gt_out, timeout: int):
    start = time.time()
    try:
        _child_setup(timeout, DEFAULT_MEMORY_BYTES)
        code = BASE_IMPORTS + "\n" + code
        compiled_sol = compile_code(code, timeout)
        method = get_function(compiled_sol, fn_name)
        if method is None:
            return -4, {"error_code": -4, "error_message": "Method not found"}, 0.0

        prediction = method(*gt_inp)
        exec_time = time.time() - start

        if isinstance(prediction, tuple):
            prediction = list(prediction)
      #  print("predictionn", prediction)
      #  print("gt_out", gt_out)
        if prediction == gt_out:
            return True, {}, exec_time

        return -2, {
            "output": safe_repr(prediction),
            "inputs": safe_repr(gt_inp),
            "expected": safe_repr(gt_out),
            "error_code": -2,
            "error_message": "Wrong Answer",
        }, exec_time

    except TimeoutException as e:
        return -3, {
            "error": repr(e),
            "error_code": -3,
            "error_message": "Time Limit Exceeded",
            "inputs": safe_repr(gt_inp),
            "expected": safe_repr(gt_out),
        }, 0.0
    except OutputTooLarge as e:
        return -2, {
            "error": repr(e),
            "error_code": -2,
            "error_message": "Wrong Answer: output too long",
            "inputs": safe_repr(gt_inp),
            "expected": safe_repr(gt_out),
        }, 0.0
    except Exception as e:
        return -4, {
            "error": repr(e),
            "error_code": -4,
            "error_message": "Runtime Error",
            "inputs": safe_repr(gt_inp),
            "expected": safe_repr(gt_out),
        }, 0.0
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass

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
    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0.0
    all_results = []

    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        flag, payload, exec_time = _spawn_and_run(
            _worker_call_based, (code, fn_name, gt_inp, gt_out, timeout), timeout
        )
        if flag is True:
            all_results.append(True)
            total_execution += exec_time
            continue

        all_results.append(flag)
        if "error_message" not in payload and flag == -2:
            payload["error_message"] = "Wrong Answer"
        if "inputs" not in payload:
            payload["inputs"] = truncatefn(gt_inp)
        if "expected" not in payload:
            payload["expected"] = truncatefn(gt_out)
        return all_results, payload

    return all_results, {"execution time": total_execution}

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
