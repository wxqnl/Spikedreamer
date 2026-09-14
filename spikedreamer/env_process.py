"""Run real DMC physics and rendering without a CUDA training context."""

import os
import subprocess
import sys
import traceback
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from types import SimpleNamespace


class ProcessDMC:
    """Synchronous DMC transport; action order and environment semantics stay local."""

    def __init__(self, config, seed):
        self._closed = False
        self._broken = False
        self._pending = None
        self._connection, remote = Pipe(duplex=True)
        environment = dict(os.environ, SPIKEDREAMER_ENV_PROCESS="0",
                           CUDA_VISIBLE_DEVICES="")
        preload = os.environ.get("SPIKEDREAMER_ENV_LD_PRELOAD")
        if preload:
            environment["LD_PRELOAD"] = preload
        try:
            self._worker = subprocess.Popen(
                [sys.executable, "-u", "-m", "spikedreamer.env_process",
                 str(remote.fileno())],
                pass_fds=(remote.fileno(),), env=environment,
                # Inherit the training process group and log. The supervisor
                # can stop the whole owned run, including its render workers.
                start_new_session=False)
        except BaseException:
            self._connection.close()
            raise
        finally:
            remote.close()
        try:
            self.worker_info = self._call("init", vars(config), seed)
            self.actions = self.worker_info["actions"]
        except BaseException:
            self.close()
            raise

    def _send(self, command, *args):
        if self._closed or self._broken:
            raise RuntimeError("DMC worker is closed or failed")
        if self._pending is not None:
            raise RuntimeError(f"DMC worker is still processing {self._pending}")
        try:
            self._connection.send((command, args))
        except (EOFError, OSError) as error:
            self._broken = True
            raise RuntimeError(f"DMC render worker {self._worker.pid} failed to accept {command}") from error
        self._pending = command

    def _receive(self):
        if self._pending is None:
            raise RuntimeError("DMC worker has no pending request")
        command = self._pending
        try:
            if not self._connection.poll(120):
                raise TimeoutError(f"DMC worker timed out during {command}")
            success, result = self._connection.recv()
        except (EOFError, OSError, TimeoutError) as error:
            self._broken = True
            code = self._worker.poll()
            raise RuntimeError(
                f"DMC render worker {self._worker.pid} failed during {command}; "
                f"exit_code={code}: {error}") from error
        finally:
            self._pending = None
        if not success:
            self._broken = True
            raise RuntimeError(f"DMC render worker failed during {command}:\n{result}")
        return result

    def _call(self, command, *args):
        self._send(command, *args)
        return self._receive()

    def reset(self):
        return self._call("reset")

    def step(self, action):
        return self._call("step", action)

    def step_async(self, action):
        self._send("step", action)

    def step_wait(self):
        if self._pending != "step":
            raise RuntimeError("DMC worker has no pending step")
        return self._receive()

    def close(self):
        if self._closed:
            return
        try:
            if not self._broken and self._pending is None and self._worker.poll() is None:
                self._call("close")
        except RuntimeError:
            pass
        finally:
            self._closed = True
            self._connection.close()
            try:
                self._worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._worker.terminate()
                try:
                    self._worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._worker.kill()
                    self._worker.wait()


def worker_main(fd):
    # No torch import or CUDA initialization in this process. CUDA visibility
    # does not select EGL: retain the parent's explicit physical EGL device.
    from .env import DMC

    connection = Connection(fd)
    env = None
    try:
        while True:
            command, args = connection.recv()
            try:
                if command == "init" and env is None:
                    config, seed = args
                    env = DMC(SimpleNamespace(**config), seed)
                    result = dict(actions=env.actions, pid=os.getpid(),
                                  egl_device=os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                                  backend=os.environ.get("MUJOCO_GL", "egl"),
                                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                  torch_imported="torch" in sys.modules,
                                  ld_preload=os.environ.get("LD_PRELOAD"))
                elif command == "reset" and env is not None:
                    result = env.reset()
                elif command == "step" and env is not None:
                    result = env.step(*args)
                elif command == "close":
                    if env is not None:
                        env.close()
                        env = None
                    result = None
                else:
                    raise ValueError(f"Invalid DMC worker command: {command}")
            except Exception:
                connection.send((False, traceback.format_exc()))
                break
            connection.send((True, result))
            if command == "close":
                break
    except (EOFError, BrokenPipeError):
        # A dead trainer closes its pipe. Do not leave renderer processes behind.
        pass
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            connection.close()


if __name__ == "__main__":
    worker_main(int(sys.argv[1]))
