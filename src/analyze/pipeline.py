import ast
import hashlib
import inspect
import json
import pickle
from datetime import timedelta
from pathlib import Path
import time

from analyze.util import normalize_path


class Pipeline:
    FORCE = True

    def __init__(self, pipeline, out=".out/", play_sound=True, min_cache_seconds=2.0):
        self.pipeline = pipeline
        self.out = normalize_path(out)
        self.min_cache_seconds = min_cache_seconds
        Path(self.out).mkdir(parents=True, exist_ok=True)
        self.stale = self._analyze_pipeline(pipeline, self.out)
        self.play_sound = play_sound

    @staticmethod
    def _get_source(fn):
        try:
            return inspect.getsource(fn)
        except (OSError, TypeError):
            return f"<no-source:{fn.__qualname__}>"

    @staticmethod
    def _function_calls(tree, visited=None, module_path=None):
        """
        >>> def b():
        ...     b()
        ...     pass
        >>> def a():
        ...     b()
        >>> [e.__name__ for e in Pipeline._function_calls(a, module_path={"a": a, "b": b})]
        ['a', 'b']
        """
        if visited is None:
            visited = set()

        if module_path is None:
            module_path = getattr(tree, "__globals__")

        if callable(tree) and tree not in visited:
            visited.add(tree)
            yield tree
            try:
                parsed = ast.parse(inspect.getsource(tree).strip())
                yield from Pipeline._function_calls(parsed, visited, module_path)
            except (OSError, TypeError):
                pass  # built-in or C extension — can't recurse, identity already yielded
        if isinstance(tree, ast.Call):
            func = tree.func
            if not isinstance(func, ast.Name):
                return

            name = func.id
            if (f := module_path.get(name)) is not None:
                yield from Pipeline._function_calls(f, visited, module_path)
        elif isinstance(tree, ast.Expr):
            yield from Pipeline._function_calls(tree.value, visited, module_path)
        elif hasattr(tree, "body"):
            body = tree.body
            for expr in body:
                yield from Pipeline._function_calls(expr, visited, module_path)

    @staticmethod
    def _analyze_pipeline(pipeline, path: str):
        stale = set()

        digest_path = Path(path + "sig.json")
        if digest_path.exists():
            with open(digest_path, "r") as f:
                digests = json.load(f)
        else:
            digests = {}
        new_digests = {}

        for stage in pipeline:
            if not callable(stage):
                raise Exception(f"Stage {stage} is not callable.")

            digest = hashlib.sha512("".join([
                Pipeline._get_source(e)
                for e in Pipeline._function_calls(stage)
            ]).encode()).hexdigest()

            if digests.get(stage.__name__) == digest:
                stale.add(stage)
            new_digests[stage.__name__] = digest

        with open(digest_path, "w") as f:
            json.dump(new_digests, f, indent=3)

        return stale

    def process(self, data):
        data_signature_path = Path(self.out + "input.sha512")
        data_signature = hashlib.sha512(pickle.dumps(data)).hexdigest()
        force = Pipeline.FORCE or not (data_signature_path.exists() and data_signature_path.read_text() == data_signature)
        data_signature_path.write_text(data_signature, encoding="utf-8")

        pipeline_path = Path(self.out + "pipe.json")
        if pipeline_path.exists():
            with open(pipeline_path, "r") as f:
                old_pipe = json.load(f)
        else:
            old_pipe = []


        start_t = time.perf_counter()

        for i, stage in enumerate(self.pipeline):
            stale = stage in self.stale
            force = force or not stale or i >= len(old_pipe) or stage.__name__ != old_pipe[i]

            serial_path = self.out + f"{stage.__name__}.pkl"

            if force or not Path(serial_path).exists():
                print()
                print(f"---- Stage {i} - {stage.__name__} (Running...) ----")
                stage_start_t = time.perf_counter()
                try:
                    data = stage(data)
                except Exception as e:
                    # playsound(f"{PROJECT_DIR}/bell_sound.mp3")
                    raise e

                stage_elapsed = time.perf_counter() - stage_start_t
                if stage_elapsed >= self.min_cache_seconds:
                    with open(serial_path, "wb") as file:
                        pickle.dump(data, file)
                force = True  # stage ran; downstream must recompute regardless of their pkls

                print(f"(Stage took: {stage_elapsed * 1000:.0f}ms)")
            else:
                print(f"---- Stage {i} - {stage.__name__} (Cache-Hit) ----")
                with open(serial_path, "rb") as file:
                    data = pickle.load(file)

        with open(pipeline_path, "w") as f:
            json.dump([e.__name__ for e in self.pipeline], f, indent=3)

        print()
        total_elapsed = time.perf_counter() - start_t
        t = timedelta(seconds=total_elapsed)
        print(f"---- All done (Took {t}). ----")
        if t.seconds > 5 and self.play_sound:
            pass
            # playsound(f"{PROJECT_DIR}bell_sound.mp3")
        return data