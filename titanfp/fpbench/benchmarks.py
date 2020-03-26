"""Infrastructure to provide FPBench tests and benchmarks."""


import traceback

from . import toolserver
from . import fpcparser


class Benchmarks(object):
    """All of the FPBench benchmarks."""

    def _import_cores(self, paths):
        cores = []
        for path in paths:
            try:
                cores += fpcparser.compfile(path)
            except fpcparser.FPCoreParserError:
                print('='*80)
                print('Unable to parse file {}'.format(path))
                traceback.print_exc()
                print('='*80)
                print(flush=True)
        return cores

    def __init__(self, fpbench_path=None):
        if fpbench_path is None:
            self.fpbench_path = toolserver.fpbench_path
        else:
            self.fpbench_path = fpbench_path

        benchmarks, sanity, tests = toolserver.locate_fpbench_benchmarks(self.fpbench_path)
        self.benchmark_paths = benchmarks
        self.sanity_paths = sanity
        self.test_paths = tests

        self.benchmarks = self._import_cores(self.benchmark_paths)
        self.sanity = self._import_cores(self.sanity_paths)
        self.tests = self._import_cores(self.test_paths)
