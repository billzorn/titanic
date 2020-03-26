"""Communication between Titanic and FPBench."""


import os


here = os.path.dirname(os.path.realpath(__file__))
titanfp_top = os.path.normpath(os.path.join(here, '..', '..'))


def isfpbench(path):
    return (os.path.isdir(path)
            and os.path.isdir(os.path.join(path, 'benchmarks'))
            and os.path.isdir(os.path.join(path, 'tests'))
            and os.path.isfile(os.path.join(path, 'toolserver.rkt')))


def locate_fpbench_repo():
    parent = os.path.normpath(os.path.join(titanfp_top, '..'))
    siblings = os.listdir(parent)

    priority_1 = [name for name in siblings if name.lower() == 'fpbench']
    priority_2 = [name for name in siblings if name.lower() != 'fpbench' and 'fpbench' in name.lower()]
    priority_3 = [name for name in siblings if 'fpbench' not in name.lower()]

    to_check = sorted(priority_1) + sorted(priority_2) + sorted(priority_3)

    for name in to_check:
        path = os.path.join(parent, name)
        if isfpbench(path):
            return path

    return None

def locate_fpbench_benchmarks(repo):
    benchmark_dir = os.path.join(repo, 'benchmarks')
    benchmarks = [os.path.join(benchmark_dir, name) for name in os.listdir(benchmark_dir) if name.lower().endswith('fpcore')]
    test_dir = os.path.join(repo, 'tests')
    sanity = [os.path.join(test_dir, name)
              for name in os.listdir(test_dir)
              if name.lower().startswith('sanity') and name.lower().endswith('fpcore')]
    tests = [os.path.join(test_dir, name)
             for name in os.listdir(test_dir)
             if name.lower().startswith('test') and name.lower().endswith('fpcore')]
    return benchmarks, sanity, tests


fpbench_path = locate_fpbench_repo()
