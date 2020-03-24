"""Imperative interpreter. (prototype)"""


from ..fpbench import fpcast as ast

# integrated testing
from ..fpbench import fpcparser

def preorder(e):
    s = []
    s.append(e)
    while s:
        current = s.pop()
        print(current)
        try:
            for child in reversed(current.children):
                s.append(child)
        except AttributeError:
            pass

def postorder(e):
    s = []
    current = e
    last = None
    while s or current:
        if current:
            s.append(current)
            try:
                current = current.children[0]
            except AttributeError:
                current = None

        else:
            peek = s[-1]
            if (hasattr(peek, 'children') and len(peek.children) == 2 and last is not peek.children[1]):
                current = peek.children[1]
            else:
                last = s.pop()
                print(last)
                current = None


def p2(e):
    last_stack = []
    next_stack = [e]
    result_stack = []
    while last_stack or next_stack:
        if next_stack:
            current = next_stack.pop()
            last_stack.append(current)
            try:
                next_stack += current.children
                print('<- ', current)
            except AttributeError:
                print(' - ', current)
        else:
            current = last_stack.pop()

            try:
                child_count = len(current.children)
            except AttributeError:
                child_count = 0

            total = 1
            for i in range(child_count):
                total += result_stack.pop()
            print(total, child_count, current)
            result_stack.append(total)

    print(result_stack)



core = fpcparser.compile1(
"""(FPCore
 (x)
 :name
 "NMSE problem 3.3.1"
 :cite
 (hamming-1987 herbie-2015)
 :fpbench-domain
 textbook
 :pre
 (!= x 0)
 (- (/ 1 (+ x 1)) (! :precision binary32 (/ 1 x))))
""")

preorder(core.e)
print()
postorder(core.e)
print()
p2(core.e)

print()
print(core)
print(core.e)
e2 = core.e.copy()

e3 = core.e.canonicalize_annotations()
print(e3)

e4 = core.e.canonicalize_annotations({'precision': 'binary64'})
print(e4)

e5 = e4.condense_annotations()
print(e5)

e6 = e4.remove_annotations()
print(e6)
