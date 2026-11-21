"""Immutable proposal identity and bounded submission waits."""
import ast
import hashlib
import time

def program_identity(source):
    """Ignore comments and docstrings, so cosmetic text cannot satisfy novelty."""
    class RemoveDocstrings(ast.NodeTransformer):
        def visit(self, node):
            node = super().visit(node)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body = node.body[1:]
            return node
    try:
        tree = RemoveDocstrings().visit(ast.parse(source))
        canonical = ast.dump(tree, include_attributes=False)
    except SyntaxError:
        canonical = source.strip()
    return hashlib.sha256(canonical.encode()).hexdigest()

def wait_file(path,timeout=3600):
 start=time.monotonic()
 while not path.exists():
  if time.monotonic()-start>timeout:raise TimeoutError(f'waiting for {path.name}')
  time.sleep(.25)
