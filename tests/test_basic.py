import ast
import re
from pathlib import Path

# Helper to load specific function definitions from doctr_ocr_to_csv.py without
# importing heavy dependencies.

def load_funcs(*names):
    src = Path('doctr_ocr_to_csv.py').read_text()
    module = ast.parse(src)
    ns = {'re': re}
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            code = ast.Module(body=[node], type_ignores=[])
            exec(compile(code, 'doctr_ocr_to_csv.py', 'exec'), ns)
    return [ns[name] for name in names]

safe_filename, normalize_ticket_number, get_manifest_validation_status, get_ticket_validation_status = load_funcs(
    'safe_filename',
    'normalize_ticket_number',
    'get_manifest_validation_status',
    'get_ticket_validation_status'
)


def test_safe_filename():
    assert safe_filename('hello world.pdf') == 'hello_world_pdf'
    assert safe_filename('abc@#$% 123') == 'abc_____123'


def test_normalize_ticket_number():
    assert normalize_ticket_number(None) is None
    assert normalize_ticket_number('NO 12345') == '12345'
    assert normalize_ticket_number('A 038270') == 'A038270'
    assert normalize_ticket_number(' 123 ') == '123'


def test_manifest_validation_status():
    assert get_manifest_validation_status(None) == 'invalid'
    assert get_manifest_validation_status('14999999') == 'valid'
    assert get_manifest_validation_status('1499999') == 'review'
    assert get_manifest_validation_status('123') == 'invalid'


def test_ticket_validation_status():
    assert get_ticket_validation_status('123', r'\d+') == 'valid'
    assert get_ticket_validation_status('abc', r'\d+') == 'invalid'
    assert get_ticket_validation_status(None, r'\d+') == 'invalid'
    assert get_ticket_validation_status('123', None) == 'not checked'
