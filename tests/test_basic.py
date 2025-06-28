import ast
import re
import os
from pathlib import Path

# Helper to load specific function definitions from doctr_ocr_to_csv.py without
# importing heavy dependencies.

def load_funcs(*names):
    src = Path('doctr_ocr_to_csv.py').read_text()
    module = ast.parse(src)
    ns = {'re': re, 'os': os}
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            code = ast.Module(body=[node], type_ignores=[])
            exec(compile(code, 'doctr_ocr_to_csv.py', 'exec'), ns)
    return [ns[name] for name in names]

safe_filename, normalize_ticket_number, get_manifest_validation_status, get_ticket_validation_status, build_roi_image_path = load_funcs(
    'safe_filename',
    'normalize_ticket_number',
    'get_manifest_validation_status',
    'get_ticket_validation_status',
    'build_roi_image_path'
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


def test_build_roi_image_path():
    path = build_roi_image_path('docs/sample.pdf', 1, '/out/images', '/out/output.csv', 'Vendor', '123')
    assert path == os.path.relpath('/out/images/sample/0001_Vendor_123_roi.png', start=os.path.dirname('/out/output.csv'))
    manifest_path = build_roi_image_path('docs/sample.pdf', 1, '/out/images', '/out/output.csv', 'Vendor', '123', 'manifest')
    assert manifest_path.endswith('sample/0001_Vendor_123_manifest_roi.png')
