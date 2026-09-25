# -*- coding: utf-8 -*-
import re, difflib

v2 = open('code/solve_multicore_v2.py', encoding='utf-8').read()
v3 = open('code/solve_multicore_v3.py', encoding='utf-8').read()
print('v2 行数:', v2.count('\n') + 1, ' v3 行数:', v3.count('\n') + 1)

for name, txt in [('v2', v2), ('v3', v3)]:
    print('---', name, '---')
    m = re.search(r'"""(.*?)"""', txt, re.S)
    if m:
        print('docstring:', m.group(1)[:300].strip())
    for fn in re.findall(r'def (\w+)\(', txt):
        print('  def', fn)

# 关键差异行（只比较去掉空行后的骨架）
print()
print('=== 主要策略类/函数级差异（粗对比）===')
for line in difflib.unified_diff(v2.splitlines(), v3.splitlines(), lineterm='', n=0):
    if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
        print(line[:150])
