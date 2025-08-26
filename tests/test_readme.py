import re
import os

def test_readme():
    with open("README.md") as f:
        text = f.read()

    # Find code blocks
    pattern = r"```python\n(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)

    for code in code_blocks:
        exec(code)

    os.remove("data.safeserialize")
