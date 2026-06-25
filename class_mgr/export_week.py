"""
Usage: python export_week.py <주차> [notebook.ipynb]

Examples:
  python export_week.py 12
  python export_week.py 12주차
  python export_week.py 11 hello.ipynb
"""
import json, re, sys
from pathlib import Path

def export_week(week: str, notebook: str = "hello.ipynb"):
    # "12" 또는 "12주차" 둘 다 허용
    week = week.strip().removesuffix("주차")
    header = f"# {week}주차"

    nb_path = Path(notebook)
    if not nb_path.exists():
        print(f"파일을 찾을 수 없습니다: {nb_path}")
        sys.exit(1)

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    in_section = False
    code_blocks = []

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            stripped = src.strip()
            if stripped == header:
                in_section = True
            elif in_section and re.match(r"^#\s+\d+주차", stripped):
                in_section = False
        elif cell["cell_type"] == "code" and in_section:
            if src.strip():
                code_blocks.append(src)

    if not code_blocks:
        print(f"'{header}' 섹션을 찾을 수 없거나 코드 셀이 없습니다.")
        sys.exit(1)

    output_path = nb_path.parent / f"week{week}.py"
    content = "\n\n".join("# %%\n" + block for block in code_blocks)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"저장 완료: {output_path} ({len(code_blocks)}개 셀)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    week_arg = sys.argv[1]
    notebook_arg = sys.argv[2] if len(sys.argv) >= 3 else "hello.ipynb"
    export_week(week_arg, notebook_arg)
