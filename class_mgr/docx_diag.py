#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOCX 텍스트/이미지 구조 진단 스크립트
사용: python docx_diag.py <파일.docx>
"""
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


def local(tag: str) -> str:
    return tag.split('}')[-1] if '}' in tag else tag


def iter_row_cells(row_elem):
    """행 내 w:tc를 재귀 탐색. w:sdt 래퍼도 통과. 중첩 w:tbl은 건너뜀."""
    for child in row_elem:
        loc = local(child.tag)
        if loc == 'tc':
            yield child
        elif loc != 'tbl':
            yield from iter_row_cells(child)


def iter_cell_paras(elem):
    """셀 내 w:p를 재귀 탐색. 중첩 w:tbl은 건너뜀."""
    for child in elem:
        loc = local(child.tag)
        if loc == 'p':
            yield child
        elif loc != 'tbl':
            yield from iter_cell_paras(child)


def para_lines(para):
    """단락 하나에서 줄 목록 반환. w:br=줄바꿈, w:tab=탭."""
    buf = []
    lines = []
    for el in para.iter():
        loc = local(el.tag)
        if loc == 't' and el.text is not None:
            buf.append(el.text)
        elif loc == 'tab':
            buf.append('\t')
        elif loc == 'br':
            lines.append(''.join(buf))
            buf = []
    lines.append(''.join(buf))
    return lines


def diag(docx_path: str):
    p = Path(docx_path)
    if not p.exists():
        print(f"파일 없음: {p}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"진단 대상: {p.name}")
    print(f"{'='*60}")

    with zipfile.ZipFile(str(p)) as z:
        names = z.namelist()

        # ── 1. 미디어 파일 목록 ───────────────────────────
        media = [n for n in names if 'media' in n]
        print(f"\n[미디어 파일] {len(media)}개")
        for m in media:
            print(f"  {m}")

        # ── 2. document.xml 파싱 ─────────────────────────
        doc_name = 'word/document.xml'
        if doc_name not in names:
            print(f"\n{doc_name} 없음")
            return

        root = ET.fromstring(z.read(doc_name).decode('utf-8', errors='replace'))

        # ── 3. 표 구조 전체 출력 ─────────────────────────
        body = next((el for el in root.iter() if local(el.tag) == 'body'), root)
        tables = [el for el in body if local(el.tag) == 'tbl']
        print(f"\n[표(tbl) 수] {len(tables)}개")

        for ti, tbl in enumerate(tables):
            rows = [el for el in tbl if local(el.tag) == 'tr']
            print(f"\n  ┌── 표 {ti+1}: {len(rows)}행")

            for ri, row in enumerate(rows):
                cells = list(iter_row_cells(row))
                print(f"  │  행{ri+1}: {len(cells)}셀")

                for ci, tc in enumerate(cells):
                    paras = list(iter_cell_paras(tc))
                    blips = [el for el in tc.iter() if local(el.tag) == 'blip']
                    print(f"  │    셀{ci+1}: {len(paras)}단락, blip={len(blips)}개")

                    for pi, para in enumerate(paras):
                        lines = para_lines(para)
                        joined = repr('\n'.join(lines))
                        if len(joined) > 82:
                            joined = joined[:79] + "…'"
                        print(f"  │      단락{pi+1}: {joined}")

            print(f"  └──")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python docx_diag.py <파일.docx>")
        sys.exit(1)
    diag(sys.argv[1])
