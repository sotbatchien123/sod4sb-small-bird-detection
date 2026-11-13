"""Build main_elsevier.tex (Elsevier elsarticle, journal 'Array') from the IEEE main.tex.
The body (sections, tables, figures, equations) is shared; only the front matter, the class and the back matter
change. Re-run after every edit of main.tex:

    python to_elsevier.py && tectonic -X compile main_elsevier.tex
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE/'main.tex').read_bytes().decode('utf-8').replace('\r\n', '\n')   # git may check out CRLF on Windows

HIGHLIGHTS = [   # Elsevier: 3-5 bullets, at most 85 characters each (spaces included)
    'Exhaustive tile sampling (slicing) lifts YOLOv8-M from 70.2% to 79.3% AP50',
    'SAHI tiled inference hurts a detector that already sees native 4K frames',
    'One WBF of YOLO, flipped YOLO and GFL reaches 84.5% AP50 on validation',
    'YOLO is stronger on 8-16 px birds and GFL on 32-64 px birds',
    'Merging rules are checked on held-out halves of the validation set',
]
for h in HIGHLIGHTS:
    assert len(h) <= 85, (len(h), h)

PREAMBLE = r"""\documentclass[5p,a4paper]{elsarticle}
\usepackage{iftex}
\ifXeTeX
  % XeLaTeX/Tectonic: TeX Gyre Termes is a Times clone with full Vietnamese diacritics for the author names.
  \usepackage{fontspec}
  \setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
    ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
\else
  \usepackage[T5,T1]{fontenc}
  \usepackage[utf8]{inputenc}
\fi
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{xurl}        % URLs may break anywhere (narrow 5p columns)
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit}
\setlength{\emergencystretch}{2.5em}   % natbib ties citations to the previous word; allow looser lines instead of overfull ones

\journal{Array}

\begin{document}
"""


def tex(s):
    return s.replace('%', r'\%')


FRONT = r"""\begin{frontmatter}

\title{TITLE}

\author[fpt]{Vũ Quốc Đạt}
\ead{qd130506@gmail.com}
\author[fpt]{Nguyễn Gia Phúc}
\ead{phucoik@gmail.com}
\author[fpt]{Ngô Ngọc Trường Sơn\corref{cor1}}
\cortext[cor1]{Corresponding author.}
\ead{truongsonks2006@gmail.com}
\author[fpt]{Phan Thị Thu Hồng}
\ead{[email]}
\affiliation[fpt]{organization={FPT University},
            city={Da Nang},
            country={Vietnam}}

\begin{abstract}
ABSTRACT
\end{abstract}

\begin{highlights}
HIGHLIGHTS
\end{highlights}

\begin{keyword}
KEYWORDS
\end{keyword}

\end{frontmatter}
"""

BACK = r"""
\bibliographystyle{elsarticle-num}
\bibliography{refs}
"""


def between(s, a, b):
    i = s.index(a) + len(a); return s[i:s.index(b, i)]


title = between(src, r'\title{', '}\n')
abstract = between(src, r'\begin{abstract}', r'\end{abstract}').strip()
keywords = between(src, r'\begin{IEEEkeywords}', r'\end{IEEEkeywords}').strip()
body = src[src.index(r'\end{IEEEkeywords}') + len(r'\end{IEEEkeywords}'):src.index('% BACKMATTER')]
# Funding, competing interest, generative-AI use, data availability: same declarations as the lecturer's Array paper
backmatter = src[src.index('% BACKMATTER'):src.index(r'\bibliographystyle{IEEEtran}')]
backmatter = backmatter[backmatter.index('\n') + 1:].rstrip() + '\n'

body = body.replace('\\begin{table}[t]\n', '\\begin{table}[t]\n\\small\n')   # journal tables are smaller than body text
front = (FRONT.replace('TITLE', title).replace('ABSTRACT', abstract)
         .replace('HIGHLIGHTS', '\n'.join(r'\item ' + tex(h) for h in HIGHLIGHTS))
         .replace('KEYWORDS', ' \\sep '.join(k.strip() for k in keywords.split(','))))
out = PREAMBLE + '\n' + front + body.rstrip() + '\n\n' + backmatter + BACK + '\n\\end{document}\n'
assert 'IEEE' not in out.replace('IEEE Conf', '').replace('IEEE/CVF', ''), [l for l in out.splitlines() if 'IEEE' in l][:3]
(HERE/'main_elsevier.tex').write_bytes(out.encode('utf-8'))
print('main_elsevier.tex', len(out.splitlines()), 'lines; highlights', [len(h) for h in HIGHLIGHTS])
