"""Build main_elsevier.tex (Elsevier elsarticle, journal 'Array') from the IEEE main.tex.
The body (sections, tables, figures, equations) is shared; only the front matter, the class and the back matter
change. Re-run after every edit of main.tex:

    python to_elsevier.py && tectonic -X compile main_elsevier.tex
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE/'main.tex').read_bytes().decode('utf-8')

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
\author[fpt]{Nguyễn Gia Phúc}
\author[fpt]{Ngô Ngọc Trường Sơn\corref{cor1}}
\cortext[cor1]{Corresponding author.}
% \ead{...}   % e-mail of the corresponding author: required by Elsevier before submission
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
\section*{CRediT authorship contribution statement}
% Fill in with the CRediT roles of each author (https://credit.niso.org), e.g.
% \textbf{Vũ Quốc Đạt:} Investigation, Software. \textbf{Nguyễn Gia Phúc:} ... \textbf{Ngô Ngọc Trường Sơn:} ...
\textbf{Vũ Quốc Đạt, Nguyễn Gia Phúc, Ngô Ngọc Trường Sơn:} [to be completed by the authors].

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have
appeared to influence the work reported in this paper.

\section*{Data availability}
Code, configurations, per-image results and trained weights are available at
\url{https://github.com/sotbatchien123/sod4sb-small-bird-detection}. The MVA 2023 SOD4SB and Drone2021 images are
not redistributed and can be obtained from the challenge organisers.

\bibliographystyle{elsarticle-num}
\bibliography{refs}
"""


def between(s, a, b):
    i = s.index(a) + len(a); return s[i:s.index(b, i)]


title = between(src, r'\title{', '}\n')
abstract = between(src, r'\begin{abstract}', r'\end{abstract}').strip()
keywords = between(src, r'\begin{IEEEkeywords}', r'\end{IEEEkeywords}').strip()
body = src[src.index(r'\end{IEEEkeywords}') + len(r'\end{IEEEkeywords}'):src.index(r'\bibliographystyle{IEEEtran}')]

body = body.replace('\\begin{table}[t]\n', '\\begin{table}[t]\n\\small\n')   # journal tables are smaller than body text
front = (FRONT.replace('TITLE', title).replace('ABSTRACT', abstract)
         .replace('HIGHLIGHTS', '\n'.join(r'\item ' + tex(h) for h in HIGHLIGHTS))
         .replace('KEYWORDS', ' \\sep '.join(k.strip() for k in keywords.split(','))))
out = PREAMBLE + '\n' + front + body.rstrip() + '\n' + BACK + '\n\\end{document}\n'
assert 'IEEE' not in out.replace('IEEE Conf', '').replace('IEEE/CVF', ''), [l for l in out.splitlines() if 'IEEE' in l][:3]
(HERE/'main_elsevier.tex').write_bytes(out.encode('utf-8'))
print('main_elsevier.tex', len(out.splitlines()), 'lines; highlights', [len(h) for h in HIGHLIGHTS])
