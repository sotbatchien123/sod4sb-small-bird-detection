"""Build main_springer.tex (Springer Nature journal template, sn-jnl.cls, December 2024 package) from the IEEE main.tex.
The body (sections, tables, figures, equations) is shared; the class, front matter, figure widths (single column)
and back matter (Declarations) change. Re-run after every edit of main.tex:

    python to_springer.py && tectonic -X compile main_springer.tex

sn-jnl.cls and sn-mathphys-num.bst are not redistributed here (LPPL: only together with the LaTeX base system).
Download the template package from https://www.springernature.com/gp/authors/campaigns/latex-author-support and copy
both files next to this script; they are git-ignored.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE/'main.tex').read_bytes().decode('utf-8')
for f in ('sn-jnl.cls', 'sn-mathphys-num.bst'):
    if not (HERE/f).exists():
        print(f'note: {f} missing - copy it from the Springer Nature template package before compiling')

PREAMBLE = r"""\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}% Math and Physical Sciences Numbered Reference Style
\usepackage{iftex}
\ifXeTeX
  % XeLaTeX/Tectonic: Latin Modern = the template's Computer Modern look, with Vietnamese diacritics for author names.
  \usepackage{fontspec}
  \setmainfont{lmroman10}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
    ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
\else
  \usepackage[T5,T1]{fontenc}
  \usepackage[utf8]{inputenc}
\fi
\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{booktabs}%
\usepackage{array}%
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit}

\raggedbottom

\begin{document}
"""

FRONT = r"""\title[Where Should Resolution Be Preserved?]{TITLE}

% Vietnamese order: family name (\sur) first, then given names (\fnm).
\author[1]{\sur{Vũ} \fnm{Quốc Đạt}}\email{qd130506@gmail.com}
\author[1]{\sur{Nguyễn} \fnm{Gia Phúc}}\email{phucoik@gmail.com}
\author*[1]{\sur{Ngô} \fnm{Ngọc Trường Sơn}}\email{truongsonks2006@gmail.com}

\affil*[1]{\orgname{FPT University}, \orgaddress{\city{Da Nang}, \country{Vietnam}}}

\abstract{ABSTRACT}

\keywords{KEYWORDS}

\maketitle
"""

BACK = r"""
\backmatter

\bmhead{Supplementary information}
Per-image predictions, per-epoch metrics and every result table are provided in the code repository
(Code availability).

\section*{Declarations}

\begin{itemize}
\item \textbf{Funding:} [to be completed by the authors].
\item \textbf{Conflict of interest/Competing interests:} The authors declare that they have no known competing
financial interests or personal relationships that could have appeared to influence the work reported in this paper.
\item \textbf{Ethics approval and consent to participate:} Not applicable.
\item \textbf{Consent for publication:} Not applicable.
\item \textbf{Data availability:} The MVA 2023 SOD4SB and Drone2021 images are not redistributed and can be obtained
from the challenge organisers; the audited label manifest is included in the code repository.
\item \textbf{Materials availability:} Not applicable.
\item \textbf{Code availability:} Code, configurations, per-image results and trained weights are available at
\url{https://github.com/sotbatchien123/sod4sb-small-bird-detection}.
\item \textbf{Author contribution:} [to be completed by the authors].
\end{itemize}

\bibliography{refs}
"""


def between(s, a, b):
    i = s.index(a) + len(a); return s[i:s.index(b, i)]


title = between(src, r'\title{', '}\n')
abstract = between(src, r'\begin{abstract}', r'\end{abstract}').strip()
keywords = between(src, r'\begin{IEEEkeywords}', r'\end{IEEEkeywords}').strip()
body = src[src.index(r'\end{IEEEkeywords}') + len(r'\end{IEEEkeywords}'):src.index(r'\bibliographystyle{IEEEtran}')]

# single-column page: column-wide graphics would span the whole text width
body = body.replace(r'\includegraphics[width=\columnwidth]', r'\includegraphics[width=0.75\textwidth]')
body = body.replace(r'\resizebox{\columnwidth}{!}', r'\resizebox{0.85\textwidth}{!}')

front = FRONT.replace('TITLE', title).replace('ABSTRACT', abstract).replace('KEYWORDS', keywords)
out = PREAMBLE + '\n' + front + body.rstrip() + '\n' + BACK + '\n\\end{document}\n'
bad = [l for l in out.splitlines() if 'IEEE' in l and 'IEEE Conf' not in l and 'IEEE/CVF' not in l]
assert not bad, bad[:3]
(HERE/'main_springer.tex').write_bytes(out.encode('utf-8'))
print('main_springer.tex', len(out.splitlines()), 'lines')
