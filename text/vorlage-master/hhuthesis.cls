\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hhuthesis}[2021/06/28 HHU Thesis LaTeX document class v1.20]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % Options fallback.
\ProcessOptions\relax

\LoadClass[11pt,a4paper,twoside,titlepage]{article}


%%% Define user-settings commands

%% Cannot use \startdate somehow
\newcommand{\begindate}[1]{\renewcommand{\@begin}{#1}}
\newcommand{\@begin}{}
\newcommand{\duedate}[1]{\renewcommand{\@due}{#1}}
\newcommand{\@due}{}

\newcommand{\firstexaminer}[1]{\renewcommand{\@fstexaminer}{#1}}
\newcommand{\@fstexaminer}{}
\newcommand{\secondexaminer}[1]{\renewcommand{\@sndexaminer}{#1}}
\newcommand{\@sndexaminer}{}

\newcommand{\graduationtype}[1]{\renewcommand{\@degree}{#1}}
\newcommand{\@degree}{Bachelor}

\newcommand{\subject}[1]{\renewcommand{\@subject}{#1}}
\newcommand{\@subject}{Informatik}

\newcommand{\subtitle}[1]{\renewcommand{\@subtitle}{#1}}
\newcommand{\@subtitle}{}

%% Allows switching between colour and black/white printing
\newif\ifprintblackwhite
\printblackwhitefalse

\newcommand{\blackwhiteprint}{
  \printblackwhitetrue
  \pgfplotsset{cycle list name=hhubwcycle}
  \pgfplotsset{%
    /pgfplots/bar cycle list/.style={/pgfplots/cycle list name=hhubwbars}
  }
}


%%% Necessary packages

%% Allows direct insertion of umlauts (ä, ö, ü instead of \"a, \"o, \"u)
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}

\RequirePackage[nottoc]{tocbibind}
\RequirePackage{iflang} % Needed for language dependend distinctions

\RequirePackage{xcolor}
%% HHU Corporate Design colours
\definecolor{hhublue}{HTML}{006ab3} % Blau
\definecolor{hhudarkblue}{HTML}{003964} % Dunkelblau
\definecolor{hhuiceblue}{HTML}{b5cbd6} % Eisblau
\definecolor{hhucyan}{HTML}{57bab1} % Türkis
\definecolor{hhugreen}{HTML}{8cb110} % Grün
\definecolor{hhuorange}{HTML}{ee7f00} % Orange
\definecolor{hhured}{HTML}{be0a26} % Rot

\RequirePackage{ifthen}

\RequirePackage{tikz}
\RequirePackage{pgf,pgfplots}
%% Setup of colour cylce lists for pgfplots
\pgfplotscreateplotcyclelist{hhucolorcycle}{
  solid,hhublue,every mark/.append style={solid},mark=*\\
  dashed,hhured,every mark/.append style={solid},mark=square*\\
  dashdotted,hhugreen,every mark/.append style={solid},mark=diamond*\\
  dotted,hhuorange,every mark/.append style={solid},mark=triangle*\\
  densely dashed,hhucyan,every mark/.append style={solid},mark=star\\
  loosely dashed,hhublue,every mark/.append style={solid},mark=o\\
  loosely dotted,hhured,every mark/.append style={solid},mark=square\\
  densely dashed,hhugreen,every mark/.append style={solid},mark=diamond\\
  loosely dashdotted,hhuorange,every mark/.append style={solid},mark=triangle\\
  densely dotted,hhucyan,every mark/.append style={solid},mark=star\\
}
\pgfplotscreateplotcyclelist{hhubwcycle}{ % For black/white print
  solid,every mark/.append style={solid,fill=gray},mark=*\\
  dashed,every mark/.append style={solid,fill=gray},mark=square*\\
  dashdotted,every mark/.append style={solid,fill=gray},mark=diamond*\\
  dotted,every mark/.append style={solid,fill=gray},mark=triangle*\\
  densely dashed,every mark/.append style={solid},mark=star\\
  loosely dashed,every mark/.append style={solid},mark=o\\
  loosely dotted,every mark/.append style={solid},mark=square\\
  densely dashed,every mark/.append style={solid},mark=diamond\\
  loosely dashdotted,every mark/.append style={solid},mark=triangle\\
  densely dotted,every mark/.append style={solid},mark=star\\
}
%% The same for bar plots
\pgfplotscreateplotcyclelist{hhucolorbars}{
  {fill=hhublue!30!white, draw=hhublue, text=hhublue},
  {fill=hhured!30!white, draw=hhured, text=hhured},
  {fill=hhugreen!30!white, draw=hhugreen, text=hhugreen},
  {fill=hhuorange!30!white, draw=hhuorange, text=hhuorange},
  {fill=hhucyan!30!white, draw=hhucyan, text=hhucyan}
}
\pgfplotscreateplotcyclelist{hhubwbars}{
  {fill=black!75!white, draw=black},
  {fill=black!10!white, draw=black},
  {fill=black!65!white, draw=black},
  {fill=black!30!white, draw=black},
  {fill=black!50!white, draw=black}
}
%% Default colour cycle list
%% The colour cycle lists are further defined below.
%% Note that calling \blackwhiteprint switches this to hhubwcycle.
\pgfplotsset{cycle list name=hhucolorcycle}
\pgfplotsset{%
  /pgfplots/bar cycle list/.style={/pgfplots/cycle list name=hhucolorbars}
}



%%% Formatting and Typesetting

%% Include helvet for title page.
\RequirePackage[scaled=0.95]{helvet}

\pagestyle{headings}

\setlength{\textwidth}{15cm}

\setlength{\oddsidemargin}{10mm}
\setlength{\evensidemargin}{0mm}
\setlength{\marginparwidth}{20mm}

\setlength{\baselineskip}{15pt}


\parindent0mm
\parskip9pt

\RequirePackage{caption}
\captionsetup{width=0.9\textwidth,labelfont=bf,font=small}

% Adjust ToC spacing
\RequirePackage{tocloft}
\renewcommand\cftsecafterpnum{\vskip10pt}
\renewcommand\cftsubsecafterpnum{\vskip5pt}
\renewcommand\cftsubsubsecafterpnum{\vskip5pt}

%% Allow use of \frontmatter, \mainmatter, \appendix, and \backmatter
\newcommand{\frontmatter}{
  \cleardoublepage
  \pagenumbering{roman}
}

\newcommand{\mainmatter}{
  \cleardoublepage
  \pagenumbering{arabic}
}

\renewcommand{\appendix}{
  \clearpage
  {\LARGE\bf \appendixpagename}%
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\@Alph\c@section}
  \let\section\appendix@section
}
\newcommand{\appendixpagename}{\iflanguage{ngerman}{Anhang}{Appendices}}
\newcommand{\appendix@section}[2][\@nil]{%
  \refstepcounter{section}%
  \orig@section*{\thesection\hspace{1em}#2}%
  \def\tmp{#1}%
  \ifx\tmp\@nnil% % Puts the optional short title into the ToC if not empty
    \addcontentsline{toc}{section}{\appendixname{}~\thesection\hspace{1em}#2}%
  \else%
    \addcontentsline{toc}{section}{\appendixname{}~\thesection\hspace{1em}#1}%
  \fi%
  \markboth{\thesection\hspace{1em}\MakeUppercase{#2}}{}%
}
\let\orig@section\section

\newcommand{\backmatter}{
  \clearpage
  \let\section\orig@section
  \setcounter{secnumdepth}{0}
}

%% Define link colors for hyperref package
\newcommand{\sethyperrefhhucolors}{%
  \hypersetup{
    linkcolor=hhublue,
    linkbordercolor=hhublue,
    menucolor=hhublue,
    menubordercolor=hhublue,
    citecolor=hhublue,
    citebordercolor=hhublue,
    filecolor=hhublue,
    filebordercolor=hhublue,
    urlcolor=hhublue,
    urlbordercolor=hhublue
  }
}

%% Define PDF meta data
\newcommand{\sethyperrefpdfinfos}{
  \hypersetup{
    pdfinfo={
      Title={\@title},
      Author={\@author},
      Subject={\@degree's Thesis}
    }
  }
}

%% Adapt lstlisting display
\newcommand{\lstkeyword}[1]{%
  \ifprintblackwhite\color{black}\else\color{hhublue}\fi\bfseries#1}
\newcommand{\lststring}[1]{%
  \ifprintblackwhite\color{black}\else\color{hhudarkblue}\fi#1}
\newcommand{\setlstlistingstyle}{%
  \lstset{
    basicstyle=\small\ttfamily,
    numbers=left, numberstyle=\footnotesize,
    keywordstyle=\lstkeyword,
    commentstyle=\color{gray}\itshape,
    stringstyle=\lststring,
    showstringspaces=false,
    frame=tb,
  }%
  \iflanguage{ngerman}{\renewcommand{\lstlistingname}{Quellcode}}{}%
  \renewcommand*\thelstnumber{{\the\value{lstnumber}}:}%
  \let\old@lstlistoflistings\lstlistoflistings%
  \renewcommand{\lstlistoflistings}{%
    \iflanguage{ngerman}%
      {\renewcommand{\lstlistlistingname}{Quellcodeverzeichnis}}%
      {\renewcommand{\lstlistlistingname}{List of Listings}}%
    \addcontentsline{toc}{section}{\lstlistlistingname}%
    \old@lstlistoflistings%
  }
}

%% Adapt styling of algorithms
\newcommand{\setalgorithmstyle}{%
  \algrenewcommand\ALG@beginalgorithmic{\small}
  \iflanguage{ngerman}{\floatname{algorithm}{Algorithmus}}{}%
  \let\old@listofalgorithms\listofalgorithms%
  \renewcommand{\listofalgorithms}{%
    \iflanguage{ngerman}%
      {\renewcommand{\listalgorithmname}{Algorithmenverzeichnis}}{}%
    \addcontentsline{toc}{section}{\listalgorithmname}%
    \old@listofalgorithms%
  }
}


%%% Setup of title page

\RequirePackage{graphicx}

\newcommand{\@makeheader}{
  %% "Header"
  \vspace*{-5.46cm}
  \hspace{-13.0mm}
  \raisebox{-15mm}{
    \begin{minipage}[b]{92mm}
      \ifprintblackwhite
        \includegraphics[width=92mm]{fig/hhu-logo-black}%
      \else
        \includegraphics[width=92mm]{fig/hhu-logo-rgb}%
      \fi
    \end{minipage}%
  }%
  \hfill%
  \begin{minipage}[c]{6.8cm}
    \begin{center}
      \fontfamily{\sfdefault}\selectfont
      {\large \textsc{Institut f\"ur Informatik \linebreak}}
      {\small\centering Softwaretechnik und Programmiersprachen\linebreak}
      {\scriptsize Universit\"atsstr. 1 \hspace{7mm} D--40225 D\"usseldorf}
    \end{center}
  \end{minipage}
}

\newcommand\@maketitle{%
  \vspace{3cm}
  \begin{center}
    {\fontfamily{\sfdefault}\selectfont \Huge \textbf \@title \par}%
    \if\relax\detokenize{\@subtitle}\relax
    \else
      {\fontfamily{\sfdefault}\selectfont \huge \textbf \@subtitle \par}%
    \fi%
    \vfill
    \begin{otherlanguage}{ngerman}
      {{\Large \@degree{}arbeit}\par
        im Studiengang \@subject \\
        zur Erlangung des akademischen Grades \par
        {\Large\@degree{} of Science
          (\ifthenelse{\equal{\@degree}{Master}}{M}{B}.Sc.) \par}
        \par}%
      \vspace{20mm}
      {vorgelegt von\\\Large \textbf \@author \par}%
      \vfill
      \begin{tabular}{ll}
        Beginn der Arbeit: & \@begin \\
        Abgabe der Arbeit: & \@due \\
        \\
        Erstgutachter:     & \@fstexaminer \\
        Zweitgutachter:    & \@sndexaminer \\
      \end{tabular}
    \end{otherlanguage}
  \end{center}
}

\if@titlepage
  \renewcommand\maketitle{
    \begin{titlepage}%
      \let\footnotesize\small
      \let\footnoterule\relax
      \let\footnote\thanks
      \null\vfil
      \vskip 60\p@
      \@makeheader
      \@maketitle
      \@thanks
      \vfil\null
      \thispagestyle{empty}
    \end{titlepage}%
    \setcounter{footnote}{0}%
    % \global\let\thanks\relax
    % \global\let\maketitle\relax
    % \global\let\@thanks\@empty
    % \global\let\@date\@empty
    % \global\let\@title\@empty
    % \global\let\title\relax
    % \global\let\author\relax
    % \global\let\date\relax
    % \global\let\and\relax
    % \global\let\@begin\relax
    % \global\let\@due\relax
    % \global\let\@fstexaminer\relax
    % \global\let\@sndexaminer\relax
  }
\else
  \renewcommand\maketitle{\par
    \@makeheader
    \begingroup
      \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
      \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
      \long\def\@makefntext##1{\parindent 1em\noindent
              \hb@xt@1.8em{%
                  \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
      \if@twocolumn
        \ifnum \col@number=\@ne
          \@maketitle
        \else
          \twocolumn[\@maketitle]%
        \fi
      \else
        \newpage
        \global\@topnum\z@   % Prevents figures from going at top of page.
        \@maketitle
      \fi
      \thispagestyle{plain}\@thanks
    \endgroup
    \setcounter{footnote}{0}%
    % \global\let\thanks\relax
    % \global\let\maketitle\relax
    % \global\let\@maketitle\relax
    % \global\let\@thanks\@empty
    % \global\let\@date\@empty
    % \global\let\@title\@empty
    % \global\let\title\relax
    % \global\let\author\relax
    % \global\let\date\relax
    % \global\let\and\relax
    % \global\let\@begin\relax
    % \global\let\@due\relax
    % \global\let\@fstexaminer\relax
    % \global\let\@sndexaminer\relax
  }
\fi

%% Renew titlepage to not reset page counter and to not fiddle with page style
\if@compatibility
\renewenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\renewenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        % \setcounter{page}\@ne
     \fi
    }
\fi

%% Redesign Abstract
\if@titlepage
  \renewenvironment{abstract}{%
      \titlepage
      \@beginparpenalty\@lowpenalty
      \begin{center}%
        \bfseries \large \abstractname
        \@endparpenalty\@M
      \end{center}}%
     {\par\endtitlepage}
\else
  \renewenvironment{abstract}{%
      \if@twocolumn
        \section*{\abstractname}%
      \else
        \small
        \begin{center}%
          {\bfseries \large \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
      \fi}
      {\if@twocolumn\else\endquotation\fi}
\fi

%% Design official statement of independend and autonomous work
\newcommand{\@makedeclaration}{%
  \@beginparpenalty\@lowpenalty
  \begin{otherlanguage}{ngerman}
    \begin{center}%
      \bfseries \large Selbstst\"andigkeitserkl\"arung
      \@endparpenalty\@M
    \end{center}%

    Hiermit versichere ich
    die vorliegende \@degree{}arbeit selbstst\"andig verfasst und keine anderen
    als die angegebenen Quellen und Hilfsmittel benutzt zu haben.
    Alle Stellen, die aus den Quellen entnommen wurden,
    sind als solche kenntlich gemacht worden.
    Diese Arbeit hat in gleicher oder \"ahnlicher Form
    noch keiner Pr\"ufungsbeh\"orde vorgelegen.

    \vspace{2cm}

    \begin{centering}
      \begin{tabular}{lc}
      D\"usseldorf, den \@due \hspace*{1.8cm} & \underline{\hspace{6cm}}\\
      & \@author
      \end{tabular}
    \end{centering}
  \end{otherlanguage}
}

\if@titlepage
  \newcommand\makedeclaration{
    \begin{titlepage}%
      \null\vfil
      \vskip 60\p@
      \@makedeclaration
      \vfil\null
    \end{titlepage}%
  }
\else
  \newcommand\makedeclaration{\par
    \begingroup
      \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
      \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
      \long\def\@makefntext##1{\parindent 1em\noindent
              \hb@xt@1.8em{%
                  \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
      \if@twocolumn
        \ifnum \col@number=\@ne
          \@makedeclaration
        \else
          \twocolumn[\@makedeclaration]%
        \fi
      \else
        \newpage
        \global\@topnum\z@   % Prevents figures from going at top of page.
        \@makedeclaration
      \fi
      \thispagestyle{plain}\@thanks
    \endgroup
  }
\fi

%% design Acknowledgements
\newcommand{\acknowledgementsname}{%
  \iflanguage{ngerman}{Danksagung}{Acknowledgements}%
}
\if@titlepage
  \newenvironment{acknowledgements}{%
      \titlepage
      \@beginparpenalty\@lowpenalty
      \begin{center}%
        \bfseries \large \acknowledgementsname
        \@endparpenalty\@M
      \end{center}}%
     {\par\endtitlepage\cleardoublepage}
\else
  \newenvironment{acknowledgements}{%
      \if@twocolumn
        \section*{\acknowledgementsname}%
      \else
        \small
        \begin{center}%
          {\bfseries \large \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
      \fi}
      {\if@twocolumn\else\endquotation\fi\cleardoublepage}
\fi


%%% Simple command to set up frontmatter

\newcommand{\makefrontmatter}{
  \maketitle
  \cleardoublepage

  \makedeclaration % Declaration of Authorship
  \cleardoublepage

  \begin{abstract}
    \input{abstract.tex}
  \end{abstract}
  \cleardoublepage

  % Ensure cleverref knows how to talk about listings.
  \iflanguage{ngerman}{
    \crefname{listing}{Quellcode}{Quellcodes}
    \Crefname{listing}{Quellcode}{Quellcodes}
  }{}
}
