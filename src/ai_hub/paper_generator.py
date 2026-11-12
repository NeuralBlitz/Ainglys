"""
Automated Paper Generation from Experiments

Generates arXiv-ready research papers from experimental results.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PaperSection:
    """Represents a paper section."""

    title: str
    content: str
    subsections: Optional[List["PaperSection"]] = None


class SectionCollection(list):
    """Ordered sections with legacy title-based lookups."""

    def __contains__(self, item):
        if isinstance(item, str):
            return any(section.title == item for section in self)
        return super().__contains__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            for section in self:
                if section.title == item:
                    return section.content
            raise KeyError(item)
        return super().__getitem__(item)


class PaperGenerator:
    """
    Generate research papers automatically from experimental results.

    Creates LaTeX documents with:
    - Mathematical foundations
    - Experimental results tables
    - Figures and diagrams
    - Citations and references
    """

    def __init__(self, title: str = "Research Manuscript", authors: Optional[List[str]] = None,
                 project_name: str = "Research Project"):
        self.title = title
        self.authors = list(authors) if authors else ["Author not specified"]
        self.project_name = project_name
        self.sections: List[PaperSection] = SectionCollection()
        self.references: List[str] = []
        self.figures: List[Dict] = []
        self.tables: List[Dict] = []

    def generate_latex(self) -> str:
        """Generate a LaTeX draft from explicitly supplied manuscript content."""
        if not self.sections:
            raise ValueError("Cannot generate a manuscript without sections")
        latex = []

        # Preamble
        latex.append(r"\documentclass[11pt,a4paper]{article}")
        latex.append(r"\usepackage[utf8]{inputenc}")
        latex.append(r"\usepackage{amsmath,amssymb,amsthm}")
        latex.append(r"\usepackage{graphicx}")
        latex.append(r"\usepackage{booktabs}")
        latex.append(r"\usepackage{algorithm}")
        latex.append(r"\usepackage{algpseudocode}")
        latex.append(r"\usepackage{tikz}")
        latex.append(r"\usepackage{pgfplots}")
        latex.append(r"\pgfplotsset{compat=1.17}")
        latex.append("")

        # Title and authors
        latex.append(r"\title{" + self.title + "}")
        latex.append(r"\author{" + r" \and ".join(self.authors) + "}")
        latex.append(r"\date{" + datetime.now().strftime("%B %Y") + "}")
        latex.append("")
        latex.append(r"\begin{document}")
        latex.append(r"\maketitle")
        latex.append("")

        # Abstract
        abstract = self._generate_abstract()
        latex.append(r"\begin{abstract}")
        latex.append(abstract)
        latex.append(r"\end{abstract}")
        latex.append("")

        # Sections
        for section in self.sections:
            latex.extend(self._render_section(section))

        # Figures
        if self.figures:
            latex.append(r"\newpage")
            latex.append(r"\section{Figures}")
            for i, fig in enumerate(self.figures, 1):
                latex.append(self._render_figure(fig, i))

        # Tables
        if self.tables:
            latex.append(r"\section{Tables}")
            for i, table in enumerate(self.tables, 1):
                latex.append(self._render_table(table, i))

        # References
        if self.references:
            latex.append(r"\bibliographystyle{plain}")
            latex.append(r"\begin{thebibliography}{99}")
            for i, ref in enumerate(self.references, 1):
                latex.append(f"\\bibitem{{{i}}} {ref}")
            latex.append(r"\end{thebibliography}")

        latex.append(r"\end{document}")

        return "\n".join(latex)

    def _generate_abstract(self) -> str:
        """Avoid making empirical or novelty claims absent supporting evidence."""
        for section in self.sections:
            if section.title.casefold() == "abstract":
                return section.content
        return (
            f"This manuscript documents the research project {self.project_name}. "
            "Any performance comparisons, novelty claims, and conclusions require "
            "independent supporting evidence from the accompanying experiments."
        )

    def _render_section(self, section: PaperSection, level: int = 1) -> List[str]:
        """Render a section to LaTeX."""
        latex = []
        cmd = (
            "section" if level == 1 else "subsection" if level == 2 else "subsubsection"
        )
        latex.append(f"\\{cmd}{{{section.title}}}")
        latex.append("")
        latex.append(section.content)
        latex.append("")

        if section.subsections:
            for sub in section.subsections:
                latex.extend(self._render_section(sub, level + 1))

        return latex

    def _render_figure(self, fig: Dict, number: int) -> str:
        """Render a figure environment."""
        return f"""
\\begin{{figure}}[htbp]
\\centering
\\caption{{{fig.get("caption", f"Figure {number}")}}}
\\label{{fig:{number}}}
\\end{{figure}}
"""

    def _render_table(self, table: Dict, number: int) -> str:
        """Render a table environment."""
        cols = table.get("columns", [])
        rows = table.get("rows", [])

        latex = [f"\\begin{{table}}[htbp]", r"\centering"]
        latex.append(f"\\caption{{{table.get('caption', f'Table {number}')}}}")
        latex.append(f"\\label{{tab:{number}}}")

        col_spec = "c" * len(cols)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append(r"\toprule")
        latex.append(" & ".join(cols) + r" \\")
        latex.append(r"\midrule")

        for row in rows:
            latex.append(" & ".join(str(x) for x in row) + r" \\")

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        return "\n".join(latex)

    def add_section(self, title: str, content: str):
        """Add or replace a named section while preserving insertion order."""
        if not title or not isinstance(content, str):
            raise ValueError("A nonempty title and string content are required")
        for i, section in enumerate(self.sections):
            if section.title == title:
                self.sections[i] = PaperSection(title, content)
                return
        self.sections.append(PaperSection(title, content))

    def add_figure(self, image_path: str, caption: str):
        """Record a figure, leaving image verification to the publishing process."""
        self.figures.append({"path": image_path, "caption": caption})

    def add_table(self, dataframe, caption: str):
        """Record real data rows without inventing benchmark values."""
        if not hasattr(dataframe, "columns") or not hasattr(dataframe, "itertuples"):
            raise TypeError("Expected a pandas-like DataFrame")
        self.tables.append({
            "columns": [str(col) for col in dataframe.columns],
            "rows": [list(row) for row in dataframe.itertuples(index=False, name=None)],
            "caption": caption,
        })

    def add_introduction(self, content: str):
        """Add introduction section."""
        self.sections.append(PaperSection("Introduction", content))

    def add_methodology(
        self, mathematical_formulation: str, algorithm_steps: List[str]
    ):
        """Add methodology section with math and algorithms."""
        content = f"""
We formalize the problem as follows. Given input space ${{\\mathcal{{X}}}}$ and output space 
${{\\mathcal{{Y}}}}$, we seek to learn a function $f: \\mathcal{{X}} \\rightarrow \\mathcal{{Y}}$.

{mathematical_formulation}

The algorithm proceeds as follows:

\\begin{{algorithm}}
\\caption{{{self.project_name} Algorithm}}
\\begin{{algorithmic}}[1]
"""
        for i, step in enumerate(algorithm_steps, 1):
            content += f"\\State {step}\n"

        content += r"\end{algorithmic}" + "\n" + r"\end{algorithm}"

        self.sections.append(PaperSection("Methodology", content))

    def add_experiments(self, results: Dict[str, Any]):
        """Include only supplied experimental measurements, not fabricated baselines."""
        metrics = results.get("metrics", {})
        content = "Experimental results provided for this run are reported below."
        if isinstance(metrics, dict) and metrics:
            # Supports both {metric: value} and {model_name: {metric: value}}.
            if all(isinstance(v, dict) for v in metrics.values()):
                fields = sorted({key for values in metrics.values() for key in values})
                rows = [[name] + [values.get(key, "") for key in fields]
                        for name, values in metrics.items()]
                self.tables.append({
                    "caption": "Supplied experimental measurements",
                    "columns": ["Model"] + fields,
                    "rows": rows,
                })
            else:
                self.tables.append({
                    "caption": "Supplied experimental measurements",
                    "columns": ["Metric", "Value"],
                    "rows": [[name, value] for name, value in metrics.items()],
                })
        else:
            content += " No measurements were supplied; no performance claims are made."
        self.sections.append(PaperSection("Experiments", content))

    def add_conclusion(self):
        """Add conclusion section."""
        content = (
            f"This manuscript describes {self.project_name}. Its claims should be "
            "evaluated using reproducible experiments, baselines and independent review."
        )
        self.sections.append(PaperSection("Conclusion", content))

    def save(self, filepath: str):
        """Save LaTeX document to file."""
        latex = self.generate_latex()
        with open(filepath, "w") as f:
            f.write(latex)
        print(f"✅ Paper saved to: {filepath}")


class ExperimentToPaper:
    """Convert experimental results to paper automatically."""

    def __init__(self, experiment_tracker):
        self.tracker = experiment_tracker

    def generate_from_experiment(
        self, experiment_name: str, run_id: Optional[str] = None
    ) -> PaperGenerator:
        """Generate paper from experiment results."""
        # Get experiment data
        runs = self.tracker.get_run_history()

        if not runs:
            raise ValueError("No runs found")

        run = runs[-1]  # Use most recent run
        if run_id:
            run = next((r for r in runs if r.run_id == run_id), run)

        # Create paper
        paper = PaperGenerator(
            title=f"{experiment_name}: A Novel Approach",
            authors=["Anonymous Author"],
            project_name=experiment_name,
        )

        # Add sections
        paper.add_introduction(
            f"In this work, we investigate {experiment_name} for machine learning tasks. "
            "Our approach addresses key challenges in the field."
        )

        # Add methodology with hyperparameters
        params_str = "\\begin{align}\n"
        for key, value in run.params.items():
            params_str += f"{key} &= {value} \\\\\n"
        params_str += "\\end{align}"

        paper.add_methodology(
            mathematical_formulation=params_str,
            algorithm_steps=[
                "Initialize model parameters",
                "For each epoch, process training data",
                "Update parameters using gradient descent",
                "Evaluate on validation set",
            ],
        )

        # Add experiments
        paper.add_experiments({"metrics": run.metrics})

        # Add conclusion
        paper.add_conclusion()

        return paper


# Export
__all__ = ["PaperGenerator", "PaperSection", "ExperimentToPaper"]
