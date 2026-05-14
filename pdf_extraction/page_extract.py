from typing import Literal, Union
from pydantic import BaseModel, Field

instructions = """
You are extracting a textbook page into structured page blocks.

text and formulas should be extracted verbatim.

use latex for all math.
Use "$" for inline equation and EquationBlock type for block equations.

some inline equations should be treated as block equations
if there's little text around them.

important: don't skip any text. if something is not possible to
recognize, include a placeholder

Extraction rules:
1) Preserve reading order. The blocks list must match the order a human reads the page.
2) Do NOT include OCR or layout details (no coordinates, fonts, line breaks, or scan artifacts).
3) Prefer fewer, larger TextBlocks over many tiny ones. Group adjacent paragraphs when they belong together.
4) Use LaTeX for all math in EquationBlock.latex.
5) Section headings must be SectionHeadingBlock only; do not include body text in them.
6) FigureBlock.description should explain what the figure conveys conceptually (graphs, curves, relationships),
   not how it looks on the page.
7) TableBlock should capture semantic columns and rows. Include units in column names if shown.
8) Store the running page header (if any) in Page.header.
9) If uncertain, make a best-faith concise extraction; do not invent content.
""".strip()

class PageBlock(BaseModel):
    type: str = Field(
        ...,
        description="Discriminator that identifies which kind of page block this is.",
    )


class SectionHeadingBlock(PageBlock):
    type: Literal["section_heading"] = "section_heading"
    title: str = Field(..., description="The section heading text.")


class TextBlock(PageBlock):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Explanatory prose from the textbook.")


class EquationBlock(PageBlock):
    """
    A mathematical expression written in LaTeX.
    """

    type: Literal["equation"] = "equation"
    latex: str = Field(..., description="The equation in LaTeX format.")
    description: str | None = Field(
        None,
        description="Optional plain-language meaning or interpretation of the equation.",
    )


class FigureBlock(PageBlock):
    type: Literal["figure"] = "figure"
    caption: str | None = Field(
        None, description="Figure caption or label, if present."
    )
    description: str = Field(
        ...,
        description="Conceptual description of what the figure shows and why it matters.",
    )
    figure_number: int = Field(
        ..., description="Figure number as mentioned in the book."
    )


class TableBlock(PageBlock):
    type: Literal["table"] = "table"
    caption: str | None = Field(None, description="Table caption or label, if present.")
    columns: list[str] = Field(..., description="Column headers in reading order.")
    rows: list[list[str]] = Field(..., description="Table rows aligned with columns.")


PageBlockUnion = Union[
    SectionHeadingBlock,
    TextBlock,
    EquationBlock,
    FigureBlock,
    TableBlock,
]


class Page(BaseModel):
    page_number: int = Field(..., description="Printed page number in the textbook.")
    header: str | None = Field(None, description="Running page header text, if any.")
    blocks: list[PageBlockUnion] = Field(
        ..., description="Ordered list of extracted page blocks."
    )

    def print(self):
        print(self.page_number)
        print(self.header)

        for block in self.blocks:
            if block.type == "text":
                print(block.text)

            elif block.type == "equation":
                print(f"$${block.latex}$$")

            elif block.type == "figure":
                # print(block)
                print(block.caption)
                print(block.description)
                print("Fig.", block.figure_number)

            else:
                print(block)

            print()


class PageResponse(BaseModel):
    page: Page
    cost: float
