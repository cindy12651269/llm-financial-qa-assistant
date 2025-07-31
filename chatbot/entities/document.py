from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Document:
    """Class for storing a piece of text and associated metadata, including financial-specific fields."""

    page_content: str
    """The main text content of the document."""
    metadata: dict = field(default_factory=dict)
    """Arbitrary metadata about the document content.
    Financial-specific fields may include:
        - source (str): The original source of the document.
        - fiscal_year (str): Fiscal year of the report, e.g. '2023'.
        - report_type (str): Type of financial document, e.g. '10-K', 'Earnings Call'.
        - organization (str): Company or institution that published the document.
        - sector (str): Industry sector, e.g. 'Technology', 'Healthcare'.
    """
    type: Literal["Document"] = "Document"

    def set_finance_metadata(
        self,
        source: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        report_type: Optional[str] = None,
        organization: Optional[str] = None,
        sector: Optional[str] = None,
    ):
        """
        Set financial-specific metadata fields into the document metadata dictionary.

        Args:
            source (str, optional): Origin or URL of the document.
            fiscal_year (str, optional): Fiscal year for financial data.
            report_type (str, optional): Type of financial report.
            organization (str, optional): The company or institution related to the document.
            sector (str, optional): Industry or sector classification.
        """
        self.metadata.update(
            {
                "source": source,
                "fiscal_year": fiscal_year,
                "report_type": report_type,
                "organization": organization,
                "sector": sector,
            }
        )
