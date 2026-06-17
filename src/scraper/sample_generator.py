"""Sample legal document generator for pipeline testing.

Produces realistic Omani legal documents in Arabic and English so that the
entire GraphRAG pipeline can be developed and demonstrated without relying
on live scraping.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from src.config.settings import settings

logger = logging.getLogger(__name__)


class SampleGenerator:
    """Generates synthetic but realistic Omani legal documents."""

    DOCUMENT_TYPES = [
        "Royal Decree",
        "Ministerial Decision",
        "Ministerial Order",
        "Executive Decision",
    ]

    ISSUERS = [
        "Ministry of Legal Affairs",
        "Ministry of Finance",
        "Ministry of Labour",
        "Central Bank of Oman",
        "Supreme Council for Planning",
    ]

    TOPICS_POOL = [
        "Taxation",
        "Labour Law",
        "Omanization",
        "Maritime Law",
        "Judicial Fees",
        "Commercial Registration",
        "Foreign Investment",
        "Environmental Protection",
        "Intellectual Property",
        "Banking Regulation",
        "Insurance",
        "Real Estate",
        "Customs Duties",
        "Civil Service",
        "Social Security",
    ]

    ARABIC_TOPICS = {
        "Taxation": "الضرائب",
        "Labour Law": "قانون العمل",
        "Omanization": "التعمين",
        "Maritime Law": "القانون البحري",
        "Judicial Fees": "رسوم القضاء",
        "Commercial Registration": "السجل التجاري",
        "Foreign Investment": "الاستثمار الأجنبي",
        "Environmental Protection": "حماية البيئة",
        "Intellectual Property": "الملكية الفكرية",
        "Banking Regulation": "تنظيم المصارف",
        "Insurance": "التأمين",
        "Real Estate": "العقارات",
        "Customs Duties": "الرسوم الجمركية",
        "Civil Service": "الخدمة المدنية",
        "Social Security": "الضمان الاجتماعي",
    }

    def __init__(self, output_dir: Path = None) -> None:
        self.output_dir = output_dir or settings.SAMPLE_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _random_date(self) -> str:
        """Generate a random date within the last 20 years."""
        days_back = random.randint(0, 365 * 20)
        date = datetime.now() - timedelta(days=days_back)
        return date.strftime("%Y-%m-%d")

    def _generate_english_content(
        self, title: str, doc_type: str, number: str, topics: List[str]
    ) -> str:
        """Generate realistic English legal markdown content."""
        lines = [
            f"# {title}",
            "",
            f"**Type:** {doc_type}",
            f"**Number:** {number}",
            f"**Date:** {self._random_date()}",
            "",
            "## Preamble",
            "",
            f"In accordance with the provisions of the Basic Law of the State, and in pursuance of public interest, this {doc_type} has been issued.",
            "",
            "## Article 1 - Objectives",
            "",
            f"The provisions of this {doc_type} aim to regulate matters relating to {', '.join(topics)}. ",
            "All persons and entities subject to the jurisdiction of the Sultanate of Oman shall comply with the provisions hereof.",
            "",
            "## Article 2 - Definitions",
            "",
            f"For the purposes of this {doc_type}, the following terms shall have the meanings assigned thereto:",
            "",
            "- **Competent Authority**: The ministry or government entity entrusted with implementation.",
            "- **Regulated Activity**: Any activity falling within the scope of this legislation.",
            "",
            "## Article 3 - Scope of Application",
            "",
            "This legislation applies to all natural and legal persons conducting activities within the territory of the Sultanate of Oman.",
            "",
            "## Article 4 - Penalties",
            "",
            "Any person violating the provisions of this legislation shall be subject to the penalties prescribed by the relevant laws and regulations.",
            "",
            "## Article 5 - Repeals",
            "",
            "Any provision contrary to or inconsistent with the provisions of this legislation is hereby repealed.",
        ]
        return "\n".join(lines)

    def _generate_arabic_content(
        self, title_ar: str, doc_type_ar: str, number: str, topics_ar: List[str]
    ) -> str:
        """Generate realistic Arabic legal markdown content."""
        lines = [
            f"# {title_ar}",
            "",
            f"**النوع:** {doc_type_ar}",
            f"**الرقم:** {number}",
            f"**التاريخ:** {self._random_date()}",
            "",
            "## الديباجة",
            "",
            f"بناءً على أحكام القانون الأساسي للدولة، ومصلحة الرأي العام، تم إصدار هذا {doc_type_ar}.",
            "",
            "## المادة الأولى - الأهداف",
            "",
            f"تهدف أحكام هذا {doc_type_ar} إلى تنظيم الأمور المتعلقة بـ {', '.join(topics_ar)}.",
            "",
            "## المادة الثانية - التعاريف",
            "",
            "لأغراض هذا النص التشريعي، يكون للمصطلحات التالية المعاني المخصصة لها:",
            "",
            "- **الجهة المختصة**: الوزارة أو الجهة الحكومية المكلفة بالتنفيذ.",
            "- **النشاط المنظم**: أي نشاط يدخل في نطاق هذا التشريع.",
            "",
            "## المادة الثالثة - نطاق التطبيق",
            "",
            "ينطبق هذا التشريع على جميع الأشخاص الطبيعيين والاعتباريين الذين يمارسون أنشطة داخل أراضي سلطنة عمان.",
            "",
            "## المادة الرابعة - العقوبات",
            "",
            "يكون كل شخص يخالف أحكام هذا التشريع عرضة للعقوبات المنصوص عليها في القوانين واللوائح ذات الصلة.",
            "",
            "## المادة الخامسة - الإلغاء",
            "",
            "يلغى كل نص يتعارض مع أحكام هذا التشريع.",
        ]
        return "\n".join(lines)

    def generate_document(self, index: int) -> Dict[str, Any]:
        """Generate a single synthetic legal document."""
        doc_type = random.choice(self.DOCUMENT_TYPES)
        year = random.randint(2000, 2024)
        serial = random.randint(1, 150)
        number = f"{serial}/{year}"
        issuer = random.choice(self.ISSUERS)

        topics = random.sample(self.TOPICS_POOL, k=random.randint(2, 4))
        topics_ar = [self.ARABIC_TOPICS.get(t, t) for t in topics]

        title = f"{doc_type} No. {number} regarding {', '.join(topics)}"
        title_ar = f"{doc_type} رقم {number} بشأن {', '.join(topics_ar)}"

        doc_id = f"sample-doc-{index:04d}"

        return {
            "id": doc_id,
            "title": title,
            "document_type": doc_type,
            "number": number,
            "issue_date": self._random_date(),
            "issuer": issuer,
            "source_url": f"https://qanoon.om/sample/{doc_id}",
            "contentEn": self._generate_english_content(title, doc_type, number, topics),
            "contentAr": self._generate_arabic_content(title_ar, doc_type, number, topics_ar),
            "contentFr": "",
            "raw_markdown": "",
        }

    def generate(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate a batch of sample documents and save them."""
        docs = [self.generate_document(i) for i in range(1, count + 1)]

        # Individual files
        for doc in docs:
            file_path = self.output_dir / f"{doc['id']}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

        # Combined file for batch ingestion
        combined_path = self.output_dir / "sample_documents.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

        logger.info("Generated %d sample documents in %s", len(docs), self.output_dir)
        return docs


def main() -> None:
    """CLI entrypoint for sample generation."""
    import sys

    logging.basicConfig(level=settings.LOG_LEVEL)

    count = 50
    if len(sys.argv) > 1:
        count = int(sys.argv[1])

    generator = SampleGenerator()
    generator.generate(count=count)


if __name__ == "__main__":
    main()
