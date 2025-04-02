from typing import List
from langchain_core.output_parsers import BaseOutputParser
import re
from loguru import logger


class TaxonomyOutputParser(BaseOutputParser):
    pattern: str
    relation: str
    match_last: bool = True

    def parse(self, text: str, group: str = "") -> list[dict]:
        matches = re.findall(self.pattern, text)

        if len(matches) == 0:
            logger.error(f"Parse failed with {self.pattern} on {text}")
            return None
        if self.match_last:
            match = matches[-1]
        else:
            match = matches[0]

        result = []
        for line in match[0].split("\n"):
            if self.relation not in line:
                continue
            elements = line.split(self.relation)
            child = elements[0]
            parent = elements[1]
            result.append(
                {"group": group, "child": child.strip(), "parent": parent.strip()}
            )

        return result
